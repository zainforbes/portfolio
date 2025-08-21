import json
import re
from typing import Any, Dict, List

from langgraph.graph import StateGraph
from googleapiclient.errors import HttpError

from src.core.state_schema import AssistantState
from src.core.llm_client import GeminiClient
from src.agents.gmail_client import GmailClient
from src.agents.calendar_client import GoogleCalendarClient
from src.agents.brave_client import BraveSearchClient
from src.agents.task_prioritizer import TaskPrioritizer

# Optional helpers (retry + micro-cache)
try:
    from utils.utils_retry import retry
except Exception:
    # Fallback minimal retry if helper not present
    def retry(fn, retries: int = 2, backoff: float = 0.6, exceptions=(Exception,)):
        import time
        def wrapped(*args, **kwargs):
            attempt, delay = 0, backoff
            while True:
                try:
                    return fn(*args, **kwargs)
                except exceptions:  # type: ignore
                    attempt += 1
                    if attempt > retries:
                        raise
                    time.sleep(delay)
                    delay *= 2
        return wrapped

try:
    from utils.utils_cache import cache_get, cache_set
except Exception:
    _CACHE: Dict[str, Any] = {}
    import time
    def cache_get(key: str, ttl: int = 60):
        v = _CACHE.get(key)
        if not v:
            return None
        val, exp = v
        return val if exp > time.time() else None
    def cache_set(key: str, val: Any, ttl: int = 60):
        _CACHE[key] = (val, time.time() + ttl)  # type: ignore

# ---------- Routing config ----------
ROUTES = ["gmail", "calendar", "search", "task"]
CONFIDENCE_THRESHOLD = 0.6

# More precise synonym mapping to avoid conflicts
SYN_TO_INTERNAL = {
    "email": "gmail", "mail": "gmail", "gmail": "gmail", "inbox": "gmail", "message": "gmail", "messages": "gmail",
    "calendar": "calendar", "cal": "calendar", "schedule": "calendar", "event": "calendar", "events": "calendar", "meeting": "calendar", "meetings": "calendar",
    "search": "search", "research": "search", "find": "search", "lookup": "search", "news": "search",
    "task": "task", "prioritize": "task", "priority": "task", "todo": "task", "to-do": "task",
}

def _normalize_route(route: str) -> str:
    r = (route or "").strip().lower()
    return SYN_TO_INTERNAL.get(r, r)


def _keyword_route(text: str) -> str | None:
    """More precise keyword routing with priority order to avoid conflicts"""
    t = (text or "").lower()
    
    # Priority order: most specific patterns first
    patterns = [
        # Gmail patterns - be very specific
        (r"\b(gmail|inbox|latest.*mail|check.*mail|email.*messages|mail.*messages)\b", "gmail"),
        # Calendar patterns - specific to calendar/scheduling
        (r"\b(calendar|upcoming.*event|calendar.*event|schedule|meeting|today.*event|tomorrow.*event)\b", "calendar"),
        # Task patterns - specifically about prioritization/tasks
        (r"\b(prioriti[sz]e.*task|task.*priorit|todo|to-do|task.*list)\b", "task"),
        # Search patterns - general search/research
        (r"\b(search.*for|research|find.*information|lookup|news)\b", "search"),
    ]

    for pat, route in patterns:
        if re.search(pat, t):
            return route
    
    # Fallback for broader patterns if no specific match
    broader_patterns = [
        (r"\bemail\b", "gmail"),
        (r"\bcalendar\b", "calendar"),
        (r"\btask\b", "task"),
        (r"\bsearch\b", "search"),
    ]
    
    for pat, route in broader_patterns:
        if re.search(pat, t):
            return route
            
    return None


# ---------- Clients ----------
gemini = GeminiClient()
gmail_client = GmailClient()
calendar_client = GoogleCalendarClient()
brave_client = BraveSearchClient()
task_prioritizer = TaskPrioritizer()

# ---------- Helpers ----------
def _parse_json_block(txt: str):
    """Try to parse strict JSON from an LLM response; fall back to first {...} block."""
    try:
        return json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None

# ---------- Nodes ----------
def request_classifier(state: AssistantState) -> AssistantState:

     # SAFE FLAG: skip re-classification if we've already classified in this run
    if state.context.get("_classified"):
        return state
    
    # 1) deterministic keyword routing
    kr = _keyword_route(state.user_input)
    if kr:
        state.route = kr
        state.route_confidence = 1.0
        state.logs.append(f"classifier (keyword) → route={kr} conf=1.00")
        state.context["_classified"] = True   # <-- SET FLAG HERE
        return state

    # 2) fallback to LLM JSON
    prompt = f"""
Classify the request into one of: {ROUTES}.
Return STRICT JSON ONLY:
{{"route":"<one of {ROUTES}>","confidence":0.0}}

Examples:
- "Check my latest Gmail messages" → {{"route":"gmail","confidence":0.9}}
- "What are my upcoming calendar events?" → {{"route":"calendar","confidence":0.9}}
- "Prioritize my tasks" → {{"route":"task","confidence":0.9}}
- "Search for LangGraph patterns" → {{"route":"search","confidence":0.9}}

Request: {state.user_input}
"""
    raw = gemini.chat(prompt)
    parsed = _parse_json_block(raw) or {"route": "search", "confidence": 0.5}
    route = _normalize_route(parsed.get("route", "search"))
    conf = float(parsed.get("confidence", 0.5))
    if route not in ROUTES:
        route, conf = "search", 0.5

    state.route = route
    state.route_confidence = conf
    state.logs.append(f"classifier → route={route} conf={conf:.2f}")

    state.context["_classified"] = True #Flag
    return state

def confidence_gate(state: AssistantState) -> AssistantState:
    """If classifier confidence is low, apply a safe fallback policy."""
    if state.route_confidence < CONFIDENCE_THRESHOLD:
        state.logs.append(
            f"confidence low ({state.route_confidence:.2f}) → fallback to 'search'"
        )
        state.route = "search"
    return state


@retry
def _safe_calendar_fetch(max_results: int, calendar_id: str):
    return calendar_client.get_upcoming_events(max_results, calendar_id)


@retry
def _safe_gmail_list(max_results: int):
    return gmail_client.list_messages(max_results)


@retry
def _safe_brave_search(q: str, n: int):
    key = f"brave::{q}::{n}"
    cached = cache_get(key, ttl=120)
    if cached is not None:
        return cached
    res = brave_client.search(q, n)
    cache_set(key, res, ttl=120)
    return res


def agent_router(state: AssistantState) -> AssistantState:
    try:
        if state.route == "gmail":
            messages = _safe_gmail_list(5)
            if not messages:
                state.result = "No Gmail messages found."
                return state
            
            # Process Gmail messages for display
            if hasattr(gmail_client, "get_message_details"):
                details = [gmail_client.get_message_details(m["id"]) for m in messages]
                state.result = details
                
                # Extract subjects for potential task prioritization
                subjects = []
                for m in messages:
                    try:
                        meta = gmail_client.service.users().messages().get(
                            userId="me", id=m["id"], format="metadata", metadataHeaders=["Subject"]
                        ).execute()
                        hdrs = meta.get("payload", {}).get("headers", [])
                        subj = next((h["value"] for h in hdrs if h["name"] == "Subject"), "Email")
                        subjects.append(subj)
                    except Exception:
                        subjects.append("Email")
                
                state.context["gmail_tasks"] = subjects
                
                # Only delegate to task if user explicitly wants prioritization
                # Check if the original request was about prioritization
                if any(word in state.user_input.lower() for word in ["prioritize", "priority", "task", "important"]):
                    state.delegate = "task"
                    state.logs.append("gmail → delegating to task for prioritization")
                    
            else:
                # Fallback if get_message_details not available
                ids = [m["id"] for m in messages]
                state.result = ids
                state.context["gmail_tasks"] = [f"Email {i+1}" for i in range(len(ids))]

        elif state.route == "calendar":
            events = _safe_calendar_fetch(5, calendar_id="primary")
            if not events:
                state.result = "No upcoming calendar events."
                return state
                
            state.result = [
                "📅 "
                + (e["start"].get("dateTime", e["start"].get("date", "")))
                + f" → {e.get('summary','No title')}"
                for e in events
            ]
            # Make summaries available to other agents
            state.context["calendar_tasks"] = [e.get("summary", "Event") for e in events]
            
            # Only delegate to task if user explicitly wants prioritization
            if any(word in state.user_input.lower() for word in ["prioritize", "priority", "task", "important"]):
                state.delegate = "task"
                state.logs.append("calendar → delegating to task for prioritization")

        elif state.route == "search":
            results = _safe_brave_search(state.user_input, 3)
            state.result = (
                [f"🔎 {r['title']} ({r['url']})" for r in results]
                if results else "No search results found."
            )
        
        elif state.route == "task":
            # Build tasks from context (gmail + calendar); if empty, enrich from calendar
            gmail_tasks = state.context.get("gmail_tasks", [])
            cal_tasks = state.context.get("calendar_tasks", [])

            # If no tasks in context, try to gather from available sources
            if not gmail_tasks and not cal_tasks:
                try:
                    # Try to get calendar events for task prioritization
                    events = _safe_calendar_fetch(3, calendar_id="primary")
                    if events:
                        cal_tasks = [e.get("summary", "Event") for e in events]
                        state.context["calendar_tasks"] = cal_tasks
                    
                    # Try to get Gmail messages for task prioritization  
                    messages = _safe_gmail_list(3)
                    if messages:
                        gmail_subjects = []
                        for m in messages:
                            try:
                                meta = gmail_client.service.users().messages().get(
                                    userId="me", id=m["id"], format="metadata", metadataHeaders=["Subject"]
                                ).execute()
                                hdrs = meta.get("payload", {}).get("headers", [])
                                subj = next((h["value"] for h in hdrs if h["name"] == "Subject"), "Email")
                                gmail_subjects.append(subj)
                            except Exception:
                                gmail_subjects.append("Email")
                        state.context["gmail_tasks"] = gmail_subjects
                        gmail_tasks = gmail_subjects
                        
                except Exception as e:
                    state.logs.append(f"task gathering failed: {e}")

            combined = [t for t in (gmail_tasks + cal_tasks) if isinstance(t, str) and t.strip()]
            if not combined:
                state.result = "No tasks found in Gmail or Calendar to prioritize."
                return state

            # Call prioritizer and accept either list OR dict
            try:
                res = task_prioritizer.prioritize(combined)

                # If prioritizer returns a dict (with scoring)
                if isinstance(res, dict):
                    tasks = res.get("tasks", combined)
                    scoring = res.get("scoring")
                    state.result = tasks
                    if scoring:
                        state.context["task_scoring"] = scoring

                # If prioritizer returns a simple list
                elif isinstance(res, list):
                    state.result = res

                # Unknown shape → fallback
                else:
                    state.logs.append(f"task prioritizer returned unsupported type: {type(res)}")
                    state.result = combined

                state.logs.append(f"task prioritized {len(state.result) if isinstance(state.result, list) else 'n/a'} items")
            except Exception as e:
                state.logs.append(f"task prioritization failed: {e}")
                state.result = combined

        else:
            state.result = "❌ Sorry, I don't understand your request."

        return state

    except HttpError as e:
        state.error = f"Google API error: {getattr(e, 'status_code', '')} {e}"
        state.logs.append(state.error)
        state.result = "⚠️ Google API error. Try again shortly."
        return state
    except Exception as e:
        state.error = f"Agent error: {e}"
        state.logs.append(state.error)
        state.result = "⚠️ Something went wrong while processing your request."
        return state


def delegate_router(state: AssistantState) -> AssistantState:
    """Enhanced delegation that properly handles the flow."""
    if state.delegate:
        target = state.delegate
        state.logs.append(f"delegating from {state.route} to {target}")
        state.route = target
        state.delegate = None  # prevent loops
        
        # Process the delegated route
        return agent_router(state)
    return state


def response_node(state: AssistantState) -> AssistantState:
    """Terminal node: state.result already holds the answer; just return state."""
    return state


def verifier_node(state: AssistantState) -> AssistantState:
    """
    Light hallucination/quality check with a single-pass, low-cost self-review.
    """
    try:
        route = (state.route or "").lower()
        payload = state.result

        # Build a compact verification prompt
        prompt = f"""
You are a verification agent. Examine the ROUTE and PAYLOAD and return STRICT JSON ONLY:
{{
  "score": 0.0,              // 0..1 overall quality
  "notes": ["issue or ok"],  // bullets
  "corrected": null          // optional corrected payload with the same type/shape; otherwise null
}}

Rules:
- For "search": check on-topic relevance to "{state.user_input}", remove duplicates/near-duplicates, keep 3–5 best.
- For "task": ensure it's a ranked list; no boilerplate; keep concise items.
- For "gmail"/"calendar": ensure non-empty; if empty, note fallback.
- Do NOT invent new, external facts. Only reformat or filter.

ROUTE: {route}
PAYLOAD:
{json.dumps(payload, ensure_ascii=False) if not isinstance(payload, str) else payload}
"""

        raw = gemini.chat(prompt)
        parsed = _parse_json_block(raw) or {"score": 0.6, "notes": ["fallback parse"], "corrected": None}

        state.verify_score = float(parsed.get("score", 0.6))
        notes = parsed.get("notes") or []
        if isinstance(notes, list):
            state.verify_notes = [str(n) for n in notes][:6]
        else:
            state.verify_notes = [str(notes)]

        corrected = parsed.get("corrected")
        # If the verifier produced a same-shape correction, adopt it.
        if corrected is not None:
            # Keep simple: accept corrected for 'search' and 'task' if it's a list
            if route in ("search", "task") and isinstance(corrected, list):
                state.logs.append("verifier: adopted corrected payload")
                state.result = corrected

        # SAFE FLAG: only refine search once per run
        if route == "search" and state.verify_score < 0.5 and not state.context.get("_search_refined"):
            rq = parsed.get("refined_query")
            if isinstance(rq, str) and rq.strip():
                state.context["refined_query"] = rq.strip()
            state.context["_search_refined"] = True   # <-- SET FLAG HERE
            state.delegate = "search"
            state.logs.append(f"verifier: low search quality ({state.verify_score:.2f}) → refine & re-run search")

        # Optional: if task quality low, force a minimal cleanup
        if route == "task" and isinstance(state.result, list):
            cleaned = [str(x).strip() for x in state.result if str(x).strip()]
            if cleaned and cleaned != state.result:
                state.logs.append("verifier: cleaned task list formatting")
                state.result = cleaned

    except Exception as e:
        state.logs.append(f"verifier error: {e}")

    return state


# ---------- Build & compile graph ----------
graph = StateGraph(AssistantState)

graph.add_node("classifier", request_classifier)
graph.add_node("conf_gate", confidence_gate)
graph.add_node("router", agent_router)
graph.add_node("delegate", delegate_router)
graph.add_node("verifier", verifier_node)
graph.add_node("responder", response_node)

graph.set_entry_point("classifier")
graph.add_edge("classifier", "conf_gate")
graph.add_edge("conf_gate", "router")
graph.add_edge("router", "delegate")

# Add conditional edge from delegate back to verifier or handle re-delegation
def should_continue_delegation(state: AssistantState) -> str:
    """Determine if we need to continue delegation or move to verification"""
    if state.delegate:
        return "router"  # Continue delegation cycle
    else:
        return "verifier"  # Move to verification

graph.add_conditional_edges(
    "delegate",
    should_continue_delegation,
    {
        "router": "router",
        "verifier": "verifier"
    }
)

graph.add_edge("verifier", "responder")

app = graph.compile()