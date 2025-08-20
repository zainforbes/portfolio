# src/core/langgraph_workflow.py

import json
import re
from typing import Any, Dict, List

from langgraph.graph import StateGraph
from googleapiclient.errors import HttpError

from src.core.state_schema import AssistantState
from src.core.llm_client import GeminiClient
from src.core.gmail_client import GmailClient
from src.core.calendar_client import GoogleCalendarClient
from src.core.brave_client import BraveSearchClient
from src.core.task_prioritizer import TaskPrioritizer

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

# To help fuzzy cases where LLM might get confused with tasks
SYN_TO_INTERNAL = {
    "email": "gmail", "mail": "gmail", "gmail": "gmail",
    "calendar": "calendar", "cal": "calendar", "schedule": "calendar",
    "search": "search", "research": "search",
    "task": "task", "prioritize": "task", "priority": "task", "todo": "task",
}

def _normalize_route(route: str) -> str:
    r = (route or "").strip().lower()
    return SYN_TO_INTERNAL.get(r, r)


def _keyword_route(text: str) -> str | None:
    t = (text or "").lower()
    patterns = [
        (r"\b(gmail|email|inbox|mail|message|messages)\b", "gmail"),
        (r"\b(calendar|cal|schedule|event|events|meeting|meetings|today|tomorrow)\b", "calendar"),
        (r"\b(search|research|find|lookup|news)\b", "search"),
        (r"\b(task|prioriti[sz]e|priority|todo|to-do)\b", "task"),
    ]
    import re
    for pat, route in patterns:
        if re.search(pat, t):
            return route
    return None


# ---------- Clients ----------
gemini = GeminiClient()
gmail_client = GmailClient()
calendar_client = GoogleCalendarClient()
brave_client = BraveSearchClient()
task_prioritizer = TaskPrioritizer()

# ---------- Routing config ----------
ROUTES = ["gmail", "calendar", "search", "task"]
CONFIDENCE_THRESHOLD = 0.6

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
    kr = _keyword_route(state.user_input)
    if kr:
        state.route = kr
        state.route_confidence = 1.0
        state.logs.append(f"classifier (keyword) → route={kr} conf=1.00")
        return state

    # 2) Fall back to LLM classification
    prompt = f"""
Classify the request into one of: {ROUTES}.
Return STRICT JSON ONLY:
{{"route":"<one of {ROUTES}>","confidence":0.0}}

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
            else:
                # ── Put your snippet right here ─────────────────────────
                if hasattr(gmail_client, "get_message_details"):
                    # keep previews for display
                    details = [gmail_client.get_message_details(m["id"]) for m in messages]
                    state.result = details

                    # but feed SUBJECT-ONLY into the Task Prioritizer
                    subjects = []
                    for m in messages:
                        meta = gmail_client.service.users().messages().get(
                            userId="me", id=m["id"], format="metadata", metadataHeaders=["Subject"]
                        ).execute()
                        hdrs = meta.get("payload", {}).get("headers", [])
                        subj = next((h["value"] for h in hdrs if h["name"] == "Subject"), "Email")
                        subjects.append(subj)

                    state.context["gmail_tasks"] = subjects
                else:
                    # fallback if you didn't add get_message_details()
                    ids = [m["id"] for m in messages]
                    state.result = ids
                    state.context["gmail_tasks"] = ids

                # Optional: trigger delegation to the Task agent
                if state.context.get("gmail_tasks"):
                    state.delegate = "task"
                    state.logs.append("gmail produced tasks → delegating to task")

        elif state.route == "calendar":
            # NOTE: still using 'primary' to stay focused on plan; user can swap ID later
            events = _safe_calendar_fetch(5, calendar_id="primary")
            if not events:
                state.result = "No upcoming calendar events."
            else:
                state.result = [
                    "📅 "
                    + (e["start"].get("dateTime", e["start"].get("date", "")))
                    + f" → {e.get('summary','No title')}"
                    for e in events
                ]
                # Make summaries available to other agents
                state.context["calendar_tasks"] = [e.get("summary", "Event") for e in events]

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

            if not cal_tasks:
                try:
                    events = _safe_calendar_fetch(3, calendar_id="primary")
                    cal_tasks = [e.get("summary", "Event") for e in (events or [])]
                    state.context["calendar_tasks"] = cal_tasks
                except Exception:
                    pass

            combined = [t for t in (gmail_tasks + cal_tasks) if isinstance(t, str) and t.strip()]
            if not combined:
                state.result = "No tasks found in Gmail or Calendar."
                return state

            # ---- Call prioritizer and accept either list OR dict ----
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
            state.result = "❌ Sorry, I don’t understand your request."

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


# Bi-Directional Delegation
def delegate_router(state: AssistantState) -> AssistantState:
    """If an agent set a delegate, hand off and continue within the same run."""
    if state.delegate:
        state.logs.append(f"delegating to {state.delegate}")
        state.route = state.delegate
        state.delegate = None  # prevent loops
        return agent_router(state)
    return state


def response_node(state: AssistantState) -> AssistantState:
    """Terminal node: state.result already holds the answer; just return state."""
    return state


def verifier_node(state: AssistantState) -> AssistantState:
    """
    Light hallucination/quality check with a single-pass, low-cost self-review.
    - For 'search': check relevance & duplicates, optionally refine.
    - For 'task': ensure it's a list, sorted; nudge formatting.
    - For 'gmail'/'calendar': ensure non-empty, otherwise suggest fallback.
    Produces verify_score (0..1) and verify_notes.
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

        # Optional: if search quality low, try one refinement (delegate back to search)
        if route == "search" and state.verify_score < 0.5:
            state.logs.append(f"verifier: low search quality ({state.verify_score:.2f}) → refine query")
            state.user_input = f"Refine and narrow this query for higher relevance: {state.user_input}"
            state.delegate = "search"

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
graph.add_node("responder", response_node)
graph.add_node("verifier", verifier_node)

graph.set_entry_point("classifier")
graph.add_edge("classifier", "conf_gate")
graph.add_edge("conf_gate", "router")
graph.add_edge("router", "delegate")
graph.add_edge("delegate", "verifier")
graph.add_edge("verifier", "responder")

app = graph.compile()
