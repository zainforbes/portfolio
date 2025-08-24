# src/intelligence/planner.py
from __future__ import annotations
import json
from typing import Dict, Any, List, Optional
from src.utils.gemini_client import GeminiClient

"""
Planner output schema (ALWAYS JSON):

{
  "steps": [
    {
      "mode": "agent|mcp",         // "agent": call named agent.execute; "mcp": call MCP tool directly
      "agent": "email|calendar|search|default",
      "tool":  "gmail_list_recent|gmail_read|gmail_send|gcal_list_events|gcal_create_event|gcal_update_event|gcal_delete_event|web_search|none",
      "args":  { ... },            // plain JSON only; may include {{var}} placeholders to reference prior step results
      "assign":"var_name",         // optional: store last payload into blackboard[var_name]
      "confirmation_required": false,
      "explain": "1 short sentence describing WHY this step exists"
    }
  ],
  "thoughts":  ["2–4 short bullets describing the plan at a glance"],
  "followups": ["1–3 short, concrete next-suggestions for the user"],
  "clarify_question": "optional, 1 sentence; set ONLY when critical info is missing"
}

Rules:
- Output **valid JSON only**; no markdown, no prose outside JSON.
- Prefer **the fewest steps** that satisfy the user safely; max 4 steps per turn.
- If a step **changes state** (send email, create/update/delete calendar events), set `"confirmation_required": true`.
- If critical info is missing (e.g., which email to open, event time), **do not guess**; set `"clarify_question"` and return with `"steps":[]`.
- Use `"mode": "agent"` when the agent already handles that operation; use `"mode": "mcp"` to call a registered MCP tool directly.
- When you need to re-use a previous step’s result, assign it with `"assign":"name"` and reference with `"{{name}}"` in later `"args"`.
  (Do NOT invent structured paths like `{{name.items[0].id}}`; the runtime only supports whole-value substitution.)
- Never fabricate IDs, addresses, times, or attendees. Ask to clarify instead.
"""

_SYS = """
You are the Planning Orchestrator for a multi-tool personal assistant.
Your job: translate the user's request (plus a short chat context) into a compact,
safe STEP PLAN that the runtime executes with Gmail, Google Calendar, and Web Search.

### Capabilities you can plan
- **Email (via agent or MCP tools)**
  - `gmail_list_recent(query, max_results)`
  - `gmail_read(id)`
  - `gmail_send(to, subject, body)`  ⟶ *mutating, requires confirmation*

- **Calendar (via agent or MCP tools)**
  - `gcal_list_events(time_min, time_max, max_results)`
  - `gcal_create_event(summary, start_iso, end_iso, location, description, attendees[])`  ⟶ *mutating, requires confirmation*
  - `gcal_update_event(event_id, fields{})`                                           ⟶ *mutating, requires confirmation*
  - `gcal_delete_event(event_id)`                                                     ⟶ *mutating, requires confirmation*

- **Web Search**
  - `web_search(query, count)`

- **Default Chat**
  - Use agent `default` + `tool:"none"` for general conversation.

### Where to run steps
- `mode:"agent"`: call that agent’s `.execute(state)`; pass parameters via `args` under `routing.filters`.
  *Example:* `{ "mode":"agent","agent":"email","tool":"gmail_list_recent","args":{"query":"is:unread","max_results":10} }`
- `mode:"mcp"`: call an MCP tool directly by name with its `args`.
  *Example:* `{ "mode":"mcp","agent":"calendar","tool":"gcal_create_event","args":{...} }`

### Safety & confirmation
- Any **mutating** action (send email, create/update/delete event) **must** set `"confirmation_required": true`.
- If the user intent is ambiguous or missing critical fields, **don’t guess**. Return **no steps** and a single
  `"clarify_question"` asking exactly what is missing.

### Blackboard variables
- To reuse outputs, mark a step with `"assign":"name"`. The runtime will store that step’s **payload** on a blackboard.
- Later steps may place `"{{name}}"` inside `args` values to substitute the entire payload. No nested indexing is supported.

### Style
- Keep `thoughts` very concise (bullet fragments).
- Keep `followups` concrete (e.g., “Reply to any of these?”, “Create an event from this email?”).
- Keep each step’s `explain` to one short sentence.
- Max 4 steps per plan; collapse where possible.

Return **valid JSON** with keys: `steps`, `thoughts`, `followups`, and optional `clarify_question`.
"""

# Few-shot patterns to anchor behavior
_FEW_SHOTS = r"""
# === EXAMPLE 1: Unread email overview (read-only; safe) ===
User: check if I have unread emails
{
  "steps": [
    {
      "mode": "agent",
      "agent": "email",
      "tool": "gmail_list_recent",
      "args": { "query": "is:unread", "max_results": 10 },
      "assign": "unread",
      "confirmation_required": false,
      "explain": "List recent unread emails."
    }
  ],
  "thoughts": ["User wants unread status", "One safe read-only step is enough"],
  "followups": ["Open one of these?", "Mark any as read?", "Reply to a specific sender"]
}

# === EXAMPLE 2: Read a specific email by ID (needs ID; ask to clarify) ===
User: open the first one
{
  "steps": [],
  "thoughts": ["Needs a specific email id or subject to open"],
  "followups": ["Provide the id or subject to read it"],
  "clarify_question": "Which email should I open? Please provide an id or clear subject line."
}

# === EXAMPLE 3: Create a calendar event (mutating; confirm) ===
User: create event 'Demo' tomorrow 10:00-10:30 with Alex at Boardroom
{
  "steps": [
    {
      "mode": "mcp",
      "agent": "calendar",
      "tool": "gcal_create_event",
      "args": {
        "summary": "Demo",
        "start_iso": "tomorrow 10:00 (user local, ISO)",
        "end_iso":   "tomorrow 10:30 (user local, ISO)",
        "location": "Boardroom",
        "description": "",
        "attendees": ["alex@example.com"]
      },
      "assign": "new_event",
      "confirmation_required": true,
      "explain": "Create the requested event."
    }
  ],
  "thoughts": ["Single mutating step with explicit details", "Requires confirmation"],
  "followups": ["Invite anyone else?", "Add description or reminder?"]
}

# === EXAMPLE 4: Show calendar window (read-only; agent parses NL window) ===
User: what do I have this afternoon?
{
  "steps": [
    {
      "mode": "agent",
      "agent": "calendar",
      "tool": "gcal_list_events",
      "args": {},
      "assign": "events",
      "confirmation_required": false,
      "explain": "List events; the agent parses 'this afternoon' window."
    }
  ],
  "thoughts": ["NL time window handled by agent", "Read-only"],
  "followups": ["Move or extend any event?", "Create a buffer block?"]
}

# === EXAMPLE 5: Web search with summary followup (read-only) ===
User: search gemini api docs
{
  "steps": [
    {
      "mode": "agent",
      "agent": "search",
      "tool": "web_search",
      "args": { "query": "gemini api docs", "count": 5 },
      "assign": "docs",
      "confirmation_required": false,
      "explain": "Search the web."
    }
  ],
  "thoughts": ["Simple search", "Summarization can follow"],
  "followups": ["Want a short summary of the top results?", "Open any of these links?"]
}

# === EXAMPLE 6: Multi-step flow with confirmation (send email) ===
User: email my manager that I'll be 10 minutes late
{
  "steps": [
    {
      "mode": "agent",
      "agent": "email",
      "tool": "gmail_list_recent",
      "args": { "query": "manager OR boss", "max_results": 10 },
      "assign": "candidates",
      "confirmation_required": false,
      "explain": "Try to identify the manager email from inbox context."
    },
    {
      "mode": "mcp",
      "agent": "email",
      "tool": "gmail_send",
      "args": {
        "to": "manager@example.com",
        "subject": "Running 10 minutes late",
        "body": "Hi, just a heads up—I'm running ~10 minutes late but on my way."
      },
      "assign": "sent_msg",
      "confirmation_required": true,
      "explain": "Send the email once confirmed."
    }
  ],
  "thoughts": ["Try to find the most likely recipient", "Sending requires confirmation"],
  "followups": ["Add details?", "Text them as well?"]
}

# === EXAMPLE 7: Missing details ⟶ ask to clarify ===
User: schedule a meeting with Sam
{
  "steps": [],
  "thoughts": ["Missing date/time and Sam's email"],
  "followups": ["Provide date, start/end time, and attendee email"],
  "clarify_question": "What date and time should I schedule it, and what is Sam’s email address?"
}
"""

def _fallback(user_text: str) -> Dict[str, Any]:
    t = (user_text or "").lower()
    if any(k in t for k in ("email","gmail","inbox","unread","message","mail")):
        return {
            "steps": [{
                "mode": "agent", "agent":"email", "tool":"gmail_list_recent",
                "args":{"query":"is:unread", "max_results":10},
                "assign":"unread", "confirmation_required":False,
                "explain":"List unread emails."
            }],
            "thoughts":["Heuristic: list unread emails"],
            "followups":["Open one?", "Reply or archive?"]
        }
    if any(k in t for k in ("calendar","meeting","event","schedule","tomorrow","today","week")):
        return {
            "steps": [{
                "mode":"agent","agent":"calendar","tool":"gcal_list_events",
                "args":{}, "assign":"events", "confirmation_required":False,
                "explain":"List upcoming events."
            }],
            "thoughts":["Heuristic: show calendar"], "followups":[]
        }
    if any(k in t for k in ("search","look up","news","web","find")):
        return {
            "steps": [{
                "mode":"agent","agent":"search","tool":"web_search",
                "args":{"query":user_text,"count":5},"assign":"results",
                "confirmation_required":False,"explain":"Search the web."
            }],
            "thoughts":["Heuristic: web search"], "followups":[]
        }
    return {
        "steps":[{"mode":"agent","agent":"default","tool":"none","args":{},"assign":"",
                  "confirmation_required":False,"explain":"General chat."}],
        "thoughts":["Heuristic: default chat"], "followups":[]
    }

def plan_with_gemini(
    user_text: str,
    history_snippets: List[Dict[str,str]],
    gemini: Optional[GeminiClient]
) -> Dict[str, Any]:
    if not gemini:
        return _fallback(user_text)

    # Short rolling context (last 6 turns)
    ctx_lines: List[str] = []
    for h in history_snippets[-6:]:
        role = (h.get("role") or "").capitalize()
        ctx_lines.append(f"{role}: {h.get('content','')}")
    ctx = "\n".join(ctx_lines) if ctx_lines else "(no recent context)"

    prompt = (
        f"{_SYS}\n\n"
        f"{_FEW_SHOTS}\n\n"
        f"Recent context:\n{ctx}\n\n"
        f"User: {user_text}\n\n"
        "Return JSON with keys: steps, thoughts, followups, clarify_question."
    )
    raw = gemini.chat(prompt)
    try:
        data = json.loads(raw)
        data.setdefault("steps", [])
        data.setdefault("thoughts", [])
        data.setdefault("followups", [])
        steps = []
        for s in data["steps"]:
            steps.append({
                "mode": str(s.get("mode","agent")),
                "agent": str(s.get("agent","default")),
                "tool":  str(s.get("tool","none")),
                "args":  s.get("args", {}) or {},
                "assign": s.get("assign","") or "",
                "confirmation_required": bool(s.get("confirmation_required", False)),
                "explain": s.get("explain","")
            })
        data["steps"] = steps
        return data
    except Exception:
        return _fallback(user_text)
