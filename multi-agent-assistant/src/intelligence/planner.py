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
      "mode": "agent|mcp",         // "agent": route to agent.execute; "mcp": call MCP tool by name directly
      "agent": "email|calendar|search|default",
      "tool":  "gmail_list_recent|gmail_read|gmail_send|gcal_list_events|gcal_create_event|gcal_update_event|gcal_delete_event|web_search|none",
      "args":  { ... },            // tool arguments (may contain {{var}} placeholders referencing previous step results)
      "assign":"var_name",         // optional: store the result payload of this step on the shared blackboard[var_name]
      "confirmation_required": false,
      "explain": "short sentence about why this step is needed"
    }
  ],
  "thoughts": ["2–4 short bullets describing the overall plan"],
  "followups": ["optional suggestions for user next"]
}

Rules for Gemini (system prompt enforces):
- Prefer minimal steps that achieve the user request.
- If any step changes email/calendar (send/create/update/delete), set confirmation_required=true for that step.
- If required info is missing, return steps: [] and add a single clarification question in `clarify_question`.
"""

_SYS = (
  "You are the planner for a multi-tool assistant with Gmail, Google Calendar, and Web Search.\n"
  "TOOLS:\n"
  "- email agent tools: gmail_list_recent(query,max_results), gmail_read(id), gmail_send(to,subject,body)\n"
  "- calendar agent tools: gcal_list_events(time_min,time_max,max_results), "
  "  gcal_create_event(summary,start_iso,end_iso,location,description,attendees[]), "
  "  gcal_update_event(event_id,fields{}), gcal_delete_event(event_id)\n"
  "- search: web_search(query,count)\n\n"
  "OUTPUT: VALID JSON ONLY with keys steps[], thoughts[], followups[], clarify_question (optional).\n"
  "Keep thoughts very concise. Only add steps required to satisfy the user.\n"
  "If information is missing (e.g., time window, recipient), return no steps and set clarify_question to one short question."
)

def _fallback(user_text: str) -> Dict[str, Any]:
    t = (user_text or "").lower()
    if any(k in t for k in ("email","gmail","inbox","unread","message","mail")):
        return {
            "steps": [{
                "mode": "agent", "agent":"email", "tool":"gmail_list_recent",
                "args":{"query":"is:unread", "max_results":10},
                "assign":"unread_emails", "confirmation_required":False,
                "explain":"List unread emails"
            }],
            "thoughts":["Heuristic: list unread emails"],
            "followups":["Read a specific email","Mark as read","Reply or archive"]
        }
    if any(k in t for k in ("calendar","meeting","event","schedule","tomorrow","today")):
        return {
            "steps": [{
                "mode":"agent","agent":"calendar","tool":"gcal_list_events",
                "args":{}, "assign":"events", "confirmation_required":False,
                "explain":"Show upcoming events"
            }],
            "thoughts":["Heuristic: show upcoming events"], "followups":[]
        }
    if any(k in t for k in ("search","look up","news","web","find")):
        return {
            "steps": [{
                "mode":"agent","agent":"search","tool":"web_search",
                "args":{"query":user_text,"count":5},"assign":"results",
                "confirmation_required":False,"explain":"Search the web"
            }],
            "thoughts":["Heuristic: web search"], "followups":[]
        }
    return {
        "steps":[{"mode":"agent","agent":"default","tool":"none","args":{},"assign":"", "confirmation_required":False,"explain":"General chat"}],
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
        # keep only safe keys on each step
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
