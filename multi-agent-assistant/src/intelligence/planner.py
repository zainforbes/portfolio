# src/intelligence/planner.py
from typing import Dict, Any, List
from src.utils.gemini_client import GeminiClient

_SCHEMA = r"""
Return ONLY JSON with this exact shape (no markdown fences):

{
  "steps": [
    {
      "agent": "email|calendar|search|default",
      "tool":  "gmail_list_recent|gmail_read|gmail_send|gcal_list_events|gcal_create_event|gcal_update_event|gcal_delete_event|web_search|none",
      "args":  {},
      "assign": "optional_variable_name",
      "confirm": false,
      "instruction": "brief instruction for the agent describing the intended outcome"
    }
  ],
  "clarify": "optional single question to the user if critical info is missing",
  "thinking": ["3-6 short bullets of reasoning"],
  "explain": "1-2 sentence natural explanation of your plan"
}

Rules:
- Output must be valid JSON only. Do not include commentary, code fences, or extra keys.
- Prefer a single, minimal step for read-only actions (email list/read, calendar list, web search).
- For mutating actions (send email, create/update/delete calendar event) ALWAYS set confirm=true.
- If any required slot is missing:
  - For email sending: missing recipient/subject/body -> set clarify, steps=[]
  - For calendar create/update/delete: missing time(s), title, or event_id -> set clarify, steps=[]
- Calendar listing should use a natural window in args (e.g. {"window":"today"} or {"window":"tomorrow"}), not raw ISO times.
- Calendar creation should use start_local / end_local (e.g., "tomorrow 20:00") for the executor/agent to normalize.
- When useful, set assign to store outputs for possible chaining later.
"""

_FEWSHOTS = r"""
User: check my unread emails
{
  "steps": [
    {
      "agent":"email",
      "tool":"gmail_list_recent",
      "args":{"query":"is:unread","max_results":10},
      "assign":"unread",
      "confirm":false,
      "instruction":"List recent unread messages for review."
    }
  ],
  "thinking":["Read-only is safe","The user likely decides next action after viewing results"],
  "explain":"I'll fetch your unread emails so you can choose what to do next."
}

User: read email number 2
{
  "steps": [
    {
      "agent":"email",
      "tool":"gmail_read",
      "args":{"index":2},
      "assign":"mail2",
      "confirm":false,
      "instruction":"Read the second email from the last listed results if index-based; otherwise resolve by latest list."
    }
  ],
  "thinking":["Index likely refers to last listed set"],
  "explain":"I'll open the second email from the latest list."
}

User: email my manager that I will be 10 minutes late
{
  "steps": [],
  "clarify":"What is your manager’s email address and what subject should I use?",
  "thinking":["Recipient and subject are missing","Mutating action requires confirmation"],
  "explain":"I need their email and a subject before drafting the message."
}

User: create 'Demo' tomorrow 10:00-10:30 with Mia in Boardroom
{
  "steps": [
    {
      "agent":"calendar",
      "tool":"gcal_create_event",
      "args":{
        "summary":"Demo",
        "start_local":"tomorrow 10:00",
        "end_local":"tomorrow 10:30",
        "location":"Boardroom",
        "attendees":["mia@example.com"]
      },
      "assign":"evt",
      "confirm":true,
      "instruction":"Create the calendar event once the user confirms."
    }
  ],
  "thinking":["Mutating calendar operation","Time is specified","Attendee present","Requires confirmation"],
  "explain":"I'll prepare the meeting and create it after you confirm."
}

User: what meetings do I have tomorrow?
{
  "steps": [
    {
      "agent":"calendar",
      "tool":"gcal_list_events",
      "args":{"window":"tomorrow","max_results":20},
      "assign":"tmw",
      "confirm":false,
      "instruction":"List tomorrow’s events."
    }
  ],
  "thinking":["Read-only query"],
  "explain":"I'll list your events for tomorrow."
}

User: search gemini api docs
{
  "steps": [
    {
      "agent":"search",
      "tool":"web_search",
      "args":{"query":"gemini api docs","count":5},
      "assign":"hits",
      "confirm":false,
      "instruction":"Retrieve top web results for the query."
    }
  ],
  "thinking":["Read-only web search"],
  "explain":"I'll search the web and summarize the key docs."
}
"""

_SYS = f"""
You are the Planner Brain of a multi-tool personal assistant powered by Gemini.
You decide whether to ask a clarifying question, which tool/agent to use, and you
compose a minimal, safe plan that downstream nodes will execute. You do NOT execute tools
yourself; you only return JSON plans.

Tools you can plan for:
- Email:
  - gmail_list_recent(query,max_results)
  - gmail_read(id|index)
  - gmail_send(to,subject,body)
- Calendar:
  - gcal_list_events(window,max_results)   # window ∈ {"today","tomorrow","this week","next week","next 7 days"}
  - gcal_create_event(summary,start_local,end_local,location,description,attendees[])
  - gcal_update_event(event_id, summary?, start_local?, end_local?, location?, description?)
  - gcal_delete_event(event_id)
- Web:
  - web_search(query,count)
- Default chat:
  - agent="default", tool="none"

{_SCHEMA}

Write careful, minimal plans. Ask for missing critical slots via 'clarify' and produce NO steps in that case.
Mutating actions MUST have "confirm": true and a short 'instruction'.
Prefer reusing the conversation context (e.g., previously listed emails/events) if indexes are referenced.

Examples (for style, JSON ONLY):
{_FEWSHOTS}
"""

def make_plan(user_text: str, history: List[Dict[str, str]], gemini: GeminiClient) -> Dict[str, Any]:
    # build lightweight context window for the planner (last ~20 messages)
    ctx = "\n".join(f"{h['role'].capitalize()}: {h['content']}" for h in history[-20:] if h.get("role") and h.get("content"))
    prompt = f"{_SYS}\n\nConversation so far:\n{ctx or '(none)'}\n\nUser: {user_text}\n\nReturn JSON now."
    obj = gemini.chat_json_obj(prompt) or {}

    # defensive normalization
    if not isinstance(obj, dict):
        obj = {}
    obj.setdefault("steps", [])
    obj.setdefault("thinking", [])
    obj.setdefault("explain", "")

    # Keep signal tight
    if not obj.get("clarify"):
        obj.pop("clarify", None)

    # Cap steps for safety
    if isinstance(obj.get("steps"), list):
        obj["steps"] = obj["steps"][:5]
    else:
        obj["steps"] = []

    return obj
