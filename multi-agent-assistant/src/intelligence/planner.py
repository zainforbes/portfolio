# src/intelligence/planner.py
from typing import Dict, Any, List, Optional
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

"rules": [
"Output must be valid JSON only. Do not include commentary, code fences, or extra keys.",
"Prefer a single, minimal step for read-only actions (email list/read, calendar list, web search).",
"For mutating actions (send email, create/update/delete calendar event) ALWAYS set confirm=true.",
"Exception: If the user explicitly says 'send' or 'send it', treat that as explicit confirmation and you may set confirm=false for the send step.",
"If any required slot is missing: for email sending, if recipient/subject/body are missing -> set clarify and steps=[]; for calendar create/update/delete, if title, time(s), or event_id are missing -> set clarify and steps=[].",
"Calendar listing should use a natural window in args (e.g., {"window":"today"} or {"window":"tomorrow"}), not raw ISO times.",
"Calendar creation should use start_local / end_local strings (e.g., "tomorrow 20:00") for the executor/agent to normalize.",
"Email drafting must be non-mutating: use agent="email", tool="none" with args {to,subject,body} to create a draft; do not send in the same turn.",
"Sending an email should reference the pending draft in memory: use agent="email", tool="none", args {"action":"send"}.",
"When useful, set assign to store outputs for possible chaining later (e.g., assign search results, email list).",
"Support pulling content from memory into drafts via args {"body_from_memory":"search.last_summary"} or similar memory paths.",
"For "read email number N", prefer a single read-only step that uses args.index to resolve by last listed set.",
"Keep plans minimal (<= 5 steps), safe, and conversational. Ask exactly one concise clarification question when required."
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

User: please write an email for me
{
  "steps": [],
  "clarify": "Who should I send it to, and what should the subject and body say?",
  "thinking": ["Recipient, subject, and body are missing"],
  "explain": "I need the recipient and a brief idea of the subject and message before I draft it."
}

User: send it
{
  "steps": [
  {
      "agent": "email",
      "tool": "none",
      "args": { "action": "send" },
      "assign": "sent",
      "confirm": false,
      "instruction": "Send the last pending draft from memory."
    }
  ],
  "thinking": ["User explicitly said 'send' which counts as confirmation", "Send the pending draft from memory"],
  "explain": "I'll send your last draft now."

User: please check what impact.com is all about
{
  "steps": [
  {
      "agent": "search",
      "tool": "web_search",
      "args": { "query": "impact.com overview what is impact.com", "count": 5 },
      "assign": "impact_overview",
      "confirm": false,
      "instruction": "Find a concise overview of impact.com."
    }
  ],
  "thinking": ["Search then summarize", "Store a short summary in memory for follow-up tasks"],
  "explain": "I'll search for a quick overview.

User: please check what impact.com is all about
{
  "steps": [
  {
      "agent": "search",
      "tool": "web_search",
      "args": { "query": "impact.com overview what is impact.com", "count": 5 },
      "assign": "impact_overview",
      "confirm": false,
      "instruction": "Find a concise overview of impact.com."
    }
  ],
  "thinking": ["Search then summarize", "Store a short summary in memory for follow-up tasks"],
  "explain": "I'll search for a quick overview."

User: copy that information and email it to zainforbes@gmail.com
{
  "steps": [
  {
      "agent": "email",
      "tool": "none",
      "args": {
      "to": "zainforbes@gmail.com
      ",
      "subject": "Quick summary: impact.com",
      "body_from_memory": "search.last_summary"}<
      "assign": "draft1",
      "confirm": false,
      "instruction": "Create a draft using the last search summary; do not send yet."
    }
  ],
  "thinking": ["Use previous search summary from memory", "Draft first so the user can review"],
  "explain": "I’ll draft the email using the last summary and show it to you." 

User: reschedule event abc123 to tomorrow 15:00-16:00
{
  "steps": [
  {
      "agent": "calendar",
      "tool": "gcal_update_event",
      "args": {
      "event_id": "abc123",
      "start_local": "tomorrow 15:00",
      "end_local": "tomorrow 16:00"
      },
      "assign": "evt_update",
      "confirm": true,
      "instruction": "Update the event time after the user confirms."
    }
  ],
  "thinking": ["Mutating calendar action requires confirmation"],
  "explain": "I’ll reschedule the event after you confirm." 

User: reschedule event abc123 to tomorrow 15:00-16:00
{
  "steps": [
  {
      "agent": "calendar",
      "tool": "gcal_delete_event",
      "args": { "event_id": "abc123" },
      "assign": "del1",
      "confirm": true,
      "instruction": "Delete the event after the user confirms."
    }
  ],
  "thinking": ["Destructive action requires confirmation"],
  "explain": "I’ll delete the event after you confirm."

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

def make_plan(
    user_text: str,
    history: List[Dict[str,str]],
    gemini: GeminiClient,
    memory: Optional[Dict[str, Any]] = None
) -> Dict[str,Any]:
    ctx = "\n".join(f"{h['role'].capitalize()}: {h['content']}" for h in history[-20:])
    mem_json = memory or {}
    prompt = (
        f"{_SYS}\n\nRecent context:\n{ctx or '(none)'}"
        f"\n\nWorking memory (JSON):\n{mem_json}"
        f"\n\nUser: {user_text}\n\nReturn JSON now."
    )
    obj = gemini.chat_json_obj(prompt) or {}
    obj.setdefault("steps", [])
    obj.setdefault("thinking", [])
    obj.setdefault("explain", "")
    if not obj.get("clarify"): obj.pop("clarify", None)
    obj["steps"] = obj["steps"][:5]
    return obj
