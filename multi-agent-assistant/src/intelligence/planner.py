# src/intelligence/planner.py
from __future__ import annotations
import re
from typing import Dict, Any, List, Optional
from src.utils.gemini_client import GeminiClient


_SCHEMA = r"""
Return ONLY JSON with this exact shape (no markdown fences):

{
  "workflow_type": "new|modify|continue",
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
  "modify_step": "optional step index to modify (0-based)",
  "clarify": "optional single question to the user if critical info is missing",
  "thinking": ["3-6 short bullets of reasoning"],
  "explain": "1-2 sentence natural explanation of your plan"
}

"rules": [
"Output must be valid JSON only. Do not include commentary, code fences, or extra keys.",
"workflow_type determines how to handle this request:",
"  - 'new': Start a fresh workflow (default)",
"  - 'modify': Update a step in the current pending workflow", 
"  - 'continue': Resume/proceed with the current workflow",
"If modifying a workflow, set modify_step to the 0-based index of the step to change.",
"For modify requests, only return the modified/additional steps, not the entire workflow.",
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

_CONTEXT_EXAMPLES = r"""
CONTEXT AWARE EXAMPLES:

# Scenario: User has a pending 3-step workflow (search, email draft, calendar create)
# Current state: search done, email drafted, waiting to confirm calendar
User: change the meeting time to 3pm
{
  "workflow_type": "modify",
  "modify_step": 2,
  "steps": [
    {
      "agent": "calendar", 
      "tool": "gcal_create_event",
      "args": {"summary": "Discuss React 19 Implementation", "start_local": "next Tuesday 15:00", "end_local": "next Tuesday 16:00", "attendees": ["sarah@company.com"]},
      "assign": "meeting",
      "confirm": true,
      "instruction": "Create calendar event at 3pm instead of 2pm"
    }
  ],
  "thinking": ["User wants to modify the meeting time", "Change 14:00 to 15:00", "Keep all other details same"],
  "explain": "I'll update the meeting time to 3pm."
}

# Scenario: User has pending email draft
User: add more details about performance improvements
{
  "workflow_type": "modify", 
  "modify_step": 1,
  "steps": [
    {
      "agent": "search",
      "tool": "web_search", 
      "args": {"query": "React 19 performance improvements", "count": 3},
      "assign": "performance_info",
      "confirm": false,
      "instruction": "Search for React 19 performance details"
    },
    {
      "agent": "email",
      "tool": "none",
      "args": {"action": "update_draft", "body_from_memory": "search.performance_info.summary", "append": true},
      "assign": "updated_draft", 
      "confirm": false,
      "instruction": "Add performance information to the existing draft"
    }
  ],
  "thinking": ["User wants more content in the email", "Need to search for performance info", "Then append to existing draft"],
  "explain": "I'll search for React 19 performance details and add them to your email draft."
}

# Scenario: User confirms pending action
User: yes, send the email
{
  "workflow_type": "continue",
  "steps": [
    {
      "agent": "email",
      "tool": "none", 
      "args": {"action": "send"},
      "assign": "sent",
      "confirm": false,
      "instruction": "Send the pending email draft"
    }
  ],
  "thinking": ["User confirmed sending", "Proceed with the email action"],
  "explain": "I'll send the email now."
}
"""

_FEWSHOTS = r"""
User: check my unread emails
{
  "workflow_type": "new",
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

User: email my manager that I will be 10 minutes late
{
  "workflow_type": "new",
  "steps": [],
  "clarify":"What is your manager's email address and what subject should I use?",
  "thinking":["Recipient and subject are missing","Mutating action requires confirmation"],
  "explain":"I need their email and a subject before drafting the message."
}

User: send it
{
  "workflow_type": "continue",
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
}
"""

_SYS = f"""
You are the Planner Brain of a multi-tool personal assistant powered by Gemini.
You decide whether to ask a clarifying question, which tool/agent to use, and you
compose a minimal, safe plan that downstream nodes will execute. You do NOT execute tools
yourself; you only return JSON plans.

IMPORTANT: You are CONTEXT-AWARE. You can:
1. Start new workflows (workflow_type: "new")
2. Modify existing pending workflows (workflow_type: "modify") 
3. Continue/resume workflows (workflow_type: "continue")

When the user gives modification requests like "change the time to 3pm", "add more details about X", 
"update the subject", etc., use workflow_type: "modify" and only return the steps that need to change.

Tools you can plan for:
- Email: gmail_list_recent, gmail_read, gmail_send, tool="none" for drafting
- Calendar: gcal_list_events, gcal_create_event, gcal_update_event, gcal_delete_event  
- Web: web_search
- Default chat: agent="default", tool="none"

{_SCHEMA}

Write careful, minimal plans. Ask for missing critical slots via 'clarify' and produce NO steps in that case.
Mutating actions MUST have "confirm": true and a short 'instruction'.

Examples (for style, JSON ONLY):
{_FEWSHOTS}

Context-aware examples:
{_CONTEXT_EXAMPLES}
"""

SEND_RE = re.compile(r"\b(send( it)?|fire it off|go ahead|yes.*send|confirm)\b", re.I)
PUT_IN_EMAIL_RE = re.compile(r"(put|copy|drop)\s+(that|this|the (info|information|summary|result))\s+into an email\s+(to|for)\s+([^\s,;]+)", re.I)
MODIFY_RE = re.compile(r"\b(change|update|modify|add|include|append)\b", re.I)
CONTINUE_RE = re.compile(r"\b(continue|proceed|go ahead|next|yes|confirm)\b", re.I)

def _detect_workflow_type(user_text: str, current_plan: Optional[Dict[str, Any]], memory: Dict[str, Any]) -> str:
    """Determine if this is a new workflow, modification, or continuation."""
    text = user_text.lower().strip()
    
    # Check if there's a pending workflow
    has_pending_plan = current_plan and current_plan.get("steps") and len(current_plan["steps"]) > 0
    has_pending_draft = memory.get("email", {}).get("last_draft") is not None
    has_pending_work = has_pending_plan or has_pending_draft
    
    # Explicit continuation phrases
    if CONTINUE_RE.search(text) and has_pending_work:
        return "continue"
    
    # Explicit send commands
    if SEND_RE.search(text) and has_pending_work:
        return "continue"
    
    # Modification phrases when there's pending work
    if MODIFY_RE.search(text) and has_pending_work:
        return "modify"
        
    # Default to new workflow
    return "new"

def make_plan(
    user_text: str, 
    history: List[Dict[str, str]], 
    gemini: GeminiClient, 
    memory: Optional[Dict[str, Any]] = None,
    current_plan: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Enhanced planner with workflow context awareness."""
    
    memory = memory or {}
    workflow_type = _detect_workflow_type(user_text, current_plan, memory)
    
    # Build context for the LLM
    ctx = "\n".join(f"{h['role'].capitalize()}: {h['content']}" for h in history[-20:])
    
    # Add workflow context
    workflow_context = ""
    if current_plan and current_plan.get("steps"):
        steps_summary = []
        for i, step in enumerate(current_plan["steps"]):
            agent = step.get("agent", "?")
            tool = step.get("tool", "?") 
            instruction = step.get("instruction", "")
            steps_summary.append(f"  {i}: {agent}.{tool} - {instruction}")
        workflow_context = f"\nPENDING WORKFLOW:\n" + "\n".join(steps_summary)
    
    if memory.get("email", {}).get("last_draft"):
        draft = memory["email"]["last_draft"]
        workflow_context += f"\nPENDING EMAIL DRAFT: to={draft.get('to')}, subject='{draft.get('subject')}'"
    
    # Enhanced system prompt for context awareness
    enhanced_sys = f"""{_SYS}

CONTEXT AWARENESS RULES:
- If workflow_type is "continue": create a simple plan to execute the user's request (like "send it")
- If workflow_type is "modify": create steps to modify the pending workflow
- If workflow_type is "new": create a fresh multi-step workflow
- Always use memory paths that match actual storage: "search.last_summary" not "search.assign_var.summary"

Recent context:{ctx or '(none)'}
{workflow_context}

DETECTED WORKFLOW TYPE: {workflow_type}

User: {user_text}

Return JSON now."""
    
    obj = gemini.chat_json_obj(enhanced_sys) or {}
    obj.setdefault("workflow_type", workflow_type)
    obj.setdefault("steps", [])
    obj.setdefault("thinking", [])
    obj.setdefault("explain", "")

    # Fix memory paths in generated plan
    for step in obj.get("steps", []):
        args = step.get("args", {})
        if "body_from_memory" in args:
            # Convert planner paths to actual storage paths
            memory_path = args["body_from_memory"]
            if memory_path.startswith("search.") and ".summary" in memory_path:
                args["body_from_memory"] = "search.last_summary"

    # Heuristic post-processing (safety nets)
    text = (user_text or "").strip()

    # A) "send" / "send it" – let the agent send the last draft from memory
    if not obj["steps"] and SEND_RE.search(text):
        obj["workflow_type"] = "continue"
        obj["steps"] = [{
            "agent": "email",
            "tool": "none",
            "args": {"action": "send"},
            "assign": "sent_auto",
            "confirm": False,
            "instruction": "Send the most recent draft stored in memory."
        }]

    # B) "put that into an email to X" – compose draft using last web summary
    if not obj["steps"]:
        m = PUT_IN_EMAIL_RE.search(text)
        if m:
            to_addr = m.group(5)
            obj["workflow_type"] = "new"
            obj["steps"] = [{
                "agent": "email",
                "tool": "none",
                "args": {
                    "to": to_addr,
                    "subject": "Quick summary",
                    "body_from_memory": "search.last_summary"
                },
                "assign": "draft_from_search",
                "confirm": False,
                "instruction": "Compose a draft from the last web search summary; do not send."
            }]

    if not obj.get("clarify"):
        obj.pop("clarify", None)
    obj["steps"] = obj["steps"][:5]
    return obj