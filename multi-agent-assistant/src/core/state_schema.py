# src/core/state_schema.py
from typing import TypedDict, List, Dict, Any, Optional

class AssistantState(TypedDict, total=False):
    # Input
    user_input: str
    confirm: bool                          # set by UI when user types “send/confirm/yes”
    confirm_context: Dict[str, Any]        # the exact step (agent/tool/args) to run on confirm

    # Conversation memory (rolling chat)
    history: List[Dict[str, str]]          # [{"role":"user|assistant","content":str}, ...]

    # Shared cross-agent memory (blackboard)
    memory: Dict[str, Any]                 # {"last_search":{...}, "last_email_list":{...}, ...}

    # UI timeline / messages
    agent_messages: List[Dict[str, Any]]   # each: {"sender","message_type","payload",...}

    # Planner + orchestration
    plan: Dict[str, Any]                   # {"steps":[...], "clarify":"...", "thinking":[...], "explain":"..."}
    trace: Dict[str, Any]                  # trimmed for UI (thinking, steps, explain)
    step_index: int                        # which plan step we’re on
    results: Dict[str, Any]                # assign -> result
    working: Dict[str, Any]                # scratchpad
    pending_clarify: Optional[str]         # question to ask before proceeding

    # Render helpers
    current_agent: str                     # for UI badge
    micro_reply: Optional[str]             # short synthesis after a step
    ask_followup: Optional[str]            # micro question to pause run
    await_user: bool                       # used by graph to pause
