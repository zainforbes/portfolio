# src/core/state_schema.py
from typing import TypedDict, List, Dict, Any, Optional

class AssistantState(TypedDict, total=False):
    # Input
    user_input: str
    confirm: bool  # set by UI for mutating steps

    # Conversation memory (rolling)
    history: List[Dict[str, str]]  # [{"role":"user|assistant","content":str}, ...]

    # UI timeline (you already use this)
    agent_messages: List[Dict[str, Any]]

    # Planner output and orchestration
    plan: Dict[str, Any]           # {"steps":[...], "clarify":"...", "thinking":[...], "explain":"..."}
    trace: Dict[str, Any]          # trimmed for UI (thinking, steps, explain)
    step_index: int                # which plan step we’re on
    results: Dict[str, Any]        # assign/result map
    working: Dict[str, Any]        # scratch “blackboard”
    pending_clarify: Optional[str] # question the planner wants to ask

    # Render helpers
    current_agent: str             # for UI badge
    micro_reply: Optional[str]     # mini-synthesis after each step
    ask_followup: Optional[str]    # micro question that pauses the run
    await_user: bool               # node to pause until user responds
