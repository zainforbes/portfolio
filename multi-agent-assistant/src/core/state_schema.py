from typing import TypedDict, List, Dict, Any, Optional

class AssistantState(TypedDict, total=False):
    user_input: str
    current_agent: str
    agent_messages: List[Dict[str, Any]]

    # NEW
    routing: Dict[str, Any]
    history: List[Dict[str, str]]  # [{role:"user|assistant", "content": "..."}]
    confirm: bool                   # user approved side-effects (send email, create/delete event)
