from typing import TypedDict, List, Dict, Any

class AssistantState(TypedDict, total=False):
    user_input: str
    current_agent: str
    agent_messages: List[Dict[str, Any]]
