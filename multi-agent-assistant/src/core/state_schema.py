from typing import TypedDict, List, Dict

class AssistantState(TypedDict):
    user_input: str
    current_agent: str
    agent_messages: List[Dict]
