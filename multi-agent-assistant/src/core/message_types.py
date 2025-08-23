from typing import TypedDict, Dict, Any, Optional, List

class AgentMessage(TypedDict, total=False):
    sender: str                    # agent name
    recipient: Optional[str]
    message_type: str              # "request" | "response" | "status" | "error"
    payload: Dict[str, Any]
    notes: Optional[str]
    trace: Optional[List[str]]
