from typing import Optional, Dict, Any
from pydantic import BaseModel

class AssistantState(BaseModel):
    user_input: str
    route: Optional[str] = None   # which agent to route to
    result: Optional[Any] = None  # output of the selected agent
    context: Dict[str, Any] = {}  # any extra metadata
