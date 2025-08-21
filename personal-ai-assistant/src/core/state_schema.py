from typing import Optional, Dict, Any, List, Tuple
from pydantic import BaseModel

class AssistantState(BaseModel):
    user_input: str
    route: Optional[str] = None
    route_confidence: float = 0.0
    result: Optional[Any] = None
    context: Dict[str, Any] = {}
    delegate: Optional[str] = None
    error: Optional[str] = None
    logs: List[str] = []

    # verification
    verify_score: float = 0.0
    verify_notes: List[str] = []

    # (optional) error code for UI/tests
    error_code: Optional[str] = None

    # simple memory
    history: List[Tuple[str, str]] = []
    memory_summary: str = ""