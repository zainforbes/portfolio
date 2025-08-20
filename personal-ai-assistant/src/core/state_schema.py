from typing import Optional, Dict, Any, List
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
    verify_score: float = 0.0           # 0..1 quality score from verifier
    verify_notes: List[str] = []        # quick notes on issues / fixes