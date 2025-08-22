# src/core/state_schema.py
from typing import TypedDict, List, Dict, Any, Optional

class AssistantState(TypedDict, total=False):
    # Input
    user_input: str

    # Routing & Control
    current_agent: str
    task_type: str
    confidence_score: float
    route_reason: str

    # Agent Communication
    agent_messages: List[Dict[str, Any]]  # message audit trail
    pending_requests: List[Dict[str, Any]]
    completed_tasks: List[Dict[str, Any]]

    # Context & Memory
    conversation_history: List[Dict[str, Any]]
    user_preferences: Dict[str, Any]
    active_context: Dict[str, Any]

    # Error Handling
    retry_count: int
    error_log: List[Dict[str, Any]]
    fallback_triggered: bool

    # Verification & Performance
    verification_scores: Dict[str, Any]
    token_usage: Dict[str, Any]
    response_time: float
    cache_hits: int

    # Output
    final_response: str
    route: str
    route_confidence: float

def make_initial_state(user_input: str, user: Optional[str] = None) -> 'AssistantState':
    """Factory to create a fully-populated initial AssistantState dict."""
    return AssistantState(
        user_input=user_input,
        current_agent="",
        task_type="",
        confidence_score=0.0,
        route="",
        route_confidence=0.0,
        route_reason="",
        agent_messages=[],
        pending_requests=[],
        completed_tasks=[],
        conversation_history=[{"user": user or "me", "text": user_input}],
        user_preferences={},
        active_context={},
        retry_count=0,
        error_log=[],
        fallback_triggered=False,
        verification_scores={},
        token_usage={},
        response_time=0.0,
        cache_hits=0,
        final_response=""
    )