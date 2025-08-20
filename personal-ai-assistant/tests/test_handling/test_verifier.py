from src.core.langgraph_workflow import app
from src.core.state_schema import AssistantState

cases = [
    "Search for LangGraph multi-agent orchestration patterns",
    "Prioritize my tasks",
    "Check my latest Gmail messages",
    "What are my upcoming calendar events?"
]

for q in cases:
    out = app.invoke(AssistantState(user_input=q))
    print("\nQ:", q)
    print("route:", out["route"], "conf:", out.get("route_confidence"))
    print("verify_score:", out.get("verify_score"))
    print("notes:", out.get("verify_notes"))
    print("result:", out.get("result")[:3] if isinstance(out.get("result"), list) else out.get("result"))
