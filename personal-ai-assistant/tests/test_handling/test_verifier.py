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
    print(f"\nQ: {q}")
    # Access all properties as dict keys
    print("route:", out["route"], "conf:", out["route_confidence"])
    print("verify_score:", out["verify_score"])
    print("notes:", out["verify_notes"])
    # Safely print result preview
    result = out["result"]
    preview = result[:3] if isinstance(result, list) else result
    print("result:", preview)
    
    # Print metrics if available in context
    context = out.get("context", {})
    if "metrics" in context and isinstance(context["metrics"], dict):
        metrics = {k: v for k, v in context["metrics"].items() if isinstance(v, (int, float))}
        print("metrics:", metrics)
    
    # Print any errors
    error = out.get("error")
    if error:
        print("error:", error)
        print("error_code:", out.get("error_code"))
