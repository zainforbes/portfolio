from src.core.langgraph_workflow import app
from src.core.state_schema import AssistantState

out = app.invoke(AssistantState(user_input="help"))
print("route:", out["route"], "conf:", out.get("route_confidence"), "result:", out.get("result"))
assert out.get("result") is not None
