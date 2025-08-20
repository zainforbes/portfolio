from src.core.langgraph_workflow import app
from src.core.state_schema import AssistantState

state = AssistantState(user_input="Check my latest Gmail")
out = app.invoke(state)
print("route:", out["route"], "conf:", out.get("route_confidence"))

# Accept the four supported routes (internal canonical names)
assert out["route"] in ["gmail", "calendar", "search", "task"]
