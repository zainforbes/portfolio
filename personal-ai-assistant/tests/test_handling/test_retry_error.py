# Force a search (safe) and ensure we get a non-crashy answer
from src.core.langgraph_workflow import app
from src.core.state_schema import AssistantState

state = AssistantState(user_input="Search for LangGraph")
out = app.invoke(state)
assert out.get("result") is not None
