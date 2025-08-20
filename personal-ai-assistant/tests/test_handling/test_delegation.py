# Phrase likely to hit email then delegate to task if gmail_tasks populated
from src.core.langgraph_workflow import app
from src.core.state_schema import AssistantState

state = AssistantState(user_input="Process my actionable emails and prioritize")
out = app.invoke(state)
print(out.get("logs", []))
# No hard assert; just ensure it doesn't crash and logs exist
