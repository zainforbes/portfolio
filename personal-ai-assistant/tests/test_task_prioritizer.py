from src.core.langgraph_workflow import app
from src.core.state_schema import AssistantState

state = AssistantState(user_input="Can you prioritize my tasks?")
result = app.invoke(state)

print("Final Output:", result["result"])
