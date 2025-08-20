from src.core.langgraph_workflow import app
from src.core.state_schema import AssistantState

state = AssistantState(user_input="Prioritize my tasks")
result = app.invoke(state)

print("Dynamic Task Prioritizer Output:")

final_result = result["result"]

# Ensure it's always a list for iteration
if isinstance(final_result, str):
    print(final_result)
elif isinstance(final_result, list):
    for item in final_result:
        print("-", item)
else:
    print(final_result)

