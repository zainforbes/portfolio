from src.core.langgraph_workflow import app
from src.core.state_schema import AssistantState

def run_test(input_text):
    state = AssistantState(user_input=input_text)
    result = app.invoke(state)   # returns dict
    print(f"\nUser Input: {input_text}")
    print("Route:", result["route"])
    print("Final Output:", result["result"])

# Test Gmail
run_test("Check my latest Gmail messages")

# Test Calendar
run_test("What are my upcoming calendar events?")

# Test Search
run_test("Search for news about LangGraph")


