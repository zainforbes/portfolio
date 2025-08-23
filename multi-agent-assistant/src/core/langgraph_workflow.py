from langgraph.graph import StateGraph
from .state_schema import AssistantState

def classify_request_node(state: AssistantState) -> AssistantState:
    # TODO: Replace with Gemini classification
    state["current_agent"] = "email" if "email" in state["user_input"].lower() else "default"
    return state

graph = StateGraph(AssistantState)
graph.add_node("classifier", classify_request_node)
graph.set_entry_point("classifier")
workflow = graph.compile()
