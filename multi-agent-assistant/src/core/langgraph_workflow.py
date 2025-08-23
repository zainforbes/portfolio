from langgraph.graph import StateGraph
from .state_schema import AssistantState

def classify_request_node(state: AssistantState) -> AssistantState:
    text = (state.get("user_input") or "").lower()
    state["current_agent"] = "email" if "email" in text else "default"
    return state

def build_workflow():
    graph = StateGraph(AssistantState)
    graph.add_node("classifier", classify_request_node)
    graph.set_entry_point("classifier")
    return graph.compile()

# smoke test if you run this file directly
if __name__ == "__main__":
    wf = build_workflow()
    state = {"user_input": "check my emails", "current_agent": "", "agent_messages": []}
    print(wf.invoke(state))
