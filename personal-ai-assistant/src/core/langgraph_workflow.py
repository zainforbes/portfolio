from langgraph.graph import StateGraph
from src.core.state import AssistantState
from src.core.llm_client import GeminiClient

gemini = GeminiClient()

# --- Nodes ---
def request_classifier(state: AssistantState) -> AssistantState:
    """Classify request into route: email, calendar, search, etc."""
    prompt = f"Classify this user request into one of [gmail, calendar, search]:\n{state.user_input}"
    response = gemini.chat(prompt).lower()

    if "gmail" in response:
        state.route = "gmail"
    elif "calendar" in response:
        state.route = "calendar"
    elif "search" in response:
        state.route = "search"
    else:
        state.route = "fallback"

    return state


def agent_router(state: AssistantState) -> AssistantState:
    """Route request to the correct agent based on classifier result."""
    if state.route == "gmail":
        state.result = "[Gmail agent would run here]"
    elif state.route == "calendar":
        state.result = "[Calendar agent would run here]"
    elif state.route == "search":
        state.result = "[Search agent would run here]"
    else:
        state.result = "Sorry, I don't understand your request."

    return state


def response_node(state: AssistantState) -> str:
    return f"Route: {state.route}\nResult: {state.result}"


# --- Build Graph ---
graph = StateGraph(AssistantState)

graph.add_node("classifier", request_classifier)
graph.add_node("router", agent_router)
graph.add_node("responder", response_node)

graph.set_entry_point("classifier")
graph.add_edge("classifier", "router")
graph.add_edge("router", "responder")

app = graph.compile()
