from langgraph.graph import StateGraph
from src.core.state_schema import AssistantState
from src.core.llm_client import GeminiClient
from src.core.gmail_client import GmailClient
from src.core.calendar_client import GoogleCalendarClient
from src.core.brave_client import BraveSearchClient
from src.core.task_prioritizer import TaskPrioritizer


# Initialize once (reuse across requests)
gmail_client = GmailClient()
calendar_client = GoogleCalendarClient()
brave_client = BraveSearchClient()
gemini = GeminiClient()
task_prioritizer = TaskPrioritizer()

# --- Nodes ---
def request_classifier(state: AssistantState) -> AssistantState:
    prompt = f"""
    Classify this user request into one of: [email, calendar, search, task, meeting].
    Request: {state.user_input}
    """
    response = gemini.chat(prompt).lower()

    if "gmail" in response or "email" in response:
        state.route = "email"
    elif "calendar" in response:
        state.route = "calendar"
    elif "search" in response or "research" in response:
        state.route = "search"
    elif "task" in response or "prioritiz" in response:
        state.route = "task"
    elif "meeting" in response or "summarize" in response:
        state.route = "meeting"
    else:
        state.route = "fallback"
    return state


def agent_router(state: AssistantState) -> AssistantState:
    """Route request to the correct agent based on classifier result."""
    if state.route == "gmail":
        messages = gmail_client.list_messages(5)
        if not messages:
            state.result = "No Gmail messages found."
        else:
            details = [gmail_client.get_message_details(m["id"]) for m in messages]
            state.result = details

    elif state.route == "calendar":
        events = calendar_client.get_upcoming_events(5)
        if not events:
            state.result = "No upcoming calendar events."
        else:
            state.result = [
                f"📅 {e['start'].get('dateTime', e['start'].get('date'))} → {e.get('summary', 'No title')}"
                for e in events
            ]

    elif state.route == "search":
        results = brave_client.search(state.user_input, 3)
        if not results:
            state.result = "No search results found."
        else:
            state.result = [f"🔎 {r['title']} ({r['url']})" for r in results]

    elif state.route == "task":
        # Example: user asks "Prioritize my tasks"
        tasks = [
            "Reply to manager’s email",
            "Prepare slides for tomorrow’s meeting",
            "Doctor appointment at 10 AM",
            "Research LangGraph for project"
        ]
        state.result = task_prioritizer.prioritize(tasks)

    else:
        state.result = "❌ Sorry, I don’t understand your request."

    return state

def response_node(state: AssistantState) -> AssistantState:
    state.result = f"Route: {state.route}\nResult: {state.result}"
    return state


# --- Build Graph ---
graph = StateGraph(AssistantState)

graph.add_node("classifier", request_classifier)
graph.add_node("router", agent_router)
graph.add_node("responder", response_node)

graph.set_entry_point("classifier")
graph.add_edge("classifier", "router")
graph.add_edge("router", "responder")

app = graph.compile()
