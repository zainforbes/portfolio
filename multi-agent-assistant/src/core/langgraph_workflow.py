from typing import Dict
from langgraph.graph import StateGraph
from .state_schema import AssistantState

# Tool registry (already implemented in your project)
from src.mcp_integration.mcp_client import MCPClient
from src.mcp_integration.search_server import web_search
from src.mcp_integration.gmail_server import list_recent_emails, read_email
from src.mcp_integration.calendar_server import list_events

MCP = MCPClient()
MCP.register_tool("web_search", web_search)
MCP.register_tool("gmail_list_recent", list_recent_emails)
MCP.register_tool("gmail_read", read_email)
MCP.register_tool("gcal_list_events", list_events)

# Agents
from src.agents.coordinator_agent import CoordinatorAgent
from src.agents.default_agent import DefaultAgent
from src.agents.email_agent import EmailAgent
from src.agents.calendar_agent import CalendarAgent
from src.agents.search_agent import SearchAgent

AGENTS: Dict[str, object] = {
    "default": DefaultAgent(),
    "email":   EmailAgent(MCP),
    "calendar":CalendarAgent(MCP),
    "search":  SearchAgent(MCP),
}
COORDINATOR = CoordinatorAgent(AGENTS)

# Simple keyword classifier (Gemini routing comes in Phase 3 Step 11)
ROUTES = {
    "email":    ["email", "gmail", "inbox", "message"],
    "calendar": ["calendar", "meeting", "event", "schedule"],
    "search":   ["search", "look up", "news", "web", "find"],
}
def classify_request_node(state: AssistantState) -> AssistantState:
    text = (state.get("user_input") or "").lower()
    chosen = "default"
    for agent, keywords in ROUTES.items():
        if any(k in text for k in keywords):
            chosen = agent
            break
    state["current_agent"] = chosen
    return state

# Sync wrapper around async coordinator
import asyncio
def coordinator_node(state: AssistantState) -> AssistantState:
    return asyncio.run(COORDINATOR.execute(state))

def build_workflow():
    graph = StateGraph(AssistantState)
    graph.add_node("classifier", classify_request_node)
    graph.add_node("coordinator", coordinator_node)
    graph.set_entry_point("classifier")
    graph.add_edge("classifier", "coordinator")
    return graph.compile()

if __name__ == "__main__":
    wf = build_workflow()
    print("\nEMAIL:",   wf.invoke({"user_input":"check my emails", "agent_messages":[]}))
    print("\nCAL:",     wf.invoke({"user_input":"what meetings do I have later?", "agent_messages":[]}))
    print("\nSEARCH:",  wf.invoke({"user_input":"search gemini api docs", "agent_messages":[]}))
    print("\nDEFAULT:", wf.invoke({"user_input":"tell me a joke", "agent_messages":[]}))
