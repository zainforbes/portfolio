# src/core/langgraph_workflow.py
from typing import Dict
from langgraph.graph import StateGraph
from .state_schema import AssistantState

from src.mcp_integration.mcp_client import MCPClient
from src.mcp_integration.search_server import web_search
from src.mcp_integration.gmail_server  import list_recent_emails, read_email, send_email  # if you added send
from src.mcp_integration.calendar_server import list_events, create_event, update_event, delete_event
from src.utils.gemini_client import GeminiClient

from src.agents.coordinator_agent import CoordinatorAgent
from src.agents.default_agent     import DefaultAgent
from src.agents.email_agent       import EmailAgent
from src.agents.calendar_agent    import CalendarAgent
from src.agents.search_agent      import SearchAgent

MCP = MCPClient()
MCP.register_tool("web_search",        web_search)
MCP.register_tool("gmail_list_recent", list_recent_emails)
MCP.register_tool("gmail_read",        read_email)
MCP.register_tool("gmail_send",       send_email)        # if implemented
MCP.register_tool("gcal_list_events",  list_events)
MCP.register_tool("gcal_create_event", create_event)
MCP.register_tool("gcal_update_event", update_event)
MCP.register_tool("gcal_delete_event", delete_event)

GEM = GeminiClient()

AGENTS: Dict[str, object] = {
    "default":  DefaultAgent(gemini=GEM, mcp=MCP),
    "email":    EmailAgent(MCP, gemini=GEM),
    "calendar": CalendarAgent(MCP, gemini=GEM),
    "search":   SearchAgent(MCP, gemini=GEM),
}
COORDINATOR = CoordinatorAgent(AGENTS, gemini=GEM)

import asyncio
def coordinator_node(state: AssistantState) -> AssistantState:
    return asyncio.run(COORDINATOR.execute(state))

def build_workflow():
    graph = StateGraph(AssistantState)
    graph.add_node("coordinator", coordinator_node)
    graph.set_entry_point("coordinator")
    return graph.compile()
