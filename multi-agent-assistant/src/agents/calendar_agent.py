from typing import Dict, Any
from .base_agent import BaseAgent
from src.mcp_integration.mcp_client import MCPClient

class CalendarAgent(BaseAgent):
    name = "calendar"

    def __init__(self, mcp: MCPClient, gemini=None):
        super().__init__(gemini=gemini, mcp=mcp)

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        events = await self.mcp.call_tool("gcal_list_events", max_results=5)
        self.add_msg(state, "response", {"summary": f"{len(events)} upcoming events", "items": events})
        return state
