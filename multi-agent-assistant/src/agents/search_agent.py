from typing import Dict, Any
from .base_agent import BaseAgent
from src.mcp_integration.mcp_client import MCPClient

class SearchAgent(BaseAgent):
    name = "search"

    def __init__(self, mcp: MCPClient, gemini=None):
        super().__init__(gemini=gemini, mcp=mcp)

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        q = state.get("user_input", "").strip() or "news"
        results = await self.mcp.call_tool("web_search", query=q, count=5)
        self.add_msg(state, "response", {"query": q, "items": results})
        return state
