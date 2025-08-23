from typing import Dict, Any
from .base_agent import BaseAgent
from src.mcp_integration.mcp_client import MCPClient

class EmailAgent(BaseAgent):
    name = "email"

    def __init__(self, mcp: MCPClient, gemini=None):
        super().__init__(gemini=gemini, mcp=mcp)

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Basic: fetch a few recent emails (routing filters come later)
        emails = await self.mcp.call_tool("gmail_list_recent", query=None, max_results=5)
        self.add_msg(state, "response", {"summary": f"{len(emails)} recent emails", "items": emails})
        return state
