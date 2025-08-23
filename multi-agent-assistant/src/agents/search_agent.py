# src/agents/search_agent.py
from __future__ import annotations
from typing import Dict, Any
from .base_agent import BaseAgent
from src.mcp_integration.mcp_client import MCPClient

class SearchAgent(BaseAgent):
    name = "search"

    def __init__(self, mcp: MCPClient, gemini=None):
        super().__init__(gemini=gemini, mcp=mcp)

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        q = (state.get("user_input") or "").strip()
        results = await self.mcp.call_tool("web_search", query=q, count=5)
        payload: Dict[str, Any] = {"query": q, "items": results}
        if self.gemini:
            bullets = "\n".join(f"- {r.get('title','')} ({r.get('url','')}) — {r.get('snippet','')}" for r in results[:5])
            prompt = (
                "Given these web results, provide a concise answer if possible, "
                "then list 2-3 suggested follow-ups the user might want (like 'open #1', 'compare X vs Y', 'set an alert'). "
                "Avoid filler phrases.\n" + bullets
            )
            payload["summary_llm"] = self.gemini.chat(prompt)
        self.add_msg(state, "response", payload)
        return state
