from typing import Dict, Any
from .base_agent import BaseAgent
from src.mcp_integration.mcp_client import MCPClient
from src.intelligence.verifier import verify_response

class SearchAgent(BaseAgent):
    name = "search"

    def __init__(self, mcp: MCPClient, gemini=None):
        super().__init__(gemini=gemini, mcp=mcp)

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        q = (state.get("user_input") or "").strip()
        results = await self.mcp.call_tool("web_search", query=q, count=5)

        payload: Dict[str, Any] = {"query": q, "items": results}

        # verify
        v = verify_response(self.name, payload)
        payload.update(v)

        # optional: short LLM summary with inline refs [1], [2], ...
        if self.gemini and results:
            bullets = []
            for i, it in enumerate(results[:5], 1):
                bullets.append(f"[{i}] {it.get('title','(no title)')} — {it.get('source','')}")
            prompt = (
                "Summarize the key points from these search results in 3-5 bullet points. "
                "Reference items as [1], [2], etc. Do not invent facts.\n" +
                "\n".join(bullets)
            )
            payload["summary_llm"] = self.gemini.chat(prompt)

        self.add_msg(state, "response", payload)
        return state
