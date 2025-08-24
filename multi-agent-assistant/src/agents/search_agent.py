from __future__ import annotations
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent
from src.mcp_integration.mcp_client import MCPClient
from src.intelligence.verifier import verify_response


class SearchAgent(BaseAgent):
    """
    Planner-aware web search agent.

    - Accepts planner step:
        tool: "web_search"
        args: {"query": "...", "count": 5}
    - Writes to shared state memory:
        state["memory"]["search"]["last_results"]  -> compact list [{title,url,source,snippet}]
        state["memory"]["search"]["last_summary"]  -> short LLM summary (string)

    This lets the planner do chained actions like:
      1) search -> store summary
      2) email.draft with args {"body_from_memory":"search.last_summary"} -> then "send"
    """

    name = "search"

    def __init__(self, mcp: MCPClient, gemini=None):
        super().__init__(gemini=gemini, mcp=mcp)

    # ---------- helpers ----------
    @staticmethod
    def _compact_items(results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        compact: List[Dict[str, str]] = []
        for r in results or []:
            compact.append({
                "title":   r.get("title", "") or "",
                "url":     r.get("url", "") or "",
                "source":  r.get("source", "") or "",
                "snippet": r.get("snippet", "") or "",
            })
        return compact

    def _remember(
        self,
        state: Dict[str, Any],
        summary_text: str,
        compact_results: List[Dict[str, str]],
    ) -> None:
        mem = state.setdefault("memory", {})
        search_mem = mem.setdefault("search", {})
        search_mem["last_results"] = compact_results
        search_mem["last_summary"] = (summary_text or "").strip()

    def _summarize_with_llm(self, results: List[Dict[str, Any]]) -> str:
        if not self.gemini or not results:
            return ""
        top = results[:5]
        lines = []
        for r in top:
            t = r.get("title", "")
            u = r.get("url", "")
            s = r.get("snippet", "")
            lines.append(f"- {t} ({u}) — {s}")
        prompt = (
            "You are a concise research assistant. Read these web results and produce:\n"
            "1) A brief 2–3 sentence synthesis in plain English (no bullets, no hype).\n"
            "2) Then one suggested next action the user might take (e.g., 'draft an email with this summary').\n\n"
            + "\n".join(lines)
        )
        return (self.gemini.chat(prompt) or "").strip()

    # ---------- planner executor ----------
    async def _plan_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        state = args["_state"]
        query = (args.get("query") or state.get("user_input") or "").strip()
        count = int(args.get("count") or 5)

        results = await self.mcp.call_tool("web_search", query=query, count=count)
        compact = self._compact_items(results)
        summary_llm = self._summarize_with_llm(compact)

        # persist into shared memory for chaining
        self._remember(state, summary_llm, compact)

        # payload for UI
        payload: Dict[str, Any] = {
            "query": query,
            "items": compact,
            "summary_llm": summary_llm,
            "suggested_prompts": [
                "draft an email with this summary",
                "send this summary to <email>",
                "open #1",
                "compare #1 vs #2",
            ],
            # optional mapping if you later want to support 'open #N'
            "index_map": {str(i + 1): item.get("url", "") for i, item in enumerate(compact)},
        }

        payload.update(verify_response("search", payload))
        self.add_msg(state, "response", payload)
        return state

    # ---------- main entry ----------
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Planner-aware:
          - if step.tool == "web_search": perform search with args
        Fallback:
          - treat user's utterance as the query and search
        """
        step = state.get("current_step") or {}
        tool = (step.get("tool") or "").strip().lower()
        args = (step.get("args") or {}).copy()
        args["_state"] = state

        # planner path
        if tool == "web_search":
            return await self._plan_search(args)

        # fallback path: search user's text directly
        if state.get("user_input"):
            args.setdefault("query", state["user_input"])
        return await self._plan_search(args)
