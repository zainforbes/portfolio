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
        
        # Current storage pattern (keep for backward compatibility)
        search_mem["last_results"] = compact_results
        search_mem["last_summary"] = (summary_text or "").strip()
        
        # Store at the assign variable location if available
        current_step = state.get("current_step", {})
        assign_var = current_step.get("assign")
        if assign_var:
            search_mem[assign_var] = {
                "summary": summary_text,
                "items": compact_results
            }
        # Legacy location for backward compatibility
        mem["last_search"] = {
            "items": compact_results,
            "summary": summary_text
        }

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
        # Also store where planner expects it
        mem = state.setdefault("memory", {})
        search_mem = mem.setdefault("search", {})
        search_mem["langgraph_info"] = {"summary": summary_llm}

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
    
    def store_search_results_properly(state, summary_llm, compact, assign_var):
        """Store search results in multiple locations for compatibility."""
        mem = state.setdefault("memory", {})
        search_mem = mem.setdefault("search", {})
        
        # Current storage pattern
        search_mem["last_results"] = compact
        search_mem["last_summary"] = summary_llm
        
        # Store at the assign variable location (what planner expects)
        if assign_var and assign_var != "search_results":
            search_mem[assign_var] = {"summary": summary_llm, "items": compact}
        
        # Also store in legacy location for backward compatibility
        mem["last_search"] = {"items": compact, "summary": summary_llm}

    # ---------- main entry ----------
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        q = (state.get("user_input") or "").strip()
        results = await self.mcp.call_tool("web_search", query=q, count=5)
        payload: Dict[str, Any] = {"query": q, "items": results}

        if self.gemini:
            bullets = "\n".join(f"- {r.get('title','')} ({r.get('url','')}) — {r.get('snippet','')}" for r in results[:5])
            prompt = (
                "Given these web results, provide a concise answer if possible, "
                "then list 2–3 suggested follow-ups (e.g., 'open #1', 'compare X vs Y'). "
                "Avoid filler phrases.\n" + bullets
            )
            payload["summary_llm"] = self.gemini.chat(prompt)

        # memory patch so next tools (email) can use it
        payload["memory_patch"] = {
            "last_search": {"items": results, "summary": payload.get("summary_llm")}
        }

        self.add_msg(state, "response", payload)
        return state
