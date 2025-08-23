from typing import Dict, Any, List
from .base_agent import BaseAgent
from src.mcp_integration.mcp_client import MCPClient

def _build_gmail_query(filters: Dict[str, Any]) -> str:
    parts: List[str] = []
    s = (filters.get("sender") or "").strip()
    if s:
        parts.append(f'from:"{s}"')
    if filters.get("unread"):
        parts.append("is:unread")
    try:
        d = int(filters.get("newer_than_days") or 0)
    except (TypeError, ValueError):
        d = 0
    d = max(0, min(d, 365))
    if d > 0:
        parts.append(f"newer_than:{d}d")
    subj = (filters.get("subject") or "").strip()
    if subj:
        subj_escaped = subj.replace('"', r'\"')
        parts.append(f'subject:"{subj_escaped}"')
    extra = (filters.get("query_extra") or "").strip()
    if extra:
        parts.append(extra)
    return " ".join(parts).strip()

class EmailAgent(BaseAgent):
    name = "email"

    def __init__(self, mcp: MCPClient, gemini=None):
        super().__init__(gemini=gemini, mcp=mcp)

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        filters = (state.get("routing") or {}).get("filters", {})
        q = _build_gmail_query(filters)
        emails = await self.mcp.call_tool("gmail_list_recent", query=q or None, max_results=10)
        self.add_msg(state, "response", {"query": q, "count": len(emails), "items": emails})
        return state
