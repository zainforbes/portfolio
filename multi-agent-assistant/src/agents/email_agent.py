import re
from typing import Dict, Any, List
from .base_agent import BaseAgent
from src.mcp_integration.mcp_client import MCPClient
from src.intelligence.verifier import verify_response


EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
EMAIL_RE_FULL = re.compile(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')

def _sanitize_filters(f: Dict[str, Any]) -> Dict[str, Any]:
    f = {**f}
    s = (f.get("sender") or "").strip()
    if not s or not EMAIL_RE_FULL.fullmatch(s):
        f.pop("sender", None)  # only keep true emails
    try:
        d = int(f.get("newer_than_days") or 0)
    except (TypeError, ValueError):
        d = 0
    f["newer_than_days"] = max(0, min(d, 365))
    return f

def _parse_email_filters_nl(text: str) -> Dict[str, Any]:
    """Fallback NL parser. Only set sender if an actual email is present."""
    t = (text or "").lower()
    out: Dict[str, Any] = {}

    # unread
    if "unread" in t or "new mail" in t:
        out["unread"] = True

    # last N days/weeks
    m = re.search(r"\blast\s+(\d+)\s*(day|days|week|weeks)\b", t)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        out["newer_than_days"] = n * (7 if "week" in unit else 1)
    elif "last 7 days" in t or "past week" in t:
        out["newer_than_days"] = 7
    elif "today" in t:
        out["newer_than_days"] = 1
    elif "yesterday" in t or "last 24 hours" in t:
        out["newer_than_days"] = 1

    # sender: ONLY capture explicit emails, never phrases like "from the last 7 days"
    m = EMAIL_RE.search(text or "")
    if m:
        out["sender"] = m.group(0)

    # subject/topic
    m3 = re.search(r"\bsubject\s*:\s*([^\n]+)", text or "", flags=re.I)
    if m3:
        out["subject"] = m3.group(1).strip()
    else:
        m4 = re.search(r"\babout\s+([a-zA-Z0-9 \-_/]+)", text or "", flags=re.I)
        if m4:
            out["subject"] = m4.group(1).strip()

    # attachments
    if "attachment" in t or "attachments" in t:
        out["query_extra"] = (out.get("query_extra","") + " has:attachment").strip()

    return out

def _build_gmail_query(filters: Dict[str, Any]) -> str:
    parts: List[str] = []
    s = (filters.get("sender") or "").strip()
    # guard against accidental temporal phrases sneaking in
    if s and any(k in s.lower() for k in ("last", "day", "days", "week", "weeks")):
        s = ""
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
        parts.append(f'subject:"{subj.replace("\"", r"\\\"")}"')
    extra = (filters.get("query_extra") or "").strip()
    if extra:
        parts.append(extra)
    return " ".join(parts).strip()

class EmailAgent(BaseAgent):
    name = "email"

    def __init__(self, mcp: MCPClient, gemini=None, comm=None):
        super().__init__(gemini=gemini, mcp=mcp, comm=comm)

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        routing_filters = (state.get("routing") or {}).get("filters", {}) or {}
        if not any(routing_filters.get(k) for k in ("unread","newer_than_days","sender","subject","query_extra")):
            nl_filters = _parse_email_filters_nl(state.get("user_input",""))
            routing_filters = {**routing_filters, **{k:v for k,v in nl_filters.items() if v}}

        # NEW: sanitize
        routing_filters = _sanitize_filters(routing_filters)

        q = _build_gmail_query(routing_filters)
        emails = await self.mcp.call_tool("gmail_list_recent", query=q or None, max_results=10)

        payload: Dict[str, Any] = {"query": q, "count": len(emails), "items": emails}

        if self.comm and emails:
            import re as _re
            top_from = emails[0].get("from","")
            m = _re.search(r'@([A-Za-z0-9.-]+\.[A-Za-z]{2,})', top_from)
            if m:
                domain = m.group(1).lower()
                preview = await self.comm.ask(self.name, "search", domain)
                payload["enrichment"] = {"sender_domain": domain, "search_preview": (preview.get("items") or [])[:3]}

        # NEW: attach confidence & issues
        payload.update(verify_response("email", payload))

        self.add_msg(state, "response", payload)
        return state
