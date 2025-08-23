import re
from typing import Dict, Any, List, Tuple
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
        n = int(m.group(1)); unit = m.group(2)
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
        esc = subj.replace('"', r'\"')
        parts.append(f'subject:"{esc}"')
    extra = (filters.get("query_extra") or "").strip()
    if extra:
        parts.append(extra)
    return " ".join(parts).strip()

# ---------- write intents ----------
def _email_intent(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ("send email", "send an email", "email to", "mail to", "send to")):
        return "send"
    if "draft" in t:
        return "draft"
    return "read"

def _parse_send(text: str) -> Tuple[List[str], str, str]:
    """Very small NL parse: emails anywhere, 'subject: ...', 'body: ...'"""
    tos = EMAIL_RE.findall(text or "")
    m_sub = re.search(r"subject\s*:\s*([^\n]+)", text or "", re.I)
    m_body = re.search(r"body\s*:\s*([\s\S]+)", text or "", re.I)
    subject = (m_sub.group(1).strip() if m_sub else "")
    body = (m_body.group(1).strip() if m_body else "")
    return (tos or []), subject, body

class EmailAgent(BaseAgent):
    name = "email"

    def __init__(self, mcp: MCPClient, gemini=None, comm=None):
        super().__init__(gemini=gemini, mcp=mcp, comm=comm)

    async def _handle_read(self, state: Dict[str, Any]) -> Dict[str, Any]:
        routing_filters = (state.get("routing") or {}).get("filters", {}) or {}
        if not any(routing_filters.get(k) for k in ("unread","newer_than_days","sender","subject","query_extra")):
            nl_filters = _parse_email_filters_nl(state.get("user_input",""))
            routing_filters = {**routing_filters, **{k:v for k,v in nl_filters.items() if v}}

        routing_filters = _sanitize_filters(routing_filters)
        q = _build_gmail_query(routing_filters)
        emails = await self.mcp.call_tool("gmail_list_recent", query=q or None, max_results=10)

        payload: Dict[str, Any] = {"intent":"read", "query": q, "count": len(emails), "items": emails}

        # optional enrichment
        if self.comm and emails:
            import re as _re
            top_from = emails[0].get("from","")
            m = _re.search(r'@([A-Za-z0-9.-]+\.[A-Za-z]{2,})', top_from)
            if m:
                domain = m.group(1).lower()
                preview = await self.comm.ask(self.name, "search", domain)
                payload["enrichment"] = {"sender_domain": domain, "search_preview": (preview.get("items") or [])[:3]}

        if self.gemini and emails:
            titles = "\n".join(f"- {e.get('subject','(no subject)')} from {e.get('from','')}" for e in emails[:5])
            payload["summary_llm"] = self.gemini.chat(f"Summarize these emails:\n{titles}")

        payload.update(verify_response("email", payload))
        self.add_msg(state, "response", payload)
        return state

    async def _handle_write(self, state: Dict[str, Any], intent: str) -> Dict[str, Any]:
        text = state.get("user_input","")
        to, subject, body = _parse_send(text)
        payload: Dict[str, Any] = {"intent": intent, "to": to, "subject": subject, "body": body}

        if not to or not subject or not body:
            payload["error"] = "Missing to/subject/body."
            self.add_msg(state, "error", payload)
            return state

        if intent == "send":
            if not state.get("confirm"):
                payload["requires_confirmation"] = True
                payload["message"] = "Send intent detected. Re-run with confirm=True to actually send."
                self.add_msg(state, "response", payload)
                return state
            result = await self.mcp.call_tool("gmail_send", to=to, subject=subject, body=body)
            payload["result"] = {"id": result.get("id")}
        else:  # draft
            result = await self.mcp.call_tool("gmail_draft", to=to, subject=subject, body=body)
            payload["result"] = {"id": result.get("id")}

        if self.gemini:
            payload["summary_llm"] = self.gemini.chat(
                f"Email {'sent' if intent=='send' else 'drafted'} to {', '.join(to)} with subject '{subject}'. Summarize in one sentence."
            )
        payload.update(verify_response("email", {"count": 1, "query": ""}))
        self.add_msg(state, "response", payload)
        return state

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        intent = _email_intent(state.get("user_input",""))
        if intent == "read":
            return await self._handle_read(state)
        else:
            return await self._handle_write(state, intent)
