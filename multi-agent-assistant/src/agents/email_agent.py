from __future__ import annotations
import re
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent
from src.mcp_integration.mcp_client import MCPClient
from src.intelligence.verifier import verify_response

EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
EMAIL_RE_FULL = re.compile(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
READ_CMD_RE = re.compile(r"\b(read|open)\b\s*(?:email\s*)?(?:#\s*(\d+)|([A-Fa-f0-9]{10,}))", re.I)


def _sanitize_filters(f: Dict[str, Any]) -> Dict[str, Any]:
    f = {**f}
    s = (f.get("sender") or "").strip()
    if not s or not EMAIL_RE_FULL.fullmatch(s):
        f.pop("sender", None)
    try:
        d = int(f.get("newer_than_days") or 0)
    except (TypeError, ValueError):
        d = 0
    f["newer_than_days"] = max(0, min(d, 365))
    return f


def _parse_email_filters_nl(text: str) -> Dict[str, Any]:
    """Very small NL parser for fallback filters."""
    t = (text or "").lower()
    out: Dict[str, Any] = {}

    if "unread" in t or "new mail" in t:
        out["unread"] = True

    m = re.search(r"\blast\s+(\d+)\s*(day|days|week|weeks)\b", t)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        out["newer_than_days"] = n * (7 if "week" in unit else 1)
    elif "last 7 days" in t or "past week" in t:
        out["newer_than_days"] = 7
    elif "today" in t or "last 24 hours" in t or "yesterday" in t:
        out["newer_than_days"] = 1

    m = EMAIL_RE.search(text or "")
    if m:
        out["sender"] = m.group(0)

    m3 = re.search(r"\bsubject\s*:\s*([^\n]+)", text or "", flags=re.I)
    if m3:
        out["subject"] = m3.group(1).strip()
    else:
        m4 = re.search(r"\babout\s+([a-zA-Z0-9 \-_/]+)", text or "", flags=re.I)
        if m4:
            out["subject"] = m4.group(1).strip()

    if "attachment" in t or "attachments" in t:
        out["query_extra"] = (out.get("query_extra","") + " has:attachment").strip()
    return out


def _build_gmail_query(filters: Dict[str, Any]) -> str:
    parts: List[str] = []
    s = (filters.get("sender") or "").strip()
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


def _shorten(s: str, n: int = 140) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def _build_compact(emails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for i, e in enumerate(emails, 1):
        compact.append({
            "idx": i,
            "id": e.get("id",""),
            "from": e.get("from",""),
            "subject": e.get("subject","(no subject)"),
            "date": e.get("date",""),
            "snippet": _shorten(e.get("snippet",""), 180),
        })
    return compact


def _latest_email_index_map(agent_messages: List[Dict[str, Any]]) -> Dict[int, str]:
    """Search previous email response payloads for the last index_map."""
    for msg in reversed(agent_messages or []):
        if msg.get("sender") == "email" and msg.get("message_type") == "response":
            payload = msg.get("payload") or {}
            imap = payload.get("index_map")
            if isinstance(imap, dict) and imap:
                # keys may be strings -> coerce to int
                m = {}
                for k, v in imap.items():
                    try:
                        m[int(k)] = v
                    except Exception:
                        pass
                if m:
                    return m
    return {}


def _mem_email(state: Dict[str, Any]) -> Dict[str, Any]:
    mem = state.setdefault("memory", {})
    return mem.setdefault("email", {})


def _get_from_memory(state: Dict[str, Any], path: str) -> str:
    """
    Resolve a simple dotted path like 'search.last_summary' from state["memory"].
    """
    cur: Any = state.get("memory") or {}
    for part in (path or "").split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return ""
    return cur if isinstance(cur, str) else ""


class EmailAgent(BaseAgent):
    name = "email"

    def __init__(self, mcp: MCPClient, gemini=None, comm=None):
        super().__init__(gemini=gemini, mcp=mcp, comm=comm)

    # ---------- planner-aware: LIST ----------
    async def _plan_list(self, args: Dict[str, Any]) -> Dict[str, Any]:
        state = args["_state"]
        q = (args.get("query") or "").strip()
        max_results = int(args.get("max_results") or 10)

        emails = await self.mcp.call_tool("gmail_list_recent", query=q or None, max_results=max_results)
        compact = _build_compact(emails)
        index_map = {str(c["idx"]): c["id"] for c in compact}

        # remember last list
        mem = _mem_email(state)
        mem["last_query"] = q
        mem["last_compact"] = compact
        mem["last_index_map"] = index_map

        payload: Dict[str, Any] = {
            "mode": "list",
            "query": q,
            "count": len(emails),
            "items": emails,
            "items_compact": compact,
            "index_map": index_map,
            "suggested_prompts": [
                "read email #1",
                "open email #2",
                "read <paste-message-id>",
                "show unread emails from the last 7 days",
            ],
        }

        if self.gemini and compact:
            tops = "\n".join(f"- {c['idx']}. {c['subject']} — {c['from']}" for c in compact[:5])
            prompt = (
                "You are a helpful assistant looking at the user's recent emails. "
                "Write a short, natural summary of key items (1–3 bullets). "
                "Don't repeat a full list; be concise. Then end with one specific follow-up like "
                "“Want me to open any? (e.g., read email #2)”\n\n"
                f"Top items:\n{tops}"
            )
            payload["summary_llm"] = self.gemini.chat(prompt)

        payload.update(verify_response("email", payload))
        self.add_msg(state, "response", payload)
        return state

    # ---------- planner-aware: READ (by id or #index) ----------
    async def _plan_read(self, args: Dict[str, Any]) -> Dict[str, Any]:
        state = args["_state"]
        index = args.get("index")
        msg_id = args.get("id")

        # allow index resolution via last list; if missing, auto-list first
        if index is not None and not msg_id:
            try:
                idx = int(index)
            except Exception:
                idx = None

            imap = _latest_email_index_map(state.get("agent_messages") or [])
            if not imap and idx is not None:
                # do a fresh list to build mapping
                fallback_q = (_mem_email(state).get("last_query") or "")  # reuse last query if available
                await self._plan_list({"query": fallback_q, "max_results": 10, "_state": state})
                imap = _latest_email_index_map(state.get("agent_messages") or [])

            if idx is not None:
                msg_id = imap.get(idx)

        if not msg_id:
            self.add_msg(state, "response", {"result": "I couldn't resolve that email. Try 'read email #2' after listing."})
            return state

        detail = await self.mcp.call_tool("gmail_read", message_id=str(msg_id))
        payload = {
            "mode": "read",
            "message_id": str(msg_id),
            "email": detail,
        }

        # optional LLM skim (kept light)
        if self.gemini:
            subj = detail.get("subject","(no subject)")
            frm  = detail.get("from","")
            snip = _shorten(detail.get("snippet",""), 220)
            body_hint = _shorten((detail.get("body","") or "").strip(), 400)
            prompt = (
                f"Summarize this email in 2–3 bullets, then propose one next step (reply/forward/mark read).\n"
                f"From: {frm}\nSubject: {subj}\nSnippet: {snip}\nBody (truncated):\n{body_hint}"
            )
            payload["summary_llm"] = self.gemini.chat(prompt)

        payload.update(verify_response("email", payload))
        self.add_msg(state, "response", payload)
        return state

    # ---------- planner-aware: COMPOSE (tool='none' for draft) ----------
    async def _plan_compose(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a draft only (no send). Stores in memory so 'send' can fire next.
        Args may include:
          - to (str or list[str])
          - subject (str)
          - body (str)
          - body_from_memory (e.g., "search.last_summary")
          - instruction / guidance (free-text hint to draft with LLM)
        """
        state = args["_state"]
        to_raw = args.get("to") or []
        to = [to_raw] if isinstance(to_raw, str) else list(to_raw)
        subject = (args.get("subject") or "").strip()
        body = (args.get("body") or "").strip()

        # pull content from memory if provided
        bf = (args.get("body_from_memory") or "").strip()
        if not body and bf:
            body = _get_from_memory(state, bf).strip()

        # if still missing, try to write from instruction or last user input
        if not body and self.gemini:
            guidance = (args.get("instruction") or args.get("guidance") or state.get("user_input") or "").strip()
            if guidance:
                body = (self.gemini.chat(
                    "Write a concise, courteous email based on this instruction. "
                    "Use plain language and keep it under 120 words.\n\nInstruction:\n" + guidance
                ) or "").strip()

        # minimal subject if missing (derive from body)
        if not subject:
            subject = "Quick update"
            if self.gemini and body:
                subject = (self.gemini.chat(
                    "Write a clear, 3–6 word subject line for this email body:\n\n" + body
                ) or "Quick update").strip()

        draft = {"to": to, "subject": subject, "body": body}

        # remember last draft for 'send it'
        mem = _mem_email(state)
        mem["last_draft"] = draft

        payload: Dict[str, Any] = {
            "mode": "compose",
            "draft": draft,
            "suggested_prompts": [
                "send",
                "change subject to 'Running late'",
                "add cc bob@example.com",
            ],
        }
        if self.gemini:
            payload["summary_llm"] = self.gemini.chat(
                f"Summarize the draft in one friendly sentence so the user knows what's prepared. "
                f"End with 'Reply \"send\" to send.'\n\nTo: {', '.join(to)}\nSubject: {subject}\nBody:\n{body}"
            )

        payload.update(verify_response("email", payload))
        self.add_msg(state, "response", payload)
        return state

    # ---------- planner-aware: SEND (uses args or last draft in memory) ----------
    async def _plan_send(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send immediately (no buttons). If args missing fields, uses memory.email.last_draft.
        """
        state = args["_state"]
        mem = _mem_email(state)
        last_draft = mem.get("last_draft") or {}

        to_raw = args.get("to") or last_draft.get("to") or []
        to = [to_raw] if isinstance(to_raw, str) else list(to_raw)
        subject = (args.get("subject") or last_draft.get("subject") or "").strip()
        body = (args.get("body") or last_draft.get("body") or "").strip()

        # allow body_from_memory at send time too
        bf = (args.get("body_from_memory") or "").strip()
        if bf and not body:
            body = _get_from_memory(state, bf).strip()

        if not to or not body:
            self.add_msg(state, "response", {"result": "Missing recipient or body. Say 'compose an email to ...' first."})
            return state

        res = await self.mcp.call_tool("gmail_send", to=to, subject=subject or "(no subject)", body=body)
        payload: Dict[str, Any] = {
            "mode": "sent",
            "result": "Email sent.",
            "send_result": res,
            "sent_to": to,
            "subject": subject or "(no subject)",
        }
        if self.gemini:
            payload["summary_llm"] = self.gemini.chat(
                f"Confirm to the user that the email to {', '.join(to)} with subject “{subject or '(no subject)'}” was sent."
            )

        # clear last draft to avoid accidental re-sends
        mem["last_draft"] = None

        payload.update(verify_response("email", payload))
        self.add_msg(state, "response", payload)
        return state

    # ---------- legacy NL fallbacks ----------
    async def _list_mode(self, state: Dict[str, Any]) -> Dict[str, Any]:
        routing_filters = (state.get("routing") or {}).get("filters", {}) or {}
        if not any(routing_filters.get(k) for k in ("unread","newer_than_days","sender","subject","query_extra")):
            nl_filters = _parse_email_filters_nl(state.get("user_input",""))
            routing_filters = {**routing_filters, **{k:v for k,v in nl_filters.items() if v}}
        routing_filters = _sanitize_filters(routing_filters)
        q = _build_gmail_query(routing_filters)
        return await self._plan_list({"query": q, "max_results": 10, "_state": state})

    async def _read_mode(self, state: Dict[str, Any], idx_or_id: str) -> Dict[str, Any]:
        m_num = re.fullmatch(r"#?\s*(\d+)", idx_or_id.strip())
        if m_num:
            return await self._plan_read({"index": int(m_num.group(1)), "_state": state})
        return await self._plan_read({"id": idx_or_id.strip(), "_state": state})

    # ---------- main entry ----------
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Planner-aware paths:
          - tool == "gmail_list_recent" -> list
          - tool == "gmail_read"        -> read
          - tool == "gmail_send"        -> send (uses args or last draft)
          - tool == "none" + compose args -> compose-only (draft)
        NL fallbacks:
          - "read #N" or "open #N"
          - else list according to parsed filters
        """
        step = state.get("current_step") or {}
        tool = (step.get("tool") or "").strip().lower()
        args = (step.get("args") or {}).copy()
        args["_state"] = state

        # planner routes
        if tool == "gmail_list_recent":
            return await self._plan_list(args)
        if tool == "gmail_read":
            return await self._plan_read(args)
        if tool == "gmail_send":
            return await self._plan_send(args)
        if tool == "none":
            # treat presence of compose fields as a compose request
            if any(k in args for k in ("to", "subject", "body", "body_from_memory", "instruction", "guidance")):
                return await self._plan_compose(args)

        # NL fallbacks
        text = state.get("user_input","")
        m = READ_CMD_RE.search(text)
        if m:
            num = m.group(2)
            mid = m.group(3)
            target = num or mid
            return await self._read_mode(state, target)

        return await self._list_mode(state)
