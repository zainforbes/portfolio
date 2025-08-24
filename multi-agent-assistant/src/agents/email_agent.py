from __future__ import annotations
import re
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent
from src.mcp_integration.mcp_client import MCPClient
from src.intelligence.verifier import verify_response

# --------- regex helpers ----------
EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
EMAIL_RE_FULL = re.compile(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
READ_CMD_RE = re.compile(r"\b(read|open)\b\s*(?:email\s*)?(?:#\s*(\d+)|([A-Fa-f0-9]{10,}))", re.I)

# --------- small utils ----------
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
            "id": e.get("id", ""),
            "from": e.get("from", ""),
            "subject": e.get("subject", "(no subject)"),
            "date": e.get("date", ""),
            "snippet": _shorten(e.get("snippet", ""), 180),
        })
    return compact

def _latest_email_index_map(agent_messages: List[Dict[str, Any]]) -> Dict[int, str]:
    """
    Search previous email response payloads for the last index_map (idx -> message_id).
    This makes "read email #2" work across turns.
    """
    for msg in reversed(agent_messages or []):
        if msg.get("sender") == "email" and msg.get("message_type") == "response":
            payload = msg.get("payload") or {}
            imap = payload.get("index_map")
            if isinstance(imap, dict) and imap:
                # normalize keys back to int
                out: Dict[int, str] = {}
                for k, v in imap.items():
                    try:
                        out[int(k)] = v
                    except Exception:
                        continue
                return out
    return {}

# --------- NL fallbacks for listing filters ----------
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
    """Very small NL parser for fallback filters (read-only list)."""
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
    elif "today" in t or "yesterday" in t or "last 24 hours" in t:
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
        out["query_extra"] = (out.get("query_extra", "") + " has:attachment").strip()
    return out

def _build_gmail_query(filters: Dict[str, Any]) -> str:
    parts: List[str] = []
    s = (filters.get("sender") or "").strip()
    if s and any(k in s.lower() for k in ("last", "day", "days", "week", "weeks")):
        s = ""  # protect against phrases like "from the last 7 days"
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

# --------- memory helpers ----------
def _email_mem(state: Dict[str, Any]) -> Dict[str, Any]:
    mem = state.setdefault("memory", {})
    return mem.setdefault("email", {})

def _set_last_draft(state: Dict[str, Any], draft: Dict[str, Any]) -> None:
    _email_mem(state)["last_draft"] = {
        "to": list(draft.get("to") or []),
        "subject": (draft.get("subject") or "").strip(),
        "body": (draft.get("body") or "").strip(),
    }

def _get_last_draft(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return _email_mem(state).get("last_draft")

def _read_from_memory_path(state: Dict[str, Any], path: str) -> Optional[str]:
    """
    Supports simple dotted paths like 'search.last_summary'.
    Returns a string or None.
    """
    if not path:
        return None
    cur: Any = state.get("memory", {})
    for part in (path or "").split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    if isinstance(cur, str):
        return cur
    return None

# ==============================================
#                  EmailAgent
# ==============================================
class EmailAgent(BaseAgent):
    name = "email"

    def __init__(self, mcp: MCPClient, gemini=None, comm=None):
        super().__init__(gemini=gemini, mcp=mcp, comm=comm)

    # ---------- planner-aligned executors ----------
    async def _plan_list(self, args: Dict[str, Any]) -> Dict[str, Any]:
        state = args["_state"]
        q = (args.get("query") or "").strip()
        max_results = int(args.get("max_results") or 10)

        emails = await self.mcp.call_tool("gmail_list_recent", query=q or None, max_results=max_results)
        compact = _build_compact(emails)
        index_map = {str(c["idx"]): c["id"] for c in compact}

        payload: Dict[str, Any] = {
            "mode": "list",
            "query": q,
            "count": len(emails),
            "items": emails,             # full
            "items_compact": compact,    # numbered, UI-friendly
            "index_map": index_map,      # { "1": "<id>", ... }
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
                "Don't repeat a full list; be concise. End with one specific follow-up like "
                "“Want me to open any? (e.g., read email #2)”\n\n"
                f"Top items:\n{tops}"
            )
            payload["summary_llm"] = self.gemini.chat(prompt)

        payload.update(verify_response("email", payload))
        self.add_msg(state, "response", payload)
        return state

    async def _plan_read(self, args: Dict[str, Any]) -> Dict[str, Any]:
        state = args["_state"]
        index = args.get("index")
        msg_id = args.get("id")

        # allow reading by index if provided
        if index is not None and not msg_id:
            try:
                idx = int(index)
            except Exception:
                idx = None
            if idx is not None:
                imap = _latest_email_index_map(state.get("agent_messages") or [])
                msg_id = imap.get(idx)

        if not msg_id:
            self.add_msg(state, "response", {"result": "I couldn't resolve that email. Try 'read email #2'."})
            return state

        detail = await self.mcp.call_tool("gmail_read", message_id=str(msg_id))
        payload = {
            "mode": "read",
            "message_id": str(msg_id),
            "email": detail,
        }

        if self.gemini:
            subj = detail.get("subject", "(no subject)")
            frm = detail.get("from", "")
            snip = _shorten(detail.get("snippet", ""), 220)
            body_hint = _shorten((detail.get("body", "") or "").strip(), 400)
            prompt = (
                "Summarize this email in 2–3 bullets, then propose one useful next step "
                "(reply / forward / mark read). Keep it crisp.\n\n"
                f"From: {frm}\nSubject: {subj}\nSnippet: {snip}\nBody (truncated):\n{body_hint}"
            )
            payload["summary_llm"] = self.gemini.chat(prompt)

        payload.update(verify_response("email", payload))
        self.add_msg(state, "response", payload)
        return state

    async def _plan_draft(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create/overwrite the pending draft in memory.
        Supports body_from_memory to pull content (e.g., search.last_summary).
        Emits a 'compose' payload, not a confirmation bubble. You then say 'send' to send.
        """
        state = args["_state"]
        to = args.get("to") or []
        to = [to] if isinstance(to, str) else list(to)
        subject = (args.get("subject") or "").strip()
        body = (args.get("body") or "").strip()

        # optional: pull body from memory if requested
        body_from_memory = (args.get("body_from_memory") or "").strip()
        if not body and body_from_memory:
            mem_text = _read_from_memory_path(state, body_from_memory)
            if mem_text:
                body = mem_text

        draft = {"to": to, "subject": subject, "body": body}
        _set_last_draft(state, draft)

        payload: Dict[str, Any] = {
            "mode": "compose",
            "result": "Draft ready.",
            "draft": draft,
            "hint": "Reply with 'send' to send this draft, or say 'edit subject: ...' / 'edit body: ...' to modify.",
        }

        if self.gemini:
            prompt = (
                "You're preparing an email draft for the user. "
                "Produce a concise confirmation line and a single next-step suggestion like "
                "“Reply 'send' to send this now.” Keep it very short."
            )
            payload["summary_llm"] = self.gemini.chat(prompt)

        payload.update(verify_response("email", payload))
        self.add_msg(state, "response", payload)
        return state

    async def _plan_send(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send the pending draft from memory, or if to/subject/body are provided, send directly.
        No button confirmation — rely on planner recognizing 'send it' as explicit consent.
        """
        state = args["_state"]
        draft = _get_last_draft(state)

        # direct send (explicit args win)
        to = args.get("to")
        subject = args.get("subject")
        body = args.get("body")

        if to or subject or body:
            to_list = [to] if isinstance(to, str) else (to or [])
            res = await self.mcp.call_tool("gmail_send", to=to_list, subject=(subject or ""), body=(body or ""))
        else:
            if not draft or not (draft.get("to") and (draft.get("subject") is not None) and (draft.get("body") is not None)):
                self.add_msg(state, "response", {
                    "result": "I don't have a complete draft to send. Provide 'to', 'subject', and 'body', or say 'draft an email ...' first."
                })
                return state
            res = await self.mcp.call_tool("gmail_send", to=draft["to"], subject=draft["subject"], body=draft["body"])

        payload: Dict[str, Any] = {
            "mode": "sent",
            "result": "Email sent.",
            "send_result": res,
        }
        if self.gemini:
            to_disp = ", ".join(res.get("to", []) or draft.get("to", []) if draft else [])
            subj_disp = (res.get("subject") or (draft.get("subject") if draft else "")) or "(no subject)"
            prompt = f"Confirm to the user that the email to {to_disp} with subject “{subj_disp}” was sent."
            payload["summary_llm"] = self.gemini.chat(prompt)

        payload.update(verify_response("email", payload))
        self.add_msg(state, "response", payload)
        return state

    # ---------- NL legacy fallbacks (keep simple) ----------
    async def _list_mode(self, state: Dict[str, Any]) -> Dict[str, Any]:
        routing_filters = (state.get("routing") or {}).get("filters", {}) or {}
        if not any(routing_filters.get(k) for k in ("unread", "newer_than_days", "sender", "subject", "query_extra")):
            nl_filters = _parse_email_filters_nl(state.get("user_input", ""))
            routing_filters = {**routing_filters, **{k: v for k, v in nl_filters.items() if v}}

        routing_filters = _sanitize_filters(routing_filters)
        q = _build_gmail_query(routing_filters)
        return await self._plan_list({"query": q, "max_results": 10, "_state": state})

    async def _read_mode(self, state: Dict[str, Any], idx_or_id: str) -> Dict[str, Any]:
        m_num = re.fullmatch(r"#?\s*(\d+)", idx_or_id.strip())
        if m_num:
            return await self._plan_read({"index": int(m_num.group(1)), "_state": state})
        return await self._plan_read({"id": idx_or_id.strip(), "_state": state})

    # ---------- main execute (planner-aware) ----------
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Planner-aware handling:
          - tool == gmail_list_recent : list
          - tool == gmail_read        : read
          - tool == gmail_send        : send (direct or from draft)
          - tool == none              : draft (if to/subject/body/body_from_memory) or send (if action=='send')
        NL fallback:
          - If 'read/open #N' text -> read by index
          - Else list with parsed filters
        """
        step = state.get("current_step") or {}
        tool = (step.get("tool") or "").strip().lower()
        args = (step.get("args") or {}).copy()
        args["_state"] = state

        if tool == "gmail_list_recent":
            return await self._plan_list(args)
        if tool == "gmail_read":
            return await self._plan_read(args)
        if tool == "gmail_send":
            return await self._plan_send(args)

        if tool == "none":
            action = (args.get("action") or "").strip().lower()
            if action == "send":
                return await self._plan_send(args)
            # treat as draft creation/overwrite (non-mutating)
            return await self._plan_draft(args)

        # ---------- NL fallbacks ----------
        text = state.get("user_input", "")
        m = READ_CMD_RE.search(text)
        if m:
            num = m.group(2)
            mid = m.group(3)
            return await self._read_mode(state, num or mid)

        return await self._list_mode(state)
