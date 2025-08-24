# app.py
import os
import asyncio
from typing import Any, Dict, Optional, List, Tuple

import streamlit as st
from dotenv import load_dotenv, find_dotenv

from src.core.langgraph_workflow import build_workflow
from src.utils.gemini_client import GeminiClient

# -----------------------------
# Bootstrap
# -----------------------------
load_dotenv(find_dotenv(), override=False)
st.set_page_config(page_title="Personal AI Assistant", page_icon="🤖", layout="wide")

# -----------------------------
# Session singletons
# -----------------------------
if "wf" not in st.session_state:
    st.session_state["wf"] = build_workflow()

if "gem" not in st.session_state:
    st.session_state["gem"] = None

if "chat" not in st.session_state:
    st.session_state["chat"] = []  # conversation transcript

if "agent_messages_state" not in st.session_state:
    st.session_state["agent_messages_state"] = []

if "llm_history" not in st.session_state:
    st.session_state["llm_history"] = []

if "memory" not in st.session_state:
    st.session_state["memory"] = {}

if "confirm_context" not in st.session_state:
    st.session_state["confirm_context"] = None

if "pending_workflow" not in st.session_state:
    st.session_state["pending_workflow"] = None

if "workflow_history" not in st.session_state:
    st.session_state["workflow_history"] = []


# -----------------------------
# Helpers
# -----------------------------
def _get_gemini() -> Optional[GeminiClient]:
    key = os.getenv("GEMINI_API_KEY") or ""
    if not key:
        return None
    if st.session_state["gem"] is None:
        try:
            st.session_state["gem"] = GeminiClient()
        except Exception as e:
            st.error(f"Gemini init failed: {e}")
            return None
    return st.session_state["gem"]


def _run_graph_sync(state: Dict[str, Any]) -> Dict[str, Any]:
    wf = st.session_state["wf"]
    try:
        return asyncio.run(wf.ainvoke(state))
    except RuntimeError:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(wf.ainvoke(state), loop)
            return fut.result()
        return loop.run_until_complete(wf.ainvoke(state))


def _md_link(url: str, text: Optional[str] = None) -> str:
    text = text or url
    return f"[{text}]({url})"


def render_search(payload: Dict[str, Any]):
    items = (payload or {}).get("items") or []
    if payload.get("summary_llm"):
        st.write(payload["summary_llm"])
    if not items:
        st.write("No web results.")
        return
    for it in items:
        t = it.get("title", "(no title)")
        u = it.get("url", "")
        s = it.get("source", "open")
        st.markdown(f"- **{t}** — {_md_link(u, s)}")
        if it.get("snippet"):
            st.caption(it["snippet"])


def render_emails(payload: Dict[str, Any]):
    mode = payload.get("mode")
    if payload.get("summary_llm"):
        st.write(payload["summary_llm"])

    if mode == "read":
        e = payload.get("email") or {}
        st.markdown(f"**From:** {e.get('from','')}")
        st.markdown(f"**Subject:** {e.get('subject','(no subject)')}")
        st.markdown(f"**Date:** {e.get('date','')}")
        if e.get("snippet"):
            st.caption(e["snippet"])
        body = e.get("body", "")
        if body:
            with st.expander("Body (truncated if large)", expanded=True):
                st.write(body[:5000] if isinstance(body, str) else body)
        return

    if mode == "compose":
        d = payload.get("draft") or {}
        st.info("Draft ready. Reply 'send' to send, or reply with edits.")

        to_field = d.get("to", [])
        if isinstance(to_field, str):
            to_display = to_field
        else:
            to_display = ", ".join(to_field) if to_field else "(none)"

        st.markdown(f"**To:** {to_display}")
        st.markdown(f"**Subject:** {d.get('subject','(no subject)')}")
        with st.expander("Body", expanded=True):
            st.write((d.get("body", "") or "")[:4000])
        return

    if mode == "sent":
        st.success(
            f"✅ Email sent to {', '.join(payload.get('sent_to', [])) or '(recipient)'} — "
            f"“{payload.get('subject','(no subject)')}”."
        )
        return

    # list mode
    items_compact = payload.get("items_compact") or []
    if not items_compact:
        st.info("No emails matched.")
        return
    for c in items_compact:
        with st.expander(f"#{c['idx']}  {c['subject']} — {c['from']}", expanded=False):
            st.markdown(f"**From:** {c['from']}")
            st.markdown(f"**Date:** {c.get('date','')}")
            if c.get("snippet"):
                st.caption(c["snippet"])


def render_calendar(payload: Dict[str, Any]):
    window = payload.get("window") or {}
    if window:
        st.caption(
            f"Window: {window.get('label','')} | "
            f"{window.get('time_min','')} → {window.get('time_max','(open)')}"
        )
    if payload.get("summary_llm"):
        st.write(payload["summary_llm"])
    items = payload.get("items") or []
    if not items:
        st.info("No events found.")
        return

    def _fmt(s: str) -> str:
        return s or ""

    for ev in items:
        title = ev.get("summary", "(no title)")
        start = _fmt(ev.get("start", ""))
        end = _fmt(ev.get("end", ""))
        with st.expander(f"{title} | {start} → {end}", expanded=False):
            st.markdown(f"**When:** {start} → {end}")
            loc = ev.get("location", "")
            if loc:
                st.markdown(f"**Location:** {loc}")
            st.code(ev.get("id", ""), language="text")


def render_generic(payload: Dict[str, Any]):
    if not payload:
        st.write("OK.")
        return
    if isinstance(payload.get("result"), str):
        st.write(payload["result"])
    else:
        st.json(payload)


def render_planner_trace(payload: Dict[str, Any]):
    explain = payload.get("explain") or ""
    thinking = payload.get("thinking") or []
    steps = payload.get("steps") or []
    workflow_type = payload.get("workflow_type", "new")

    icons = {"new": "🔄", "modify": "✏️", "continue": "▶️"}
    icon = icons.get(workflow_type, "🔄")

    with st.expander(f"{icon} Reasoning trace ({workflow_type} workflow)", expanded=True):
        if explain:
            st.markdown(f"**Plan:** {explain}")
        if thinking:
            st.markdown("**Heuristics / thoughts:**")
            for t in thinking:
                st.markdown(f"- {t}")
        if steps:
            st.markdown("**Steps:**")
            for idx, s in enumerate(steps):
                step_status = ""
                if workflow_type == "modify":
                    step_status = " ✏️ (modified)" if idx == payload.get("modify_step", -1) else ""
                st.markdown(
                    f"{idx}. **agent:** `{s.get('agent')}`  "
                    f"**tool:** `{s.get('tool')}`{step_status}  — {s.get('description','') or s.get('instruction','')}"
                )


# -----------------------------
# Display policy helpers
# -----------------------------
SIGNAL_SENDERS = {"search", "email", "calendar"}
CONFIRM_KEYS = ("requires_confirmation", "proposal", "confirm_context")


def _is_confirm_msg(m: Dict[str, Any]) -> bool:
    p = (m.get("payload") or {})
    return bool(p.get("requires_confirmation")) and (p.get("proposal") or p.get("confirm_context"))


def _select_messages_for_display(
    all_msgs: List[Dict[str, Any]],
    start_idx: int
) -> List[Tuple[int, Dict[str, Any]]]:
    """
    Return a compact set of (index, message) from messages produced in THIS turn:
    - keep the planner trace (if present)
    - keep ALL confirmation prompts
    - keep only the LAST message per signal sender (search/email/calendar)
    - drop planner 'response' chatter and default filler when not needed
    """
    new = list(enumerate(all_msgs[start_idx:], start=start_idx))
    if not new:
        return []

    last_by_sender: Dict[str, Tuple[int, Dict[str, Any]]] = {}
    confirms: List[Tuple[int, Dict[str, Any]]] = []
    last_trace: Optional[Tuple[int, Dict[str, Any]]] = None

    for idx, m in new:
        sender = m.get("sender")
        mtype = m.get("message_type")
        if sender == "planner" and mtype == "trace":
            last_trace = (idx, m)
            continue

        if _is_confirm_msg(m):
            confirms.append((idx, m))
            continue

        if sender in SIGNAL_SENDERS and m.get("message_type") == "response":
            last_by_sender[sender] = (idx, m)
            continue
        # else: ignore planner chatter/default summaries for this compact view

    picks: List[Tuple[int, Dict[str, Any]]] = []
    if last_trace is not None:
        picks.append(last_trace)
    picks.extend(sorted(confirms, key=lambda x: x[0]))
    picks.extend(sorted(last_by_sender.values(), key=lambda x: x[0]))

    if not picks:
        # last-resort fallback: show the last domain message, else last default
        for idx, m in reversed(new):
            if m.get("sender") in SIGNAL_SENDERS:
                picks.append((idx, m))
                break
        if not picks:
            for idx, m in reversed(new):
                if m.get("sender") == "default":
                    picks.append((idx, m))
                    break

    return picks


# -----------------------------
# One turn
# -----------------------------
def run_turn(user_text: str, *, confirm: bool = False, confirm_context: Optional[Dict[str, Any]] = None):
    # Where this turn starts in the timeline
    start_idx = len(st.session_state["agent_messages_state"])

    state = {
        "user_input": user_text,
        "agent_messages": st.session_state["agent_messages_state"],
        "history": st.session_state["llm_history"],
        "memory": st.session_state["memory"],
        "confirm": confirm,
        "plan": st.session_state.get("pending_workflow"),  # pass pending workflow as current plan
    }
    if confirm_context:
        state["confirm_context"] = confirm_context

    out = _run_graph_sync(state)

    # Persist
    st.session_state["agent_messages_state"] = out.get("agent_messages", [])
    st.session_state["llm_history"] = out.get("history", st.session_state["llm_history"])
    st.session_state["memory"] = out.get("memory", st.session_state["memory"])

    # Pending workflow persistence
    workflow_type = out.get("workflow_type", "new")
    if workflow_type in ["new", "modify"] and out.get("plan"):
        st.session_state["pending_workflow"] = out["plan"]
    elif workflow_type == "continue" and out.get("step_index", 0) >= len(out.get("plan", {}).get("steps", [])):
        st.session_state["pending_workflow"] = None

    # Cache confirm_context if present (from any confirm prompt emitted this turn)
    for idx in range(len(st.session_state["agent_messages_state"]) - 1, start_idx - 1, -1):
        msg = st.session_state["agent_messages_state"][idx]
        p = msg.get("payload") or {}
        if p.get("requires_confirmation") and (p.get("proposal") or p.get("confirm_context")):
            st.session_state["confirm_context"] = p.get("proposal") or p.get("confirm_context")
            break

    # Decide exactly what to show from this turn
    all_msgs = st.session_state["agent_messages_state"]
    picks = _select_messages_for_display(all_msgs, start_idx)

    # Render the selected messages
    for _, m in picks:
        sender = m.get("sender")
        mtype = m.get("message_type")
        payload = m.get("payload") or {}

        with st.chat_message("assistant", avatar="🤖"):
            # 1) planner trace (always shown)
            if sender == "planner" and mtype == "trace":
                render_planner_trace(payload)
                continue

            # 2) confirmation prompts
            if payload.get("requires_confirmation") and (payload.get("proposal") or payload.get("confirm_context")):
                st.info(payload.get("message") or "Confirm?")
                st.caption("Reply with **send** to proceed, or provide edits/cancel.")
                continue

            # 3) domain payloads
            if sender == "search" and mtype == "response":
                st.subheader("🔎 Web results")
                render_search(payload.get("payload") or payload)
                continue

            if sender == "email" and mtype == "response":
                st.subheader("📧 Email")
                render_emails(payload)
                continue

            if sender == "calendar" and mtype == "response":
                st.subheader("📅 Calendar")
                render_calendar(payload)
                continue

            # 4) last resort (rare)
            render_generic(payload)

    # push last bubble snapshot for transcript (the final selected one if any)
    if picks:
        st.session_state["chat"].append({"role": "assistant", "content": picks[-1][1].get("payload", {})})


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("⚙️ Controls")
    st.caption("MCP tools: Gmail · Calendar · Brave Search")
    st.divider()
    st.write("**Env check**")
    st.write(f"BRAVE_API_KEY: {'✅' if os.getenv('BRAVE_API_KEY') else '❌'}")
    st.write(f"GEMINI_API_KEY: {'✅' if os.getenv('GEMINI_API_KEY') else '❌'}")
    st.write("Google OAuth: check tokens in `config/`")
    st.divider()
    with st.expander("LLM diagnostics"):
        key = os.getenv("GEMINI_API_KEY") or ""
        st.write("Key present:", bool(key))
        st.write("Key length:", len(key))
        st.write("Gemini client in session:", st.session_state.get("gem") is not None)
        if st.button("Re-init LLM"):
            st.session_state["gem"] = None
            ok = _get_gemini() is not None
            st.success("Re-init: ✅" if ok else "Re-init: ❌")
        if st.button("Test prompt"):
            gem = _get_gemini()
            if not gem:
                st.error("No LLM configured.")
            else:
                st.write(gem.chat("Say 'hello' in one short sentence."))
    st.divider()
    st.markdown("**Notes**")
    st.markdown("- Mutating actions use typed confirmation: reply **send**.")
    st.markdown("- The planner trace is shown every turn.")
    st.markdown("- Cross-agent memory lets me chain steps (e.g., search → email).")

# -----------------------------
# Main page
# -----------------------------
st.title("Personal AI Assistant")

# Show entire chat history (compact)
for msg in st.session_state["chat"]:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            content = msg["content"]
            if isinstance(content, dict):
                if isinstance(content.get("result"), str):
                    st.write(content["result"])
                elif "summary_llm" in content and isinstance(content["summary_llm"], str):
                    st.write(content["summary_llm"])
                else:
                    st.json(content)
            else:
                st.write(str(content))

# Chat input with typed confirm
user_text = st.chat_input("Ask me to check email, manage calendar, or search the web…")
if user_text:
    with st.chat_message("user"):
        st.write(user_text)
    st.session_state["chat"].append({"role": "user", "content": user_text})

    confirm_phrases = {"send", "send it", "confirm", "yes", "yes, send"}
    lower = user_text.strip().lower()
    is_confirm = lower in confirm_phrases

    if is_confirm and st.session_state.get("confirm_context"):
        run_turn(user_text, confirm=True, confirm_context=st.session_state["confirm_context"])
        st.session_state["confirm_context"] = None
    else:
        run_turn(user_text, confirm=False)
