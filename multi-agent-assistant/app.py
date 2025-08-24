# app.py
from __future__ import annotations
import os
import asyncio
from typing import Any, Dict, Optional

import streamlit as st
from dotenv import load_dotenv, find_dotenv

from src.core.langgraph_workflow import build_workflow
from src.utils.gemini_client import GeminiClient

# --------------------------------
# Bootstrap
# --------------------------------
load_dotenv(find_dotenv(), override=False)
st.set_page_config(page_title="Personal AI Assistant", page_icon="🤖", layout="wide")


def _safe_rerun():
    # Works on Streamlit 1.31+ and older versions
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


# --------------------------------
# Session singletons
# --------------------------------
if "wf" not in st.session_state:
    st.session_state["wf"] = build_workflow()
if "gem" not in st.session_state:
    st.session_state["gem"] = None

# Full transcript (persists across reruns)
if "chat" not in st.session_state:
    st.session_state["chat"] = []  # [{"role":"user"/"assistant","text":str?,"sender":str?,"message_type":str?,"payload":dict?}]

# Agent messages that the graph maintains
if "agent_messages_state" not in st.session_state:
    st.session_state["agent_messages_state"] = []

# Rolling LLM memory
if "llm_history" not in st.session_state:
    st.session_state["llm_history"] = []

# Cursor so we only append new agent messages to transcript once
if "_render_cursor" not in st.session_state:
    st.session_state["_render_cursor"] = 0

# Pending confirmation flag (text-based confirm/cancel)
if "pending_confirm" not in st.session_state:
    st.session_state["pending_confirm"] = False


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


# --------------------------------
# Render helpers
# --------------------------------
def _md_link(url: str, text: Optional[str] = None) -> str:
    return f"[{text or url}]({url})"

def _render_search(payload: Dict[str, Any]):
    items = (payload or {}).get("items") or []
    if not items:
        st.write("No web results.")
        return
    for it in items:
        st.markdown(f"- **{it.get('title','(no title)')}** — {_md_link(it.get('url',''), it.get('source','open'))}")
        if it.get("snippet"):
            st.caption(it["snippet"])

def _render_emails(payload: Dict[str, Any]):
    if payload.get("summary_llm"):
        st.write(payload["summary_llm"])

    if payload.get("requires_confirmation"):
        st.info(payload.get("message") or "Review the draft and reply 'send' or 'cancel'.")
        draft = payload.get("draft") or payload.get("proposal") or {}
        with st.expander("Draft", expanded=True):
            st.json(draft)
        st.caption("Reply with **send** to send, or **cancel** to discard.")
        return

    mode = payload.get("mode")
    if mode == "compose":
        st.info("Draft prepared. Reply **send** to send, or **cancel** to discard.")
        with st.expander("Draft", expanded=True):
            st.json(payload.get("draft") or {})
        return
    if mode == "sent":
        st.success("Email sent.")
        if payload.get("send_result"):
            with st.expander("Send result", expanded=False):
                st.json(payload["send_result"])
        return
    if mode == "read":
        e = payload.get("email") or {}
        st.markdown(f"**From:** {e.get('from','')}")
        st.markdown(f"**Subject:** {e.get('subject','(no subject)')}")
        st.markdown(f"**Date:** {e.get('date','')}")
        if e.get("snippet"):
            st.caption(e["snippet"])
        body = e.get("body","")
        if body:
            with st.expander("Body (truncated)", expanded=True):
                st.write(body[:5000] if isinstance(body, str) else body)
        return

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
    sug = payload.get("suggested_prompts") or []
    if sug:
        st.caption("Try: " + " · ".join(sug[:4]))

def _render_calendar(payload: Dict[str, Any]):
    window = payload.get("window") or {}
    if window:
        st.caption(f"Window: {window.get('label','')} | {window.get('time_min','')} → {window.get('time_max','(open)')}")
    if payload.get("summary_llm"):
        st.write(payload["summary_llm"])

    if payload.get("requires_confirmation"):
        st.info(payload.get("message") or "Please confirm.")
        proposal = payload.get("proposal") or payload.get("event") or {}
        if proposal:
            with st.expander("Proposed details", expanded=True):
                st.json(proposal)
        st.caption("Reply with **send**/**confirm** or **cancel**.")
        return

    items = payload.get("items") or []
    if not items:
        st.info("No events found.")
        return
    def _fmt(s: str) -> str: return s or ""
    for ev in items:
        title = ev.get("summary", "(no title)")
        start = _fmt(ev.get("start","")); end = _fmt(ev.get("end",""))
        with st.expander(f"{title} | {start} → {end}", expanded=False):
            st.markdown(f"**When:** {start} → {end}")
            if ev.get("location"):
                st.markdown(f"**Location:** {ev['location']}")
            st.code(ev.get("id",""), language="text")

    for c in payload.get("conflicts") or []:
        st.warning(f"Conflict: {c.get('a','')} overlaps {c.get('b','')} ({c.get('range','')})")

def _render_planner_trace(payload: Dict[str, Any]):
    with st.expander("🧠 Reasoning trace (planner)", expanded=False):
        if payload.get("explain"):
            st.markdown(f"**Plan:** {payload['explain']}")
        if payload.get("thinking"):
            st.markdown("**Heuristics / thoughts:**")
            for t in payload["thinking"]:
                st.markdown(f"- {t}")
        if payload.get("steps"):
            st.markdown("**Steps:**")
            for i, s in enumerate(payload["steps"]):
                st.markdown(f"{i}. **agent:** `{s.get('agent')}`  **tool:** `{s.get('tool')}` — {s.get('description') or s.get('instruction','')}")

def _render_generic(payload: Dict[str, Any]):
    if payload.get("thinking") or payload.get("steps") or payload.get("explain"):
        _render_planner_trace(payload)
        return
    if isinstance(payload.get("result"), str):
        st.write(payload["result"])
    else:
        st.json(payload or {})

def _render_assistant_message(sender: Optional[str], mtype: Optional[str], payload: Any):
    if sender == "search" and mtype == "response":
        st.subheader("🔎 Web results")
        _render_search(payload.get("payload") or payload)
    elif sender == "email" and mtype == "response":
        st.subheader("📧 Email")
        _render_emails(payload)
    elif sender == "calendar" and mtype == "response":
        st.subheader("📅 Calendar")
        _render_calendar(payload)
    elif sender == "coordinator" and mtype == "error":
        st.error((payload or {}).get("message") or "Unexpected error.")
        raw = (payload or {}).get("raw")
        if raw:
            with st.expander("Details"):
                st.code(str(raw))
    else:
        _render_generic(payload or {})


# --------------------------------
# One turn
# --------------------------------
CONFIRM_WORDS = {"send", "send it", "confirm", "yes", "y", "do it", "please send", "ship it"}
CANCEL_WORDS  = {"cancel", "stop", "no", "n", "discard"}

def run_turn(user_text: str):
    confirm_flag = False
    if st.session_state["pending_confirm"]:
        lt = user_text.strip().lower()
        if lt in CONFIRM_WORDS:
            confirm_flag = True
        elif lt in CANCEL_WORDS:
            st.session_state["pending_confirm"] = False
            st.session_state["chat"].append({"role":"assistant","sender":"default","message_type":"response","payload":{"result":"Okay, I’ve cancelled that."}})
            return

    with st.status("🤖 Thinking… (planner)", expanded=False):
        out = _run_graph_sync({
            "user_input": user_text,
            "agent_messages": st.session_state["agent_messages_state"],
            "history": st.session_state["llm_history"],
            "confirm": confirm_flag,
        })

    st.session_state["agent_messages_state"] = out.get("agent_messages", [])
    st.session_state["llm_history"] = out.get("history", st.session_state["llm_history"])

    msgs = st.session_state["agent_messages_state"]
    start = st.session_state["_render_cursor"]
    for i in range(start, len(msgs)):
        m = msgs[i]
        sender = m.get("sender")
        mtype  = m.get("message_type")
        payload= m.get("payload") or {}

        if payload.get("requires_confirmation"):
            st.session_state["pending_confirm"] = True
        if payload.get("mode") == "sent" or payload.get("result") in {"Email sent.","Event created.","Event updated.","Event deleted."}:
            st.session_state["pending_confirm"] = False

        st.session_state["chat"].append({
            "role":"assistant",
            "sender": sender,
            "message_type": mtype,
            "payload": payload
        })
    st.session_state["_render_cursor"] = len(msgs)


# --------------------------------
# Sidebar
# --------------------------------
with st.sidebar:
    st.title("⚙️ Controls")
    st.caption("MCP tools: Gmail · Calendar · Brave Search")
    st.divider()
    st.write("**Env check**")
    st.write(f"BRAVE_API_KEY: {'✅' if os.getenv('BRAVE_API_KEY') else '❌'}")
    st.write(f"GEMINI_API_KEY: {'✅' if os.getenv('GEMINI_API_KEY') else '❌'}")
    st.write("Google OAuth tokens live in `config/`.")
    st.divider()
    with st.expander("LLM diagnostics"):
        key = os.getenv("GEMINI_API_KEY") or ""
        st.write("Key present:", bool(key))
        st.write("Key length:", len(key))
        st.write("Gemini client in session:", st.session_state.get("gem") is not None)
        if st.button("Re-init LLM"):
            st.session_state["gem"] = None
            st.success("Re-init: ✅" if _get_gemini() else "Re-init: ❌")
        if st.button("Test prompt"):
            gem = _get_gemini()
            st.write(gem.chat("Say 'hello' in one short sentence.") if gem else "No LLM configured.")
    st.divider()
    st.markdown("**Notes**")
    st.markdown("- Mutating actions (send email, create/update/delete events) use text confirmation. Reply **send** or **cancel**.")
    st.markdown("- Timezone assumed: **Africa/Johannesburg**.")
    st.markdown("- All tool outputs persist in the transcript.")


# --------------------------------
# Main page
# --------------------------------
st.title("Personal AI Assistant")

# 1) Chat input FIRST so we can process it and force a rerun before rendering transcript
prompt = st.chat_input("Ask me anything… e.g. “show unread emails from the last 7 days”, “create event 'Demo' tomorrow 10:00-10:30”…")
if prompt:
    st.session_state["chat"].append({"role":"user","text": prompt})
    run_turn(prompt)
    # Important: cause an immediate rerun so the new assistant messages render right now
    _safe_rerun()

# 2) Render the full transcript (now includes the new turn if a prompt was submitted)
for msg in st.session_state["chat"]:
    if msg.get("role") == "user":
        with st.chat_message("user"):
            st.write(msg.get("text",""))
    else:
        with st.chat_message("assistant", avatar="🤖"):
            _render_assistant_message(
                sender=msg.get("sender"),
                mtype=msg.get("message_type"),
                payload=msg.get("payload")
            )
