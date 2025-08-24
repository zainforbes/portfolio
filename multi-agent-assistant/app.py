# app.py
from __future__ import annotations
import os
import asyncio
from typing import Any, Dict, Optional, List

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
    st.session_state["chat"] = []  # transcript for UI

if "agent_messages_state" not in st.session_state:
    st.session_state["agent_messages_state"] = []  # blackboard/tool state

if "llm_history" not in st.session_state:
    st.session_state["llm_history"] = []  # rolling LLM memory

if "pending_confirm" not in st.session_state:
    # {"user_input": str, "confirm": True, "confirm_context": {...}}
    st.session_state["pending_confirm"] = None

if "last_user_input" not in st.session_state:
    st.session_state["last_user_input"] = ""

# When we create mini action buttons (e.g., "Read #3"), we stash here and rerun
if "__action__" not in st.session_state:
    st.session_state["__action__"] = None


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


# ---------- Run async graph from Streamlit ----------
def _run_graph_sync(state: Dict[str, Any]) -> Dict[str, Any]:
    wf = st.session_state["wf"]
    try:
        return asyncio.run(wf.ainvoke(state))
    except RuntimeError:
        # If an event loop is already running (rare), use it
        loop = asyncio.get_event_loop()
        if loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(wf.ainvoke(state), loop)
            return fut.result()
        return loop.run_until_complete(wf.ainvoke(state))


# -----------------------------
# Render helpers
# -----------------------------
def _md_link(url: str, text: Optional[str] = None) -> str:
    text = text or url
    return f"[{text}]({url})"


def render_search(payload: Dict[str, Any]):
    items = (payload or {}).get("items") or []
    if payload.get("summary_llm"):
        st.write(payload["summary_llm"])
    if not items:
        st.info("No web results.")
        return
    for it in items:
        t = it.get("title", "(no title)")
        u = it.get("url", "")
        s = it.get("source", "open")
        st.markdown(f"- **{t}** — {_md_link(u, s)}")
        if it.get("snippet"):
            st.caption(it["snippet"])


def render_emails(payload: Dict[str, Any]):
    mode = payload.get("mode")  # "list" | "read" | "compose"
    if payload.get("summary_llm"):
        st.write(payload["summary_llm"])

    # Compose preview (typically shown inside confirmation too)
    if mode == "compose":
        draft = payload.get("draft") or {}
        to = ", ".join(draft.get("to", []) or [])
        st.markdown(f"**To:** {to or '(none)'}")
        st.markdown(f"**Subject:** {draft.get('subject','(no subject)')}")
        if draft.get("body"):
            with st.expander("Body", expanded=True):
                st.write(draft["body"])
        return

    # Read single
    if mode == "read":
        e = payload.get("email") or {}
        st.markdown(f"**From:** {e.get('from','')}")
        st.markdown(f"**Subject:** {e.get('subject','(no subject)')}")
        st.markdown(f"**Date:** {e.get('date','')}")
        if e.get("snippet"):
            st.caption(e["snippet"])
        body = e.get("body", "")
        if body:
            with st.expander("Body (may be truncated)", expanded=True):
                st.write(body if isinstance(body, str) else str(body))
        return

    # List compact
    items_compact = payload.get("items_compact") or []
    if not items_compact:
        st.info("No emails matched.")
        return

    for c in items_compact:
        idx = c["idx"]
        with st.expander(f"#{idx}  {c['subject']} — {c['from']}", expanded=False):
            st.markdown(f"**From:** {c['from']}")
            st.markdown(f"**Date:** {c.get('date','')}")
            if c.get("snippet"):
                st.caption(c["snippet"])
            cols = st.columns(4)
            with cols[0]:
                if st.button(f"Read #{idx}", key=f"read_{idx}"):
                    st.session_state["__action__"] = f"read email #{idx}"
                    st.experimental_rerun()

    # Suggested prompts
    sug = payload.get("suggested_prompts") or []
    if sug:
        st.caption("Try: " + " · ".join(sug[:4]))


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

    confs = payload.get("conflicts") or []
    if confs:
        st.warning("Conflicts:")
        for c in confs:
            st.markdown(f"- **{c.get('a','')}** overlaps **{c.get('b','')}** ({c.get('range','')})")


def render_generic(payload: Dict[str, Any]):
    if not payload:
        st.write("OK.")
        return
    if isinstance(payload.get("result"), str):
        st.write(payload["result"])
    else:
        st.json(payload)


def render_confirmation(payload: Dict[str, Any], last_user_text: str):
    msg = payload.get("message") or "Confirm?"
    st.info(msg)

    # Display proposed details (works for calendar create/update/delete or email compose)
    proposal = payload.get("proposal") or payload.get("target") or payload.get("event") or {}
    draft = payload.get("draft")

    with st.expander("Proposed details", expanded=True):
        if draft:
            # email draft format
            to = ", ".join(draft.get("to", []) or [])
            st.markdown(f"**To:** {to or '(none)'}")
            st.markdown(f"**Subject:** {draft.get('subject','(no subject)')}")
            st.markdown("**Body:**")
            st.code(draft.get("body", ""), language="text")
        elif proposal:
            st.json(proposal)
        else:
            st.write("(No additional details)")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Confirm", key="confirm_btn"):
            st.session_state["pending_confirm"] = {
                "user_input": last_user_text,
                "confirm": True,
                "confirm_context": payload.get("confirm_context"),  # <— critical
                "intent": payload.get("intent"),
            }
            st.experimental_rerun()
    with col2:
        if st.button("❌ Cancel", key="cancel_btn"):
            st.session_state["pending_confirm"] = None
            st.success("Cancelled.")
            st.experimental_rerun()


def render_planner_trace(payload: Dict[str, Any]):
    with st.expander("🧠 Reasoning trace (planner)", expanded=False):
        explain = payload.get("explain") or ""
        thinking = payload.get("thinking") or []
        steps = payload.get("steps") or []
        if explain:
            st.markdown(f"**Plan:** {explain}")
        if thinking:
            st.markdown("**Heuristics / thoughts:**")
            for t in thinking:
                st.markdown(f"- {t}")
        if steps:
            st.markdown("**Steps:**")
            for idx, s in enumerate(steps):
                st.markdown(
                    f"{idx}. **agent:** `{s.get('agent')}`  "
                    f"**tool:** `{s.get('tool')}`  — {s.get('description', s.get('instruction',''))}"
                )


# -----------------------------
# One turn
# -----------------------------
def run_turn(user_text: str, *, confirm: bool = False, confirm_context: Optional[Dict[str, Any]] = None):
    st.session_state["last_user_input"] = user_text

    with st.status("🤖 Thinking… (planner)", expanded=False):
        out = _run_graph_sync({
            "user_input": user_text,
            "agent_messages": st.session_state["agent_messages_state"],
            "history": st.session_state["llm_history"],
            "confirm": confirm,
            "confirm_context": confirm_context,  # <— planner/executor will use for mutating steps
        })

    # Persist back from graph
    st.session_state["agent_messages_state"] = out.get("agent_messages", [])
    st.session_state["llm_history"] = out.get("history", st.session_state["llm_history"])

    # Render messages from this turn (last few)
    new_msgs = st.session_state["agent_messages_state"]
    for m in new_msgs[-4:]:
        sender = m.get("sender")
        mtype = m.get("message_type")
        payload = m.get("payload") or {}

        with st.chat_message("assistant", avatar="🤖"):
            # Planner reasoning stream
            if sender == "planner" and mtype == "trace":
                render_planner_trace(payload)
                continue

            # Confirmation gate
            if payload.get("requires_confirmation"):
                render_confirmation(payload, user_text)
                continue

            # Agent renders
            if sender == "search" and mtype == "response":
                st.subheader("🔎 Web results")
                render_search(payload.get("payload") or payload)
            elif sender == "email" and mtype == "response":
                st.subheader("📧 Email")
                render_emails(payload)
            elif sender == "calendar" and mtype == "response":
                st.subheader("📅 Calendar")
                render_calendar(payload)
            elif sender == "coordinator" and mtype == "error":
                st.error(payload.get("message") or "Unexpected error.")
                raw = payload.get("raw")
                if raw:
                    with st.expander("Details"):
                        st.code(str(raw))
            else:
                render_generic(payload)

    # Append last assistant payload to chat transcript
    if new_msgs:
        st.session_state["chat"].append({"role": "assistant", "content": new_msgs[-1].get("payload", {})})


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
    st.markdown("- Mutating actions (send email, create/update/delete events) ask confirmation.")
    st.markdown("- Timezone assumed: **Africa/Johannesburg**.")
    st.markdown("- Planner (Gemini) drives routing, tools, and synthesis.")
    st.markdown("- Use compact actions (e.g., *read email #2*) to act on lists quickly.")

# -----------------------------
# Main page
# -----------------------------
st.title("Personal AI Assistant")

# Replay transcript
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

# Auto-run a pending confirmation (created by render_confirmation)
if st.session_state["pending_confirm"]:
    pc = st.session_state["pending_confirm"]
    st.session_state["pending_confirm"] = None
    run_turn(pc.get("user_input") or st.session_state["last_user_input"], confirm=True, confirm_context=pc.get("confirm_context"))

# Inline action (e.g., "Read #3" button in email list)
if st.session_state["__action__"]:
    action_text = st.session_state["__action__"]
    st.session_state["__action__"] = None
    with st.chat_message("user"):
        st.write(action_text)
    st.session_state["chat"].append({"role": "user", "content": action_text})
    run_turn(action_text, confirm=False)

# Chat input
prompt = st.chat_input(
    "Type a message, e.g. “show unread emails from the last 7 days”, "
    "“create event 'Demo' tomorrow 10:00-10:30”, or just chat..."
)
if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state["chat"].append({"role": "user", "content": prompt})
    run_turn(prompt, confirm=False)
