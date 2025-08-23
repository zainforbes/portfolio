import os
from typing import Dict, Any, List, Optional

import streamlit as st
from dotenv import load_dotenv, find_dotenv

from src.core.langgraph_workflow import build_workflow
from src.utils.gemini_client import GeminiClient

# -----------------------------
# App bootstrap
# -----------------------------
load_dotenv(find_dotenv(), override=False)
st.set_page_config(page_title="Personal AI Assistant", page_icon="🤖", layout="wide")


def _get_gemini() -> Optional[GeminiClient]:
    """Ensure a Gemini client exists in session; create it if possible."""
    api = os.getenv("GEMINI_API_KEY") or ""
    if not api:
        return None
    gem = st.session_state.get("gem")
    if gem is None:
        try:
            gem = GeminiClient()
            st.session_state["gem"] = gem
        except Exception as e:
            st.error(f"Gemini init failed: {e}")
            return None
    return gem


# -----------------------------
# Singletons in session
# -----------------------------
if "wf" not in st.session_state:
    st.session_state["wf"] = build_workflow()

if "gem" not in st.session_state:
    st.session_state["gem"] = None  # created lazily

if "chat" not in st.session_state:
    st.session_state["chat"] = []  # UI transcript

if "agent_messages_state" not in st.session_state:
    st.session_state["agent_messages_state"] = []  # tool/agent state

if "pending_confirm" not in st.session_state:
    st.session_state["pending_confirm"] = None  # holds confirm intents

if "llm_history" not in st.session_state:
    st.session_state["llm_history"] = []  # rolling LLM memory (list of {"role","content"})

# Warm-up LLM once if key is present (does nothing if already set)
if st.session_state.get("gem") is None and os.getenv("GEMINI_API_KEY"):
    _get_gemini()


# -----------------------------
# Helpers: rendering
# -----------------------------
def _md_link(url: str, text: Optional[str] = None) -> str:
    text = text or url
    return f"[{text}]({url})"


def render_search(payload: Dict[str, Any]):
    items = (payload or {}).get("items") or []
    if not items:
        st.write("No web results.")
        return
    for it in items:
        st.markdown(f"- **{it.get('title','(no title)')}** — {_md_link(it.get('url',''), it.get('source','open'))}")
        snip = it.get("snippet") or ""
        if snip:
            st.caption(snip)


def render_emails(payload: Dict[str, Any]):
    items = (payload or {}).get("items") or []
    if payload.get("summary_llm"):
        st.write(payload["summary_llm"])
    if not items:
        st.info("No emails matched.")
        return
    for e in items:
        with st.expander(f"{e.get('subject','(no subject)')} — {e.get('from','')}", expanded=False):
            st.markdown(f"**From:** {e.get('from','')}")
            st.markdown(f"**Date:** {e.get('date','')}")
            st.markdown(f"**Preview:** {e.get('snippet','')}")
            if e.get("id"):
                st.code(e["id"], language="text")


def _format_time_str(s: str) -> str:
    return s or ""


def render_calendar(payload: Dict[str, Any]):
    window = payload.get("window") or {}
    st.caption(
        f"Window: {window.get('label','upcoming')} | "
        f"{window.get('time_min','')} → {window.get('time_max','(open)')}"
    )
    if payload.get("summary_llm"):
        st.write(payload["summary_llm"])
    items = payload.get("items") or []
    if not items:
        st.info("No events found.")
        return
    for ev in items:
        with st.expander(
            f"{ev.get('summary','(no title)')} | "
            f"{_format_time_str(ev.get('start',''))} → {_format_time_str(ev.get('end',''))}",
            expanded=False,
        ):
            st.markdown(f"**When:** {_format_time_str(ev.get('start',''))} → {_format_time_str(ev.get('end',''))}")
            loc = ev.get("location", "")
            if loc:
                st.markdown(f"**Location:** {loc}")
            st.code(ev.get("id", ""), language="text")

    confs = payload.get("conflicts") or []
    if confs:
        st.warning("Conflicts detected:")
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
    proposal = payload.get("proposal") or payload.get("target") or payload.get("event") or {}
    if proposal:
        with st.expander("Proposed details", expanded=False):
            st.json(proposal)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Confirm", key="confirm_btn"):
            st.session_state["pending_confirm"] = {"user_input": last_user_text, "confirm": True}
            st.experimental_rerun()
    with col2:
        if st.button("❌ Cancel", key="cancel_btn"):
            st.session_state["pending_confirm"] = None
            st.success("Cancelled.")
            st.experimental_rerun()


# -----------------------------
# Invoke workflow and display
# -----------------------------
def run_turn(user_text: str, confirm: bool = False):
    with st.status("🤖 Thinking…", expanded=False) as status:
        status.update(label="🤖 Thinking… (routing)")

        out = st.session_state["wf"].invoke(
            {
                "user_input": user_text,
                "agent_messages": st.session_state["agent_messages_state"],
                "confirm": confirm,
                "history": st.session_state["llm_history"],  # pass rolling chat history to DefaultAgent
            }
        )

        st.session_state["agent_messages_state"] = out.get("agent_messages", [])
        st.session_state["llm_history"] = out.get("history", st.session_state["llm_history"])

        current_agent = out.get("current_agent") or "default"
        status.update(label=f"🤖 Thinking… (agent: {current_agent})")

    last = st.session_state["agent_messages_state"][-1] if st.session_state["agent_messages_state"] else {}
    sender = last.get("sender", current_agent)
    mtype = last.get("message_type")
    payload = last.get("payload") or {}

    with st.chat_message("assistant", avatar="🤖"):
        # confirmation flow
        if payload.get("requires_confirmation"):
            render_confirmation(payload, user_text)
            return

        # agent-specific renders
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
            # default chat path
            text = (payload.get("result") or "").strip()

            # Robust fallback if any legacy handler emitted sentinel/empty
            if not text or text == "(No LLM configured)." or payload.get("fallback_llm"):
                gem = st.session_state.get("gem") or _get_gemini()
                if not gem:
                    k = os.getenv("GEMINI_API_KEY") or ""
                    st.warning(f"(No LLM configured — GEMINI_API_KEY={'present' if k else 'missing'}.)")
                    return

                # build a lightweight context from history
                conv = st.session_state["llm_history"][-10:]
                hist_lines: List[str] = []
                for h in conv:
                    hist_lines.append(f"{h['role'].capitalize()}: {h['content']}")
                hist_lines.append(f"User: {user_text}")
                prompt_for_fallback = "\n".join(hist_lines)

                text = (gem.chat(prompt_for_fallback) or "").strip()
                if text and "anything else i can help with" not in text.lower():
                    text += "\n\nAnything else I can help with?"

                # keep history in sync when default handled in UI
                st.session_state["llm_history"].append({"role": "user", "content": user_text})
                st.session_state["llm_history"].append({"role": "assistant", "content": text})

            st.write(text)

    # UI transcript
    st.session_state["chat"].append({"role": "user", "content": user_text})
    st.session_state["chat"].append({"role": "assistant", "content": payload})


# -----------------------------
# Sidebar: status & controls
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
    st.write("**Notes**")
    st.markdown("- Sending emails and modifying calendar require confirmation.")
    st.markdown("- Timezone assumed: **Africa/Johannesburg**.")
    st.markdown("- Normal chat uses **Gemini** when no agent is selected.")
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
            gem = st.session_state.get("gem") or _get_gemini()
            if not gem:
                st.error("No LLM configured.")
            else:
                st.write(gem.chat("Say 'hello' in one short sentence."))


# -----------------------------
# Main page
# -----------------------------
st.title("Personal AI Assistant")

# Show previous conversation
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
                st.write(content)

# Auto-run a pending confirmation if present
if st.session_state["pending_confirm"]:
    pc = st.session_state["pending_confirm"]
    run_turn(pc["user_input"], confirm=True)
    st.session_state["pending_confirm"] = None

# Chat input
prompt = st.chat_input(
    "Type a message, e.g. “show unread emails from the last 7 days”, "
    "“create event 'Demo' tomorrow 10:00-10:30”, or just chat..."
)
if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    run_turn(prompt, confirm=False)
