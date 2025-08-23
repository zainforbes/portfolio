from __future__ import annotations
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
    st.session_state["gem"] = None  # lazy

if "chat" not in st.session_state:
    st.session_state["chat"] = []  # UI transcript (simple)

if "agent_messages_state" not in st.session_state:
    st.session_state["agent_messages_state"] = []  # tool/agent state

if "pending_confirm" not in st.session_state:
    st.session_state["pending_confirm"] = None  # {"user_input": ..., "confirm": True}

if "llm_history" not in st.session_state:
    st.session_state["llm_history"] = []  # rolling chat memory

if "pending_plan" not in st.session_state:
    st.session_state["pending_plan"] = None  # persisted planner plan for resume

if "plan_cursor" not in st.session_state:
    st.session_state["plan_cursor"] = None  # current step index

# Warm-up LLM if key exists
if st.session_state.get("gem") is None and os.getenv("GEMINI_API_KEY"):
    _get_gemini()


# -----------------------------
# Helper renders
# -----------------------------
def _md_link(url: str, text: Optional[str] = None) -> str:
    return f"[{text or url}]({url})"

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

def render_calendar(payload: Dict[str, Any]):
    window = (payload or {}).get("window") or {}
    if window:
        st.caption(f"Window: {window.get('label','upcoming')} | "
                   f"{window.get('time_min','')} → {window.get('time_max','(open)')}")
    if payload.get("summary_llm"):
        st.write(payload["summary_llm"])
    items = (payload or {}).get("items") or []
    if not items:
        st.info("No events found.")
        return
    for ev in items:
        title = ev.get("summary","(no title)")
        start = ev.get("start","")
        end   = ev.get("end","")
        with st.expander(f"{title} | {start} → {end}", expanded=False):
            st.markdown(f"**When:** {start} → {end}")
            loc = ev.get("location","")
            if loc:
                st.markdown(f"**Location:** {loc}")
            st.code(ev.get("id",""), language="text")
    confs = (payload or {}).get("conflicts") or []
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
    details = payload.get("proposal") or payload.get("target") or payload.get("event") or {}
    if details:
        with st.expander("Proposed details", expanded=True):
            st.json(details)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Confirm", key="confirm_btn"):
            st.session_state["pending_confirm"] = {"user_input": last_user_text, "confirm": True}
            st.experimental_rerun()
    with c2:
        if st.button("❌ Cancel", key="cancel_btn"):
            st.session_state["pending_confirm"] = None
            # Clear pending plan/cursor if any
            st.session_state["pending_plan"] = None
            st.session_state["plan_cursor"] = None
            st.success("Cancelled.")
            st.experimental_rerun()

def render_planner_trace(messages: List[Dict[str,Any]]):
    plan_msg = None
    for m in reversed(messages):
        if m.get("sender") == "coordinator" and m.get("message_type") == "plan":
            plan_msg = m
            break
    if not plan_msg:
        return
    pl = plan_msg.get("payload", {})
    with st.expander("🧠 Reasoning trace (planner)", expanded=False):
        thoughts = pl.get("thoughts") or []
        if thoughts:
            st.markdown("**Plan:**")
            for t in thoughts:
                st.markdown(f"- {t}")
        steps = pl.get("steps") or []
        if steps:
            st.markdown("**Steps:**")
            for s in steps:
                badge = "🟢" if not s.get("confirmation_required") else "🟡"
                st.markdown(
                    f"{badge} **{s.get('i')}** · `{s.get('mode')}` → **{s.get('agent')}**.{s.get('tool')}  "
                    f"&nbsp;&nbsp; _{s.get('explain','')}_"
                )
        fol = pl.get("followups") or []
        if fol:
            st.markdown("**Possible next:**")
            for f in fol:
                st.markdown(f"- {f}")


# -----------------------------
# Invoke workflow and display
# -----------------------------
def run_turn(user_text: str, confirm: bool = False):
    with st.status("🤖 Thinking…", expanded=False) as status:
        status.update(label="🤖 Thinking… (planner)")
        # pass through any saved pending plan/cursor to the workflow
        out = st.session_state["wf"].invoke({
            "user_input": user_text,
            "agent_messages": st.session_state["agent_messages_state"],
            "confirm": confirm,
            "history": st.session_state["llm_history"],
            "pending_plan": st.session_state.get("pending_plan"),
            "plan_cursor": st.session_state.get("plan_cursor"),
        })

        st.session_state["agent_messages_state"] = out.get("agent_messages", [])
        # capture plan/cursor/blackboard from the graph state (if the coordinator set them)
        st.session_state["pending_plan"] = out.get("pending_plan")
        st.session_state["plan_cursor"] = out.get("plan_cursor")
        st.session_state["llm_history"] = out.get("history", st.session_state["llm_history"])

        # who did the last speaking?
        last = st.session_state["agent_messages_state"][-1] if st.session_state["agent_messages_state"] else {}
        sender = last.get("sender")
        mtype  = last.get("message_type")
        payload= last.get("payload") or {}

    # Show planner trace from this turn
    render_planner_trace(st.session_state["agent_messages_state"])

    with st.chat_message("assistant", avatar="🤖"):
        if payload.get("requires_confirmation"):
            render_confirmation(payload, user_text)
            return

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
            # default/other
            text = (payload.get("summary_llm") or payload.get("result") or "").strip()
            if not text:
                # final fallback to LLM chat if absolutely nothing came back
                gem = st.session_state.get("gem") or _get_gemini()
                if gem:
                    hist = st.session_state["llm_history"][-10:]
                    ctx = "\n".join([f"{h['role'].capitalize()}: {h['content']}" for h in hist] + [f"User: {user_text}"])
                    text = gem.chat(ctx) or ""
            st.write(text or "Done.")

    # Append to transcript
    st.session_state["chat"].append({"role": "user", "content": user_text})
    st.session_state["chat"].append({"role": "assistant", "content": payload})


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
    st.write("**Notes**")
    st.markdown("- Mutating actions (send email, create/update/delete events) prompt for confirmation.")
    st.markdown("- Planner can run multiple steps per turn; complex tasks may pause and resume.")
    st.markdown("- Timezone assumed: **Africa/Johannesburg**.")
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

# show the simple transcript (optional)
for msg in st.session_state["chat"]:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            content = msg["content"]
            if isinstance(content, dict):
                if "summary_llm" in content and isinstance(content["summary_llm"], str):
                    st.write(content["summary_llm"])
                elif isinstance(content.get("result"), str):
                    st.write(content["result"])
                else:
                    st.json(content)
            else:
                st.write(content)

# Auto-run a pending confirmation (resume plan)
if st.session_state["pending_confirm"]:
    pc = st.session_state["pending_confirm"]
    run_turn(pc["user_input"], confirm=True)
    st.session_state["pending_confirm"] = None

# Chat input
prompt = st.chat_input("Type a message… e.g. “show unread emails from the last 7 days”, "
                       "“create event 'Demo' tomorrow 10:00–10:30”, or just chat…")
if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    run_turn(prompt, confirm=False)
