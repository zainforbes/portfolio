# src/core/langgraph_workflow.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
from copy import deepcopy

from langgraph.graph import StateGraph, END
from .state_schema import AssistantState

# MCP tools
from src.mcp_integration.mcp_client import MCPClient
from src.mcp_integration.search_server import web_search
from src.mcp_integration.gmail_server import (
    list_recent_emails,
    read_email,
    send_email,
)
from src.mcp_integration.calendar_server import (
    list_events,
    create_event,
    update_event,
    delete_event,
)

# LLM + agents
from src.utils.gemini_client import GeminiClient
from src.agents.default_agent import DefaultAgent
from src.agents.email_agent import EmailAgent
from src.agents.calendar_agent import CalendarAgent
from src.agents.search_agent import SearchAgent

# Planner + synthesizer
from src.intelligence.planner import make_plan
try:
    from src.intelligence.synthesizer import micro_summarize, final_summarize
except Exception:
    # Fallbacks if you don’t have a synthesizer module
    def micro_summarize(user_text, step, result, gem):
        step_desc = f"{step.get('agent','?')}.{step.get('tool','?')}"
        return {
            "summary": f"Done: {step_desc}.",
            "followup": None,
        }
    def final_summarize(user_text, history, plan, results, gem):
        return "All set."

# ---------- MCP + Agents ----------
MCP = MCPClient()
MCP.register_tool("web_search",        web_search)
MCP.register_tool("gmail_list_recent", list_recent_emails)
MCP.register_tool("gmail_read",        read_email)
MCP.register_tool("gmail_send",        send_email)
MCP.register_tool("gcal_list_events",  list_events)
MCP.register_tool("gcal_create_event", create_event)
MCP.register_tool("gcal_update_event", update_event)
MCP.register_tool("gcal_delete_event", delete_event)

GEM = GeminiClient()

AGENTS: Dict[str, Any] = {
    "default":  DefaultAgent(gemini=GEM, mcp=MCP),
    "email":    EmailAgent(MCP, gemini=GEM),
    "calendar": CalendarAgent(MCP, gemini=GEM),
    "search":   SearchAgent(MCP, gemini=GEM),
}

# ---------- helpers ----------
def add_msg(state: AssistantState, sender: str, msg_type: str, payload: Dict[str, Any]):
    msgs = state.get("agent_messages") or []
    msgs.append({"sender": sender, "message_type": msg_type, "payload": payload})
    state["agent_messages"] = msgs

def _ensure_mem(state: AssistantState) -> Dict[str, Any]:
    mem = state.get("memory")
    if not isinstance(mem, dict):
        mem = {}
        state["memory"] = mem
    return mem

def _store_assigned(state: AssistantState, assign: str, value: Dict[str, Any]):
    mem = _ensure_mem(state)
    mem[assign] = value
    mem["_last"] = value  # convenience

def _merge_memory_patch(state: AssistantState, payload: Dict[str, Any]):
    patch = payload.get("memory_patch")
    if isinstance(patch, dict) and patch:
        mem = _ensure_mem(state)
        mem.update(patch)

# Mutating actions that require confirmation
_MUTATING = {
    ("email", "gmail_send"),
    ("calendar", "gcal_create_event"),
    ("calendar", "gcal_update_event"),
    ("calendar", "gcal_delete_event"),
}

# ---------- Nodes (async) ----------
async def analyze(state: AssistantState) -> AssistantState:
    """
    Gemini plans. If a confirm_context is provided with confirm=True,
    we short-circuit and run that as a one-step plan.
    """
    _ensure_mem(state)

    # Fast path: typed confirm to execute a stored step
    if state.get("confirm") and state.get("confirm_context"):
        step = deepcopy(state["confirm_context"])
        state["plan"] = {"steps": [step], "thinking": ["confirm-context"], "explain": "Running your confirmed action."}
        state["trace"] = {"thinking": ["confirm-context"], "steps": [step], "explain": "Running your confirmed action."}
        state["step_index"] = 0
        state.pop("pending_clarify", None)
        add_msg(state, "planner", "trace", state["trace"])
        return state

    user_text = state.get("user_input", "")
    history   = state.get("history", []) or []
    mem       = state.get("memory", {})

    # Call planner; try to pass memory if your signature supports it
    try:
        plan = make_plan(user_text, history, GEM, memory=mem)  # new signature (if you updated planner)
    except TypeError:
        plan = make_plan(user_text, history, GEM)              # fallback to old signature

    state["plan"] = plan
    state["trace"] = {
        "thinking": plan.get("thinking", []),
        "steps":    plan.get("steps",    []),
        "explain":  plan.get("explain",  ""),
    }
    state["step_index"] = 0
    state["results"] = state.get("results") or {}
    state["working"] = state.get("working") or {}

    if plan.get("clarify"):
        state["pending_clarify"] = plan["clarify"]
    else:
        state.pop("pending_clarify", None)

    add_msg(state, "planner", "trace", state["trace"])
    return state

def route_from_analyze(state: AssistantState):
    if state.get("pending_clarify"):
        return "clarify"
    if not state.get("plan", {}).get("steps"):
        return "finalize"
    return "execute_step"

async def clarify(state: AssistantState) -> AssistantState:
    q = state.get("pending_clarify", "Could you clarify?")
    add_msg(state, "planner", "response", {"result": q})
    state["await_user"] = True
    return state

async def execute_step(state: AssistantState) -> AssistantState:
    steps = (state.get("plan") or {}).get("steps", [])
    i     = state.get("step_index", 0)
    if i >= len(steps):
        return state

    step       = steps[i]
    agent_name = step.get("agent", "default")
    tool       = step.get("tool", "none")
    args       = step.get("args", {}) or {}
    assign     = step.get("assign") or f"step_{i}"
    state["current_agent"] = agent_name

    # Confirm-gating for mutating actions
    if (agent_name, tool) in _MUTATING and step.get("confirm"):
        if not state.get("confirm"):
            # store confirm_context so user can just type "send" in the next turn
            state["confirm_context"] = step
            add_msg(state, "coordinator", "response", {
                "requires_confirmation": True,
                "message": "I’m ready to proceed. Reply 'send' to continue or 'cancel' to abort.",
                "proposal": step
            })
            state["await_user"] = True
            return state

    result: Any = {"note": f"not executed: {agent_name}.{tool}"}
    agent = AGENTS.get(agent_name)

    # ---- EMAIL ----
    if agent_name == "email":
        state["current_step"] = step  # optional: agents can read it
        if tool == "gmail_list_recent":
            q = (args.get("query") or "").strip()
            # Let EmailAgent handle; pass query via routing.filters.query_extra
            if q:
                state["routing"] = {"filters": {"query_extra": q}}
            out = await agent.execute(state)
            payload = out["agent_messages"][-1]["payload"]
            result  = payload.get("items", [])

            # memory: index map + compact list
            if payload.get("items_compact"):
                imap = {int(k): v for k, v in (payload.get("index_map") or {}).items()}
                mem  = _ensure_mem(state)
                mem["last_email_list"] = {
                    "list": payload["items_compact"],
                    "index_map": imap,
                    "query": payload.get("query"),
                }
                mem["last_email_index_map"] = imap

        elif tool == "gmail_read":
            # Prefer id; if none, try index via memory
            msg_id = args.get("id")
            index  = args.get("index")

            if not msg_id and index is not None:
                imap = state.get("memory", {}).get("last_email_index_map", {})
                msg_id = imap.get(int(index))

            if not msg_id:
                add_msg(state, "email", "response", {"result": "I couldn't resolve that email (missing id/index)."})
                result = {"error": "missing id/index"}
            else:
                detail = await MCP.call_tool("gmail_read", message_id=str(msg_id))
                add_msg(state, "email", "response", {"mode":"read","email": detail})
                result = detail
                _ensure_mem(state)["last_email_read"] = {"email": detail}

        elif tool == "gmail_send":
            to = args.get("to"); subject = (args.get("subject") or "").strip(); body = (args.get("body") or "").strip()
            # Auto-compose body/subject from memory if missing
            mem = state.get("memory", {})
            if not body:
                if mem.get("last_search", {}).get("items"):
                    items = mem["last_search"]["items"]
                    top   = "\n".join(f"- {it.get('title','')} — {it.get('url','')}" for it in items[:5])
                    body  = (mem["last_search"].get("summary") or "Here’s what I found:") + "\n\n" + top
                elif mem.get("last_email_read", {}).get("email"):
                    em = mem["last_email_read"]["email"]
                    quoted = (em.get("body","") or em.get("snippet","") or "")[:2000]
                    body = f"FYI, see below:\n\n---\nFrom: {em.get('from','')}\nSubject: {em.get('subject','')}\n\n{quoted}"
            if not subject:
                subject = "Summary of recent findings" if mem.get("last_search") else ("Forwarded update" if mem.get("last_email_read") else "(no subject)")

            if not (to and subject and body):
                add_msg(state, "email", "response", {
                    "mode": "compose",
                    "requires_confirmation": True,
                    "message": "Ready to send this email. Reply 'send' to proceed or edit details.",
                    "draft": {"to": to or [], "subject": subject, "body": body},
                })
                state["confirm_context"] = {"agent": "email", "tool": "gmail_send", "args": {"to": to, "subject": subject, "body": body}}
                state["await_user"] = True
                return state

            send_res = await MCP.call_tool("gmail_send", to=to, subject=subject, body=body)
            add_msg(state, "email", "response", {"mode":"sent", "result":"Email sent.", "sent_to": to, "subject": subject, "server_result": send_res})
            result = send_res

    # ---- CALENDAR ----
    elif agent_name == "calendar":
        state["current_step"] = step
        if tool == "gcal_list_events":
            tmin = args.get("time_min"); tmax = args.get("time_max")
            if tmin or tmax:
                items = await MCP.call_tool("gcal_list_events", time_min=tmin, time_max=tmax, max_results=args.get("max_results", 20))
                add_msg(state, "calendar", "response", {"items": items, "window": {"time_min": tmin, "time_max": tmax}})
                result = items
                _ensure_mem(state)["last_calendar_view"] = {"events": items, "window": {"time_min": tmin, "time_max": tmax}}
            else:
                out = await agent.execute(state)
                payload = out["agent_messages"][-1]["payload"]
                result = payload.get("items", [])
                _ensure_mem(state)["last_calendar_view"] = {"events": result, "window": payload.get("window")}

        elif tool == "gcal_create_event":
            res = await MCP.call_tool("gcal_create_event", **args)
            add_msg(state, "calendar", "response", {"result":"Event created.", "event": res})
            result = res
            _ensure_mem(state)["last_calendar_mutation"] = {"event": res}

        elif tool == "gcal_update_event":
            res = await MCP.call_tool("gcal_update_event", **args)
            add_msg(state, "calendar", "response", {"result":"Event updated.", "event": res})
            result = res
            _ensure_mem(state)["last_calendar_mutation"] = {"event": res}

        elif tool == "gcal_delete_event":
            res = await MCP.call_tool("gcal_delete_event", **args)
            add_msg(state, "calendar", "response", {"result":"Event deleted.", "target": res})
            result = res

    # ---- SEARCH ----
    elif agent_name == "search":
        state["current_step"] = step
        if tool == "web_search":
            q = args.get("query")
            if not q:
                add_msg(state, "search", "response", {"result":"Missing 'query' for web_search."})
                result = {"error":"missing query"}
            else:
                # Reuse search agent (it reads user_input)
                original = state.get("user_input", "")
                state["user_input"] = q
                out = await agent.execute(state)
                state["user_input"] = original
                payload = out["agent_messages"][-1]["payload"]
                result = payload.get("items", [])
                _ensure_mem(state)["last_search"] = {"items": result, "summary": payload.get("summary_llm")}

    # ---- DEFAULT CHAT ----
    else:
        out = await AGENTS["default"].execute(state)
        payload = out["agent_messages"][-1]["payload"]
        result  = payload.get("result", "(ok)")

    # Save result & advance
    results = state.get("results") or {}
    results[assign] = result
    state["results"] = results
    state["step_index"] = i + 1
    return state

async def micro(state: AssistantState) -> AssistantState:
    steps = (state.get("plan") or {}).get("steps", [])
    i = max(0, state.get("step_index", 0) - 1)
    if i >= len(steps):
        return state

    step   = steps[i]
    assign = step.get("assign") or f"step_{i}"
    result = (state.get("results") or {}).get(assign)
    user_text = state.get("user_input", "")

    m = micro_summarize(user_text, step, result, GEM)
    state["micro_reply"]  = m.get("summary", "")
    state["ask_followup"] = m.get("followup") or None

    if state["micro_reply"]:
        add_msg(state, "planner", "response", {"result": state["micro_reply"]})
    if state.get("ask_followup"):
        add_msg(state, "planner", "response", {"result": state["ask_followup"]})
        state["await_user"] = True
    return state

def route_after_micro(state: AssistantState):
    if state.get("await_user"):
        return "await_user"
    steps = (state.get("plan") or {}).get("steps", [])
    i = state.get("step_index", 0)
    if i < len(steps):
        return "execute_step"
    return "finalize"

async def await_user(state: AssistantState) -> AssistantState:
    return state

async def finalize(state: AssistantState) -> AssistantState:
    user_text = state.get("user_input", "")
    history   = state.get("history", []) or []
    plan      = state.get("plan", {})
    results   = state.get("results", {})

    reply = final_summarize(user_text, history, plan, results, GEM)
    add_msg(state, "default", "response", {"result": reply})

    # roll memory (chat history)
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})
    state["history"] = history[-20:]
    return state

# ---------- Build the graph ----------
def build_workflow():
    graph = StateGraph(AssistantState)
    graph.add_node("analyze", analyze)
    graph.add_node("clarify", clarify)
    graph.add_node("execute_step", execute_step)
    graph.add_node("micro", micro)
    graph.add_node("await_user", await_user)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("analyze")
    graph.add_conditional_edges("analyze", route_from_analyze, {
        "clarify": "clarify",
        "execute_step": "execute_step",
        "finalize": "finalize",
    })
    graph.add_edge("execute_step", "micro")
    graph.add_conditional_edges("micro", route_after_micro, {
        "await_user": "await_user",
        "execute_step": "execute_step",
        "finalize": "finalize",
    })
    graph.add_edge("await_user", END)
    return graph.compile()
