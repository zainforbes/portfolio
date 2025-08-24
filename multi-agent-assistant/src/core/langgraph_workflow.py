# src/core/langgraph_workflow.py
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from .state_schema import AssistantState

# MCP tools
from src.mcp_integration.mcp_client import MCPClient
from src.mcp_integration.search_server import web_search
from src.mcp_integration.gmail_server import (
    list_recent_emails,
    read_email,
    send_email,            # already registered in your code
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

# Planner + synthesizer (Gemini-driven brain)
from src.intelligence.planner import make_plan
from src.intelligence.synthesizer import micro_summarize, final_summarize


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
    msgs.append({"sender": sender, "message_type": msg_type, "payload": payload, "notes": ""})
    state["agent_messages"] = msgs

# Mutating actions that require confirm=true
_MUTATING = {
    ("email", "gmail_send"),
    ("calendar", "gcal_create_event"),
    ("calendar", "gcal_update_event"),
    ("calendar", "gcal_delete_event"),
}


# ---------- Nodes (async) ----------

async def analyze(state: AssistantState) -> AssistantState:
    """Gemini plans in natural language (tools, steps, clarifications)."""
    user_text = state.get("user_input", "")
    history   = state.get("history", []) or []

    plan = make_plan(user_text, history, GEM)
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

    # Surface reasoning to UI
    add_msg(state, "planner", "trace", state["trace"])
    return state


def route_from_analyze(state: AssistantState):
    if state.get("pending_clarify"):
        return "clarify"
    if not state.get("plan", {}).get("steps"):
        return "finalize"
    return "execute_step"


async def clarify(state: AssistantState) -> AssistantState:
    """Ask Gemini’s clarification question and pause the run."""
    q = state.get("pending_clarify", "Could you clarify?")
    add_msg(state, "planner", "response", {"result": q})
    state["await_user"] = True
    return state


async def execute_step(state: AssistantState) -> AssistantState:
    """
    Execute a single planner step.
    - Uses your existing Email/Calendar/Search agents for list/read.
    - Calls MCP tools directly for mutating actions (send/create/update/delete).
    - Stores result in state["results"][assign].
    """
    steps = (state.get("plan") or {}).get("steps", [])
    i = state.get("step_index", 0)
    if i >= len(steps):
        return state

    step = steps[i]
    agent_name = step.get("agent", "default")
    tool       = step.get("tool", "none")
    args       = step.get("args", {}) or {}
    assign     = step.get("assign") or f"step_{i}"

    state["current_agent"] = agent_name

    # Confirmation required for mutating actions
    if (agent_name, tool) in _MUTATING and step.get("confirm"):
        if not state.get("confirm"):
            add_msg(state, "coordinator", "response", {
                "requires_confirmation": True,
                "message": "I’m about to make changes. Do you want me to proceed?",
                "proposal": step
            })
            state["await_user"] = True
            return state

    # Make planner intention visible to agents (optional)
    state["planner_instruction"] = step.get("instruction", "")

    result: Any = {"note": f"not executed: {agent_name}.{tool}"}
    agent = AGENTS.get(agent_name)

    # ---- EMAIL ----
    if agent_name == "email":
        if tool == "gmail_list_recent":
            # Reuse your EmailAgent; supply a query if provided by planner
            # Your EmailAgent builds a Gmail query from routing.filters
            q = (args.get("query") or "").strip()
            if q:
                state["routing"] = {"filters": {"query_extra": q}}
            out = await agent.execute(state)
            payload = out["agent_messages"][-1]["payload"]
            result = payload.get("items", [])

        elif tool == "gmail_read":
            email_id = args.get("id")
            if email_id:
                result = await MCP.call_tool("gmail_read", id=email_id)
                add_msg(state, "email", "response", {"items": [result]})
            else:
                result = {"error": "Missing 'id' for gmail_read"}

        elif tool == "gmail_send":
            # Mutating (already gated above if confirm=true)
            to = args.get("to"); subject = args.get("subject"); body = args.get("body")
            if not (to and subject and body):
                result = {"error": "Missing fields for gmail_send (to,subject,body)"}
            else:
                result = await MCP.call_tool("gmail_send", to=to, subject=subject, body=body)
                add_msg(state, "email", "response", {"result": "Email sent.", "target": result})

    # ---- CALENDAR ----
    elif agent_name == "calendar":
        if tool == "gcal_list_events":
            # If the planner passed explicit RFC3339 bounds, call MCP directly.
            tmin = args.get("time_min"); tmax = args.get("time_max")
            if tmin or tmax:
                items = await MCP.call_tool("gcal_list_events", time_min=tmin, time_max=tmax, max_results=args.get("max_results", 20))
                add_msg(state, "calendar", "response", {"items": items, "window": {"time_min": tmin, "time_max": tmax}})
                result = items
            else:
                # Otherwise reuse your CalendarAgent’s NL window parsing on user_input
                out = await agent.execute(state)
                payload = out["agent_messages"][-1]["payload"]
                result = payload.get("items", [])

        elif tool == "gcal_create_event":
            # expected args: summary, start/dateTime/local, end/dateTime/local, location, attendees, etc.
            result = await MCP.call_tool("gcal_create_event", **args)
            add_msg(state, "calendar", "response", {"result": "Event created.", "event": result})

        elif tool == "gcal_update_event":
            result = await MCP.call_tool("gcal_update_event", **args)
            add_msg(state, "calendar", "response", {"result": "Event updated.", "event": result})

        elif tool == "gcal_delete_event":
            result = await MCP.call_tool("gcal_delete_event", **args)
            add_msg(state, "calendar", "response", {"result": "Event deleted.", "target": result})

    # ---- SEARCH ----
    elif agent_name == "search":
        if tool == "web_search":
            q = args.get("query")
            if q:
                # Reuse search agent – it expects user_input to contain the query
                old = state.get("user_input", "")
                state["user_input"] = q
                out = await agent.execute(state)
                state["user_input"] = old
                payload = out["agent_messages"][-1]["payload"]
                result = payload.get("items", [])
            else:
                result = {"error": "Missing 'query' for web_search"}

    # ---- DEFAULT CHAT ----
    elif agent_name == "default":
        out = await agent.execute(state)
        payload = out["agent_messages"][-1]["payload"]
        result = payload.get("result", "(ok)")

    # Save result & advance
    results = state.get("results") or {}
    results[assign] = result
    state["results"] = results
    state["step_index"] = i + 1

    return state


async def micro(state: AssistantState) -> AssistantState:
    """Tiny Gemini synthesis after each step; may ask a follow-up and pause."""
    steps = (state.get("plan") or {}).get("steps", [])
    i = max(0, state.get("step_index", 0) - 1)
    if i >= len(steps):
        return state

    step = steps[i]
    assign = step.get("assign") or f"step_{i}"
    result = (state.get("results") or {}).get(assign)
    user_text = state.get("user_input", "")

    m = micro_summarize(user_text, step, result, GEM)
    state["micro_reply"] = m.get("summary", "")
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
    """No-op node that ends this run; UI should collect user reply / confirm."""
    return state


async def finalize(state: AssistantState) -> AssistantState:
    """Final Gemini synthesis for the turn (natural response)."""
    user_text = state.get("user_input", "")
    history   = state.get("history", []) or []
    plan      = state.get("plan", {})
    results   = state.get("results", {})

    reply = final_summarize(user_text, history, plan, results, GEM)
    add_msg(state, "default", "response", {"result": reply})

    # roll memory (20 turns)
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})
    state["history"] = history[-20:]

    return state

def synthesize_node(state: AssistantState) -> AssistantState:
    """
    Produce a natural final line, but NEVER claim a send unless payload says so.
    """
    msgs = state.get("agent_messages") or []
    if not msgs:
        return state

    last = msgs[-1]
    sender = last.get("sender")
    payload = last.get("payload") or {}

    line = ""

    if sender == "email":
        mode = (payload or {}).get("mode")
        if mode == "compose":
            d = payload.get("draft") or {}
            to = ", ".join(d.get("to", [])) or "recipient"
            subj = d.get("subject", "(no subject)")
            line = f"Draft ready to {to} — “{subj}”. Reply **send** to send, or tell me edits."
        elif mode == "sent":
            to = ", ".join(payload.get("sent_to", [])) or "recipient"
            subj = payload.get("subject", "(no subject)")
            line = f"✅ Email sent to {to} — “{subj}”. Anything else?"
        elif mode == "read":
            subj = ((payload.get("email") or {}).get("subject") or "(no subject)")
            line = f"Opened that email: “{subj}”. Want me to reply or do anything else?"
        else:
            # list mode or other
            cnt = payload.get("count")
            if isinstance(cnt, int):
                line = f"Here are your {cnt} recent emails. Say 'read #2' to open one."
            else:
                line = "Here are the email results."

    elif sender == "calendar":
        w = (payload.get("window") or {}).get("label")
        if w:
            line = f"Calendar for {w} shown above."
        elif payload.get("event"):
            line = "Calendar update completed."
        else:
            line = "Calendar results updated."

    elif sender == "search":
        if payload.get("summary_llm"):
            line = "I’ve summarized the top results above."
        else:
            line = "Here are the top web results."

    elif sender == "default":
        line = (payload.get("result") or "").strip()

    if line:
        state["agent_messages"].append({
            "sender": "coordinator",
            "message_type": "response",
            "payload": {"result": line}
        })
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


if __name__ == "__main__":
    wf = build_workflow()
    print(wf.invoke({"user_input": "check my unread emails", "agent_messages": [], "history": []}))
