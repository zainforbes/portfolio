# src/core/langgraph_workflow.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
from copy import deepcopy
import json
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
    def micro_summarize(user_text, step, result, gem):
        step_desc = f"{step.get('agent','?')}.{step.get('tool','?')}"
        return {"summary": f"Done: {step_desc}.", "followup": None}
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

def _merge_memory_patch(state: AssistantState, payload: Dict[str, Any]):
    patch = payload.get("memory_patch")
    if isinstance(patch, dict) and patch:
        mem = _ensure_mem(state)
        mem.update(patch)

def _get_from_memory(state: Dict[str, Any], path: str) -> str:
    """
    Enhanced memory path resolution that handles multiple storage patterns.
    Supports paths like:
    - "search.last_summary" (current storage)
    - "search.langgraph_summary.summary" (planner generated)
    - "search.react_features.summary" (planner generated)
    """
    if not path:
        return ""
        
    mem = state.get("memory", {})
    parts = path.split(".")
    
    # Direct path resolution first
    cur = mem
    for part in parts:
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            # Path not found, try fallback patterns
            break
    else:
        # Successfully resolved the full path
        return cur if isinstance(cur, str) else ""
    
    # Fallback patterns for common mismatches
    if parts[0] == "search":
        search_mem = mem.get("search", {})
        
        # Try common search result locations
        fallback_paths = [
            "last_summary",  # Current storage location
            "summary",       # Alternative
        ]
        
        for fallback in fallback_paths:
            if fallback in search_mem:
                result = search_mem[fallback]
                if isinstance(result, str) and result.strip():
                    return result
        
        # Try last_search location (used by some agents)
        last_search = mem.get("last_search", {})
        if "summary" in last_search:
            result = last_search["summary"]
            if isinstance(result, str) and result.strip():
                return result
    
    return ""

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
    Enhanced planner with workflow context awareness - FIXED VERSION
    """
    _ensure_mem(state)

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
    current_plan = state.get("plan")  # Pass current plan for context

    # Enhanced planning with context awareness
    try:
        plan = make_plan(user_text, history, GEM, memory=mem, current_plan=current_plan)
        print(f"DEBUG - User input: {user_text}")
        print(f"DEBUG - Generated plan: {json.dumps(plan, indent=2)}")
    except TypeError:
        # Fallback for older signature
        plan = make_plan(user_text, history, GEM)

    workflow_type = plan.get("workflow_type", "new")
    print(f"DEBUG - Workflow type: {workflow_type}")

    # CRITICAL FIX: Always use the new plan from the planner
    # The planner has already decided what to do based on context
    state["plan"] = plan
    state["step_index"] = 0  # Start from beginning of new plan
    state["workflow_type"] = workflow_type

    # Handle workflow type metadata for UI
    if workflow_type == "modify":
        modify_step = plan.get("modify_step")
        if modify_step is not None:
            # Backup original if not already backed up
            if not state.get("original_plan"):
                state["original_plan"] = deepcopy(current_plan)
    elif workflow_type == "new":
        # Clear any previous workflow state
        state["original_plan"] = None

    # Update trace for UI
    state["trace"] = {
        "workflow_type": workflow_type,
        "thinking": plan.get("thinking", []),
        "steps":    plan.get("steps",    []),
        "explain":  plan.get("explain",  ""),
        "modify_step": plan.get("modify_step") if workflow_type == "modify" else None,
    }
    
    state["results"] = state.get("results") or {}
    state["working"] = state.get("working") or {}

    if plan.get("clarify"):
        state["pending_clarify"] = plan["clarify"]
    else:
        state.pop("pending_clarify", None)

    add_msg(state, "planner", "trace", state["trace"])
    print(f"DEBUG - Final plan steps count: {len(state['plan'].get('steps', []))}")
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
    i = state.get("step_index", 0)
    print(f"DEBUG execute_step START: step_index={i}, total_steps={len(steps)}")
    print(f"DEBUG execute_step START: steps={steps}")
    if i >= len(steps):
        return state

    step = steps[i]
    agent_name = step.get("agent", "default")
    tool = step.get("tool", "none")
    args = step.get("args", {}) or {}
    assign = step.get("assign") or f"step_{i}"
    state["current_agent"] = agent_name

    # Confirm-gating for mutating actions
    if (agent_name, tool) in _MUTATING and step.get("confirm"):
        if not state.get("confirm"):
            state["confirm_context"] = step
            add_msg(state, "coordinator", "response", {
                "requires_confirmation": True,
                "message": "I'm ready to proceed. Reply 'send' to continue or 'cancel' to abort.",
                "proposal": step
            })
            state["await_user"] = True
            return state

    result: Any = {"note": f"not executed: {agent_name}.{tool}"}
    agent = AGENTS.get(agent_name)

    # ---- EMAIL ----
    if agent_name == "email":
        state["current_step"] = step
        if tool == "gmail_list_recent":
            q = (args.get("query") or "").strip()
            if q:
                state["routing"] = {"filters": {"query_extra": q}}
            out = await agent.execute(state)
            payload = out["agent_messages"][-1]["payload"]
            result  = payload.get("items", [])
            _merge_memory_patch(state, payload)

        elif tool == "gmail_read":
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
                payload = {"mode": "read", "email": detail, "memory_patch": {"last_email_read": {"email": detail}}}
                add_msg(state, "email", "response", payload)
                _merge_memory_patch(state, payload)
                result = detail

        elif tool == "gmail_send":
            to = args.get("to"); subject = (args.get("subject") or "").strip(); body = (args.get("body") or "").strip()
            mem = state.get("memory", {})
            if not body and mem.get("last_search", {}).get("items"):
                items = mem["last_search"]["items"]
                top   = "\n".join(f"- {it.get('title','')} — {it.get('url','')}" for it in items[:5])
                body  = (mem["last_search"].get("summary") or "Here's what I found:") + "\n\n" + top
            if not body and mem.get("last_email_read", {}).get("email"):
                em = mem["last_email_read"]["email"]
                quoted = (em.get("body","") or em.get("snippet","") or "")[:2000]
                body = f"FYI, see below:\n\n---\nFrom: {em.get('from','')}\nSubject: {em.get('subject','')}\n\n{quoted}"
            if not subject:
                subject = "Summary of recent findings" if mem.get("last_search") else ("Forwarded update" if mem.get("last_email_read") else "(no subject)")
            if not (to and subject and body):
                draft = {"to": to or [], "subject": subject, "body": body}
                payload = {
                    "requires_confirmation": True,
                    "message": "Ready to send this email. Reply 'send' to proceed or edit details.",
                    "mode": "compose",
                    "draft": draft,
                    "agent": "email",
                    "intent": "email.send",
                    "confirm_context": {"agent":"email","tool":"gmail_send","args": draft},
                }
                add_msg(state, "email", "response", payload)
                state["confirm_context"] = payload["confirm_context"]
                state["await_user"] = True
                return state
            send_res = await MCP.call_tool("gmail_send", to=to, subject=subject, body=body)
            payload = {"mode":"sent","result":"Email sent.","sent_to": to,"subject": subject,"server_result": send_res}
            add_msg(state, "email", "response", payload)
            result = send_res

        elif tool == "none":  # Handle email drafting and sending
            action = args.get("action", "compose")
            
            if action == "send":
                # Handle sending
                to = args.get("to")
                subject = args.get("subject", "")
                body = args.get("body", "")
                
                # If empty, pull from memory (last draft)
                if not (to and subject and body):
                    mem = state.get("memory", {})
                    last_draft = mem.get("email", {}).get("last_draft", {})
                    to = to or last_draft.get("to", [])
                    subject = subject or last_draft.get("subject", "")
                    body = body or last_draft.get("body", "")
                
                # Convert single email to list if needed
                if isinstance(to, str):
                    to = [to]
                    
                if not (to and subject and body):
                    payload = {"result": "Cannot send - missing recipient, subject, or body"}
                    add_msg(state, "email", "response", payload)
                    result = {"error": "incomplete email"}
                else:
                    # Actually send the email
                    send_res = await MCP.call_tool("gmail_send", to=to, subject=subject, body=body)
                    payload = {
                        "mode": "sent",
                        "result": "Email sent successfully!",
                        "sent_to": to,
                        "subject": subject,
                        "server_result": send_res
                    }
                    add_msg(state, "email", "response", payload)
                    result = send_res
            else:
                # Handle drafting (compose)
                to = args.get("to")
                subject = args.get("subject", "")
                body = args.get("body", "")
                
                # Handle body_from_memory if present
                body_from_memory = args.get("body_from_memory")
                if body_from_memory and not body:
                    body = _get_from_memory(state, body_from_memory)
                
                # Store draft in memory for later sending
                draft = {"to": to, "subject": subject, "body": body}
                mem = state.setdefault("memory", {})
                email_mem = mem.setdefault("email", {})
                email_mem["last_draft"] = draft
                
                payload = {
                    "mode": "compose",
                    "draft": draft,
                    "message": "Draft email ready for review."
                }
                add_msg(state, "email", "response", payload)
                result = draft

    # ---- CALENDAR ----
    elif agent_name == "calendar":
        state["current_step"] = step
        out = await agent.execute(state)
        payload = out["agent_messages"][-1]["payload"]
        # for view/mutations, agent already added a response; capture memory patch if present
        _merge_memory_patch(state, payload)
        # Prefer the primary object as step result
        if "event" in payload:
            result = payload["event"]
        elif "items" in payload:
            result = payload["items"]
        else:
            result = payload

    # ---- SEARCH ----
    elif agent_name == "search":
        state["current_step"] = step
        q = args.get("query")
        if not q:
            add_msg(state, "search", "response", {"result":"Missing 'query' for web_search."})
            result = {"error":"missing query"}
        else:
            original = state.get("user_input", "")
            state["user_input"] = q
            out = await agent.execute(state)
            state["user_input"] = original
            payload = out["agent_messages"][-1]["payload"]
            result = payload.get("items", [])
            _merge_memory_patch(state, payload)

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
    print(f"DEBUG execute_step END: step_index now={state['step_index']}")
    print(f"DEBUG execute_step END: plan steps still={len(state.get('plan', {}).get('steps', []))}")
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

    # NOTE: do NOT pause automatically just because we have an ask_followup.
    # We only pause if the NEXT step truly requires user input/confirmation.
    return state

def route_after_micro(state: AssistantState):
   steps = (state.get("plan") or {}).get("steps", [])
   i = state.get("step_index", 0)
   
   print(f"DEBUG route_after_micro: step_index={i}, total_steps={len(steps)}")
   print(f"DEBUG route_after_micro: steps = {steps}")
   
   # If there is a next step, check if it will need confirmation (and we don't have it yet).
   if i < len(steps):
       next_step = steps[i]
       agent = next_step.get("agent", "")
       tool  = next_step.get("tool", "")
       print(f"DEBUG route_after_micro: next_step agent={agent}, tool={tool}")
       
       needs_confirm = (agent, tool) in _MUTATING and next_step.get("confirm", False)
       print(f"DEBUG route_after_micro: needs_confirm={needs_confirm}")
       
       if needs_confirm and not state.get("confirm"):
           print("DEBUG route_after_micro: -> await_user (needs confirmation)")
           # show any followup already, but pause for user confirmation
           if state.get("ask_followup"):
               add_msg(state, "planner", "response", {"result": state["ask_followup"]})
           state["await_user"] = True
           return "await_user"
       # Otherwise, continue automatically to keep multi-step flows fluid.
       print("DEBUG route_after_micro: -> execute_step (continuing)")
       return "execute_step"

   # No more steps: optionally show followup, then finalize.
   print("DEBUG route_after_micro: No more steps")
   if state.get("ask_followup"):
       print("DEBUG route_after_micro: -> await_user (has followup)")
       add_msg(state, "planner", "response", {"result": state["ask_followup"]})
       state["await_user"] = True
       return "await_user"
   print("DEBUG route_after_micro: -> finalize")
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

    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})
    state["history"] = history[-20:]
    return state

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