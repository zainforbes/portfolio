# src/agents/coordinator_agent.py
from __future__ import annotations
from typing import Dict, Any, Optional
from copy import deepcopy
import re

from src.intelligence.planner import plan_with_gemini

def _substitute_args(args: Any, blackboard: Dict[str, Any]) -> Any:
    """Replace {{var}} in strings with blackboard[var]. Works recursively."""
    if isinstance(args, dict):
        return {k: _substitute_args(v, blackboard) for k, v in args.items()}
    if isinstance(args, list):
        return [_substitute_args(x, blackboard) for x in args]
    if isinstance(args, str):
        def repl(m):
            key = m.group(1)
            return str(blackboard.get(key, m.group(0)))
        return re.sub(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}", repl, args)
    return args

class CoordinatorAgent:
    """
    Runs a Gemini plan:
      - shows a brief 'plan' message with thoughts
      - executes steps in order
      - pauses if a step requires confirmation (and confirm flag not set)
      - resumes after confirmation using saved plan + cursor
      - stores step results in a shared 'blackboard' (state['blackboard'])

    Supports two execution modes per step:
      - mode='agent': route to the named agent .execute(state)
      - mode='mcp'  : call the MCP tool by name directly (self.mcp.get_tool(name))
    """
    def __init__(self, agents: Dict[str, Any], gemini=None, mcp=None):
        self.agents = agents
        self.gemini = gemini
        self.mcp = mcp  # used for 'mcp' steps

    async def _run_agent_step(self, agent_name: str, tool_args: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        agent = self.agents.get(agent_name) or self.agents.get("default")
        # agents read filters from state["routing"]["filters"]
        state["current_agent"] = agent_name
        state["routing"] = {"filters": tool_args}
        return await agent.execute(state)

    async def _run_mcp_step(self, tool_name: str, tool_args: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        if not self.mcp or tool_name == "none":
            # degrade to default
            state["agent_messages"].append({
                "sender":"coordinator","message_type":"error",
                "payload":{"agent":"coordinator","message":f"MCP tool '{tool_name}' unavailable."}
            })
            return state
        tool = self.mcp.tools.get(tool_name)
        if not tool:
            state["agent_messages"].append({
                "sender":"coordinator","message_type":"error",
                "payload":{"agent":"coordinator","message":f"MCP tool '{tool_name}' not registered."}
            })
            return state
        try:
            result = await tool(**tool_args)
            state["agent_messages"].append({
                "sender":"coordinator","message_type":"response",
                "payload":{"tool":tool_name, "result":result}
            })
            return state
        except Exception as e:
            state["agent_messages"].append({
                "sender":"coordinator","message_type":"error",
                "payload":{"agent":"coordinator","message":"Tool error","raw":str(e)}
            })
            return state

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_text = state.get("user_input","")
        history   = state.get("history", []) or state.get("llm_history", []) or []
        confirm   = bool(state.get("confirm", False))

        state.setdefault("agent_messages", [])
        state.setdefault("blackboard", {})
        blackboard = state["blackboard"]

        # Reuse a pending plan+cursor if present; else make a fresh plan
        plan = state.get("pending_plan") or plan_with_gemini(user_text, history, self.gemini)
        steps = plan.get("steps", [])
        cursor = int(state.get("plan_cursor") or 0)

        # Emit a compact reasoning trace for the UI
        state["agent_messages"].append({
            "sender": "coordinator",
            "message_type": "plan",
            "payload": {
                "thoughts": plan.get("thoughts", []),
                "followups": plan.get("followups", []),
                "steps": [{"i":i, "agent":s.get("agent"), "mode":s.get("mode"), "tool":s.get("tool"),
                           "confirmation_required": bool(s.get("confirmation_required",False)),
                           "explain": s.get("explain","")} for i,s in enumerate(steps)]
            }
        })

        # Clarify if planner produced no steps and a question
        if not steps and plan.get("clarify_question"):
            state["agent_messages"].append({
                "sender":"default","message_type":"response",
                "payload":{"result": plan["clarify_question"]}
            })
            return state

        # Execute steps sequentially (cap per turn to avoid runaway)
        MAX_STEPS_PER_TURN = 4
        ran = 0
        while cursor < len(steps) and ran < MAX_STEPS_PER_TURN:
            step = steps[cursor]
            tool_args = _substitute_args(deepcopy(step.get("args", {})), blackboard)

            # Pause for confirmation on mutating steps
            if step.get("confirmation_required", False) and not confirm:
                # Surface a confirmation card to the UI and save plan/cursor for resume
                state["agent_messages"].append({
                    "sender":"coordinator","message_type":"response",
                    "payload":{
                        "requires_confirmation": True,
                        "message": "Please confirm this action.",
                        "proposal": {"step_index": cursor, "agent": step.get("agent"),
                                     "mode": step.get("mode"), "tool": step.get("tool"),
                                     "args": tool_args},
                    }
                })
                # persist plan + cursor for next turn
                state["pending_plan"] = plan
                state["plan_cursor"] = cursor
                return state

            # Run the step
            if step.get("mode","agent") == "mcp":
                state = await self._run_mcp_step(step.get("tool","none"), tool_args, state)
            else:
                state = await self._run_agent_step(step.get("agent","default"), tool_args, state)

            # Capture the last payload onto the blackboard if requested
            assign = (step.get("assign") or "").strip()
            if assign:
                last = state["agent_messages"][-1] if state["agent_messages"] else {}
                bb_val = last.get("payload") or last.get("result") or last
                blackboard[assign] = bb_val

            cursor += 1
            ran += 1

        # If we exhausted steps, clear pending state
        if cursor >= len(steps):
            state.pop("pending_plan", None)
            state.pop("plan_cursor", None)
        else:
            # Save progress for next turn (if more steps remain)
            state["pending_plan"] = plan
            state["plan_cursor"] = cursor

        # Post-summary (LLM explains results, suggests next actions)
        if self.gemini:
            try:
                last = state["agent_messages"][-1] if state["agent_messages"] else {}
                payload = last.get("payload") or {}
                if isinstance(payload, dict) and "summary_llm" not in payload:
                    ctx = str(payload)[:2000]
                    prompt = (
                        "Summarize the result in 2–4 sentences. "
                        "If appropriate, suggest 1–3 concrete next actions. Avoid filler."
                    )
                    payload["summary_llm"] = self.gemini.chat(ctx + "\n\n" + prompt)
                    last["payload"] = payload
            except Exception:
                pass

        return state
