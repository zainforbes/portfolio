# src/core/langgraph_workflow.py
"""
LangGraph-style workflow foundation.

This module exposes AssistantGraph which builds a small state graph:
  - classify_request_node: uses GeminiClient.classify(...) to pick an agent
  - agent_router_node: sets route and optionally calls an agent handler
  - verification_node: minimal verification stub
  - response_generator_node: composes final_response

The graph uses `langgraph.StateGraph` if present; otherwise it falls back
to an internal SimpleStateGraph which executes nodes in sequence.

AssistantGraph accepts optional agent handler functions (dict mapping agent name -> callable)
so you can wire real agent execute functions later.
"""

from typing import Callable, Dict, Any, Optional
from src.core.state_schema import AssistantState, make_initial_state

# Try to import langgraph; if not installed, use fallback
try:
    from langgraph import StateGraph  # type: ignore
    _HAS_LANGGRAPH = True
except Exception:
    _HAS_LANGGRAPH = False

# Import your LLM wrapper (Gemini) — expects a classify() method
try:
    from src.utils.gemini_client import GeminiClient
except Exception:
    # best-effort fallback: a tiny internal classifier if your llm client is not present
    class _FallbackGemini:
        def classify(self, text: str) -> dict:
            t = text.lower()
            if any(k in t for k in ("email", "inbox", "mail")):
                return {"agent": "email", "confidence": 0.95, "reason": "keyword_fallback"}
            if any(k in t for k in ("calendar", "meeting", "schedule")):
                return {"agent": "calendar", "confidence": 0.95, "reason": "keyword_fallback"}
            if any(k in t for k in ("task", "todo", "remind", "add task")):
                return {"agent": "task", "confidence": 0.95, "reason": "keyword_fallback"}
            return {"agent": "coordinator", "confidence": 0.6, "reason": "fallback"}
    GeminiClient = _FallbackGemini  # type: ignore

# --- Fallback SimpleStateGraph if langgraph not installed ---
class SimpleStateGraph:
    """
    Very small state graph runner:
      - nodes is an ordered list of (name, function)
      - each node is called with the state object and may mutate and return state
      - this is intentionally sequential and simple for quick testing
    """
    def __init__(self):
        self.nodes = []

    def add_node(self, name: str, fn: Callable[[AssistantState], AssistantState]):
        self.nodes.append((name, fn))

    def run(self, state: AssistantState) -> AssistantState:
        for name, fn in self.nodes:
            state = fn(state) or state
        return state

# --- Node implementations ---
def classify_request_node_factory(llm_client: Any):
    """
    Returns a node function that classifies user_input using llm_client.classify(...)
    and updates the state with routing fields.
    """
    def node(state: AssistantState) -> AssistantState:
        text = state.get("user_input", "") or ""
        try:
            res = llm_client.classify(text)
            agent = res.get("agent") or res.get("agent_selection") or "coordinator"
            confidence = float(res.get("confidence", 0.0))
            reason = res.get("reason", "") or res.get("reasoning", "")
        except Exception as e:
            agent, confidence, reason = "coordinator", 0.0, f"classify_error: {e}"

        state["task_type"] = agent
        state["current_agent"] = agent
        state["confidence_score"] = confidence
        state["route"] = agent
        state["route_confidence"] = confidence
        state["route_reason"] = reason
        # append to conversation history for traceability
        hist = state.get("conversation_history", [])
        hist.append({"system": f"classified route -> {agent} (conf={confidence:.2f})", "raw_reason": reason})
        state["conversation_history"] = hist
        return state
    return node

def agent_router_node_factory(agent_handlers: Optional[Dict[str, Callable[[AssistantState], AssistantState]]] = None):
    """
    Router node: validates chosen agent and (optionally) dispatches the request
    to the registered handler, if available. If handler exists, it will be called
    and its result placed into state['agent_result'].
    """
    handlers = agent_handlers or {}
    def node(state: AssistantState) -> AssistantState:
        agent = state.get("task_type", "") or state.get("current_agent", "")
        if not agent:
            state["final_response"] = "No agent selected."
            return state

        # If a handler is registered for this agent, call it and capture the result
        handler = handlers.get(agent)
        if handler:
            try:
                result = handler(state)
                # handler may return a dict or AssistantState-like updates
                state["agent_result"] = result
                # optionally, if result contains 'response', set final_response
                if isinstance(result, dict) and "response" in result:
                    state["final_response"] = result["response"]
            except Exception as e:
                state["error_log"].append({"node": "agent_router", "error": str(e)})
                state["final_response"] = f"Agent {agent} failed: {e}"
        else:
            # no handler — fill with a placeholder response
            state["final_response"] = f"(placeholder) routed to {agent}. Enable agent implementation to perform actions."
        # audit
        hist = state.get("conversation_history", [])
        hist.append({"system": f"routed to {agent}", "agent": agent})
        state["conversation_history"] = hist
        return state
    return node

def verification_node(state: AssistantState) -> AssistantState:
    """
    Simple verification stub. For now set verification_scores with route confidence.
    """
    conf = float(state.get("confidence_score", 0.0))
    state["verification_scores"] = {"route_confidence": conf, "verified": conf >= 0.7}
    hist = state.get("conversation_history", [])
    hist.append({"system": f"verification: verified={state['verification_scores']['verified']}"})
    state["conversation_history"] = hist
    return state

def response_generator_node(state: AssistantState) -> AssistantState:
    """
    Final response generator. If an agent_result with 'response' exists, return it;
    otherwise craft a default message.
    """
    if "final_response" in state and state["final_response"]:
        return state
    # Try agent_result
    res = state.get("agent_result")
    if isinstance(res, dict) and "response" in res:
        state["final_response"] = res["response"]
        return state
    # Fallback default
    agent = state.get("task_type", "coordinator")
    state["final_response"] = f"No agent action taken. Request routed to `{agent}`."
    return state

# --- AssistantGraph wrapper ---
class AssistantGraph:
    """
    Builds and runs the small workflow:
      classify_request_node -> verification_node -> agent_router_node -> response_generator_node

    Optional agent_handlers: dict mapping agent_name -> callable(state) -> dict
    """
    def __init__(self, llm_client: Optional[Any] = None, agent_handlers: Optional[Dict[str, Callable[[AssistantState], Any]]] = None):
        self.llm = llm_client or GeminiClient()
        self.agent_handlers = agent_handlers or {}
        # Use langgraph.StateGraph if present
        if _HAS_LANGGRAPH:
            # Minimal wiring for langgraph: create nodes as callables
            self.graph = StateGraph()
            self.graph.add_node("classify", classify_request_node_factory(self.llm))
            self.graph.add_node("verify", verification_node)
            self.graph.add_node("route", agent_router_node_factory(self.agent_handlers))
            self.graph.add_node("respond", response_generator_node)
            # Define simple linear transitions (StateGraph may support richer)
            self.order = ["classify", "verify", "route", "respond"]
            self._use_langgraph = True
        else:
            self.graph = SimpleStateGraph()
            self.graph.add_node("classify", classify_request_node_factory(self.llm))
            self.graph.add_node("verify", verification_node)
            self.graph.add_node("route", agent_router_node_factory(self.agent_handlers))
            self.graph.add_node("respond", response_generator_node)
            self.order = ["classify", "verify", "route", "respond"]
            self._use_langgraph = False

    def run(self, user_input: str, initial_state: Optional[AssistantState] = None) -> AssistantState:
        """
        Run the graph synchronously. Returns the final AssistantState dict.
        """
        state = initial_state or make_initial_state(user_input)
        # ensure fields exist
        if "error_log" not in state:
            state["error_log"] = []
        # Execute nodes in order
        final = self.graph.run(state)
        return final

    # convenience: run and return core results
    def invoke(self, user_input: str) -> Dict[str, Any]:
        final_state = self.run(user_input)
        return {
            "route": final_state.get("route"),
            "route_confidence": final_state.get("route_confidence"),
            "final_response": final_state.get("final_response"),
            "state": final_state
        }

# Node-level unit testing helper (optional)
def _example_agent_handler_task(state: AssistantState) -> Dict[str, Any]:
    """
    Example synchronous handler for 'task' requests (used in tests/demo).
    This writes a minimal 'response' field — in real code you'd call MCP to persist the task.
    """
    text = state.get("user_input", "")
    # Extract naive task text
    if ":" in text:
        task_text = text.split(":", 1)[1].strip()
    else:
        task_text = text
    return {"response": f"Task handler saved task: {task_text}"}
