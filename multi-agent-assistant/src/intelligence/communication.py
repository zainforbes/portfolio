from typing import Dict, Any, Optional

class AgentCommunicationHub:
    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents

    async def ask(self, from_agent: str, to_agent: str, user_input: str, extra_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Let one agent call another. Returns the callee's last payload."""
        agent = self.agents.get(to_agent)
        if not agent:
            return {"error": f"agent '{to_agent}' not found"}
        state = {"user_input": user_input, "agent_messages": [], "current_agent": to_agent}
        if extra_state:
            state.update(extra_state)
        out = await agent.execute(state)
        msgs = out.get("agent_messages", [])
        return msgs[-1]["payload"] if msgs else {}
