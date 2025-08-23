from typing import Dict, Any
from .base_agent import BaseAgent

class CoordinatorAgent(BaseAgent):
    name = "coordinator"

    def __init__(self, agents: Dict[str, BaseAgent]):
        super().__init__()
        self.agents = agents

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        target = state.get("current_agent") or "default"
        agent = self.agents.get(target) or self.agents.get("default")
        if not agent:
            self.add_msg(state, "error", {"error": f"No agent found for '{target}'"})
            return state
        return await agent.execute(state)
