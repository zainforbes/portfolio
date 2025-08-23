from typing import Dict, Any
from .base_agent import BaseAgent
from src.intelligence import error_handler as EH

class CoordinatorAgent(BaseAgent):
    name = "coordinator"

    def __init__(self, agents: Dict[str, Any]):
        super().__init__()
        self.agents = agents

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        agent_name = state.get("current_agent") or "default"
        agent = self.agents.get(agent_name, self.agents["default"])
        try:
            return await agent.execute(state)
        except Exception as e:
            # graceful error payload
            self.add_msg(state, "error", {
                "agent": agent_name,
                "message": EH.explain(e),
                "raw": str(e)[:500],
            })
            return state
