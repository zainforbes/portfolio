from typing import Dict, Any
from .base_agent import BaseAgent

class DefaultAgent(BaseAgent):
    name = "default"
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.add_msg(state, "response", {"result": "DefaultAgent handled generic request"})
        return state
