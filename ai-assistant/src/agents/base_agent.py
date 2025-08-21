# src/agents/base_agent.py
from typing import Dict, Any

class BaseAgent:
    def __init__(self, name: str):
        self.name = name

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
