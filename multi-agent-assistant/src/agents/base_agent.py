from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from src.utils.gemini_client import GeminiClient
from src.mcp_integration.mcp_client import MCPClient

class BaseAgent(ABC):
    name: str = "base"

    def __init__(self, gemini: Optional[GeminiClient] = None, mcp: Optional[MCPClient] = None, comm: Optional[object] = None):
        self.gemini = gemini
        self.mcp = mcp
        self.comm = comm

    def add_msg(self, state: Dict[str, Any], message_type: str, payload: Dict[str, Any], notes: str = "") -> None:
        state.setdefault("agent_messages", []).append({
            "sender": self.name,
            "message_type": message_type,
            "payload": payload,
            "notes": notes
        })

    @abstractmethod
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        ...
