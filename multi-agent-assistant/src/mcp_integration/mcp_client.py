import asyncio
from typing import Any, Callable, Dict

class MCPClient:
    """
    Minimal async tool registry:
      - register_tool(name: str, async func)
      - await call_tool(name, **kwargs)
    """
    def __init__(self):
        self.tools: Dict[str, Callable[..., Any]] = {}

    def register_tool(self, name: str, func: Callable[..., Any]) -> None:
        if not asyncio.iscoroutinefunction(func):
            raise TypeError(f"Tool '{name}' must be async")
        self.tools[name] = func

    async def call_tool(self, name: str, **kwargs) -> Any:
        tool = self.tools.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")
        return await tool(**kwargs)
