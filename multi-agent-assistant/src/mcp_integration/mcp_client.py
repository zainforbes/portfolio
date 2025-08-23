import asyncio
from src.intelligence import error_handler as EH

class MCPClient:
    def __init__(self):
        self.tools = {}

    def register_tool(self, name, coro):
        self.tools[name] = coro

    async def call_tool(self, name, **kwargs):
        tool = self.tools.get(name)
        if not tool:
            raise RuntimeError(f"Unknown tool: {name}")

        last_exc = None
        for attempt in range(4):  # 0..3
            try:
                return await tool(**kwargs)
            except Exception as e:
                last_exc = e
                retry, delay = EH.should_retry(e, attempt)
                if retry:
                    await asyncio.sleep(delay)
                    continue
                raise  # non-retryable -> let coordinator format it

        raise last_exc  # defensive
