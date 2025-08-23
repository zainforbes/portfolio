import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import asyncio
from src.mcp_integration.calendar_server import list_events

async def main():
    events = await list_events(max_results=3)
    print(f"Fetched {len(events)} events")
    for e in events:
        print("-", e["summary"], "|", e["start"], "→", e["end"])

if __name__ == "__main__":
    asyncio.run(main())
