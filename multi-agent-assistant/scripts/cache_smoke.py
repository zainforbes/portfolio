import asyncio
from dotenv import load_dotenv; load_dotenv()
from src.mcp_integration.search_server import web_search, web_search_stats

async def main():
    # first call = cache miss
    r1 = await web_search("gemini api docs", count=3)
    # second call = cache hit
    r2 = await web_search("gemini api docs", count=3)
    stats = web_search_stats()
    print("items1:", len(r1), "items2:", len(r2))
    print("counters:", stats["counters"])
    print("avg_ms:", stats["web_search_avg_ms"], "samples:", stats["web_search_samples"])

if __name__ == "__main__":
    asyncio.run(main())
