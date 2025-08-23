import asyncio
from dotenv import load_dotenv

# load .env so BRAVE_API_KEY is visible
load_dotenv()

from src.mcp_integration.search_server import web_search

async def main():
    results = await web_search("site:google.dev gemini api", count=2)
    for r in results:
        print(f"- {r['title']} ({r['url']})")

if __name__ == "__main__":
    asyncio.run(main())
