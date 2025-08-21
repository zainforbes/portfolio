# tests/test_search.py
import asyncio
import os
from src.mcp_integration.mcp_client import MCPClient
from src.mcp_integration.search_tools import SearchTools
# tests/test_search.py
from dotenv import load_dotenv
import os

# Ensure .env is loaded (project root)
load_dotenv()

async def test_search():
    # You need a Brave Search API key
    api_key = os.getenv('BRAVE_API_KEY')
    if not api_key:
        print("BRAVE_API_KEY environment variable required")
        return
    
    client = MCPClient()
    search_tools = SearchTools(client, api_key)
    
    # Start search server
    success = await search_tools.start()
    assert success, "Failed to start search server"
    
    # Test search
    results = await search_tools.web_search("Python MCP tutorial", count=5)
    assert len(results) > 0, "No search results returned"
    
    print(f"Found {len(results)} search results:")
    for i, result in enumerate(results[:3]):
        print(f"{i+1}. {result.get('title', 'No title')}")
        print(f"   {result.get('url', 'No URL')}")
    
    print("Search tests passed!")
    
    # Cleanup
    await client.shutdown()

if __name__ == "__main__":
    asyncio.run(test_search())