import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class SearchTools:
    """Search tools integration using MCP client and Brave Search API."""
    
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client
        # Set the API key from environment variable
        self.api_key = os.getenv("BRAVE_API_KEY")
        # Alternative attribute names in case your code uses different naming
        self.brave_api_key = self.api_key
        
        if not self.api_key:
            logger.error("BRAVE_API_KEY not found in environment variables")
        else:
            logger.info(f"Brave API key loaded (length: {len(self.api_key)})")
    
    async def start(self):
        """Start the search tools service."""
        if not self.api_key:
            raise ValueError("Brave API key required. Check your .env file contains BRAVE_API_KEY=your_key")
        
        # Initialize any additional setup here
        logger.info("SearchTools started successfully")
        return True
    
    async def web_search(self, query: str, count: int = 10) -> List[Dict[str, Any]]:
        """
        Perform web search using Brave Search API.
        
        Args:
            query: Search query string
            count: Number of results to return (default: 10)
            
        Returns:
            List of search results with title, url, description
        """
        if not self.api_key:
            raise ValueError("Brave API key not available")
        
        try:
            import httpx
            
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self.api_key
            }
            params = {
                "q": query,
                "count": count
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                
                data = response.json()
                results = []
                
                # Parse Brave Search API response
                web_results = data.get("web", {}).get("results", [])
                for result in web_results:
                    results.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "description": result.get("description", ""),
                        "age": result.get("age", ""),
                        "language": result.get("language", ""),
                        "family_friendly": result.get("family_friendly", True)
                    })
                
                logger.info(f"Web search completed: {len(results)} results for '{query}'")
                return results
                
        except ImportError:
            raise ImportError("httpx library required for web search. Install with: pip install httpx")
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the search tools."""
        logger.info("SearchTools shutdown completed")
        pass