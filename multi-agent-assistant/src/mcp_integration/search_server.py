import os
import httpx
from typing import List, Dict

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

class NotConfigured(Exception):
    pass

async def web_search(query: str, count: int = 5) -> List[Dict]:
    """Real Brave Search. Returns simplified results."""
    if not BRAVE_API_KEY:
        raise NotConfigured("BRAVE_API_KEY not set in environment/.env")

    params = {"q": query, "count": min(max(count, 1), 20)}
    headers = {"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY}

    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(BRAVE_ENDPOINT, params=params, headers=headers)
        r.raise_for_status()
        data = r.json()

    results = []
    for item in (data.get("web", {}).get("results", []) or []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("description", ""),
            "source": item.get("meta_url", {}).get("hostname", "")
        })
        if len(results) >= count:
            break
    return results
