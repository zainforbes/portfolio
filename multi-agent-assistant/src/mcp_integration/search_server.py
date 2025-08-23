import os
import httpx
from typing import List, Dict

class NotConfigured(Exception):
    pass

def _get_api_key() -> str:
    key = os.getenv("BRAVE_API_KEY")
    if not key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            key = os.getenv("BRAVE_API_KEY")
        except Exception:
            pass
    return key or ""

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

async def web_search(query: str, count: int = 5) -> List[Dict]:
    api_key = _get_api_key()
    if not api_key:
        raise NotConfigured("BRAVE_API_KEY not set in environment/.env")

    params = {"q": query, "count": min(max(count, 1), 20)}
    headers = {"Accept": "application/json", "X-Subscription-Token": api_key}

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
