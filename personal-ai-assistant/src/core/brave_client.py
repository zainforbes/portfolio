import os
import requests
from dotenv import load_dotenv

load_dotenv()

class BraveSearchClient:
    def __init__(self):
        self.api_key = os.getenv("BRAVE_API_KEY")
        if not self.api_key:
            raise ValueError("Missing BRAVE_API_KEY in .env")
        self.url = "https://api.search.brave.com/res/v1/web/search"

    def search(self, query: str, count: int = 5):
        headers = {"X-Subscription-Token": self.api_key}
        params = {"q": query, "count": count}
        resp = requests.get(self.url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()

        # Extract results safely
        results = data.get("web", {}).get("results", [])
        return [{"title": r["title"], "url": r["url"]} for r in results[:count]]
