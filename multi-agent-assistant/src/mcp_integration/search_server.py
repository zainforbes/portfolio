# src/mcp_integration/search_server.py
from __future__ import annotations

import os
import asyncio
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

import httpx

from src.utils.cache_manager import TTLCache
from src.utils.performance_monitor import Perf


BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"
_CACHE = TTLCache(ttl=120, maxsize=200)  # 2 min cache for identical queries


class NotConfigured(RuntimeError):
    """Raised when required configuration (e.g., API keys) is missing."""


def _get_api_key() -> str:
    key = os.getenv("BRAVE_API_KEY") or os.getenv("X_SUBSCRIPTION_TOKEN")
    if not key:
        raise NotConfigured("BRAVE_API_KEY not set in environment/.env")
    return key


def _hostname(u: str) -> str:
    try:
        return urlparse(u).hostname or ""
    except Exception:
        return ""


def _normalize_results(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Brave response shape (simplified):

    {
      "web": {
        "results": [
          {
            "title": "...",
            "url": "https://...",
            "description": "...",
            "extra_snippets": ["...", "..."],
            "meta_url": {"hostname": "example.com", ...}
          }, ...
        ]
      }
    }
    """
    out: List[Dict[str, str]] = []
    web = (data or {}).get("web") or {}
    results = web.get("results") or []
    for r in results:
        title = r.get("title") or ""
        url = r.get("url") or ""
        desc = r.get("description") or ""
        extras = r.get("extra_snippets") or []
        snippet = desc if desc else (" ".join(extras) if extras else "")
        source = (r.get("meta_url") or {}).get("hostname") or _hostname(url) or ""
        out.append({
            "title": title,
            "url": url,
            "snippet": snippet,
            "source": source,
        })
    return out


async def _web_search_impl(query: str, count: int = 5) -> List[Dict[str, str]]:
    """
    Low-level Brave call. Raises httpx.HTTPStatusError for non-2xx,
    which is handled/normalized by the coordinator via error_handler.
    """
    api_key = _get_api_key()
    count = max(1, min(int(count or 5), 10))  # clamp 1..10 (Brave supports up to 20; keep it modest)

    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key,
        # "Accept-Encoding": "gzip",  # httpx enables this automatically
    }
    params = {
        "q": query,
        "count": count,
        # Add optional parameters here if desired:
        # "country": "ZA",          # focus results to South Africa (optional)
        # "safesearch": "moderate", # "off" | "moderate" | "strict"
        # "freshness": "pd",        # "pd" past day, "pw" past week, etc.
    }

    timeout = httpx.Timeout(10.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        resp = await client.get(BRAVE_API_URL, params=params)
        # Ensure HTTP errors raise with context (status code, URL)
        resp.raise_for_status()
        data = resp.json()

    return _normalize_results(data)


async def web_search(query: str, count: int = 5) -> List[Dict[str, str]]:
    """
    Public tool function: adds TTL caching and timing metrics around the Brave call.
    """
    Perf.counters["web_search_calls"] = Perf.counters.get("web_search_calls", 0) + 1

    key = f"{query}|{count}"
    cached = _CACHE.get(key)
    if cached is not None:
        Perf.counters["web_search_cache_hits"] = Perf.counters.get("web_search_cache_hits", 0) + 1
        return cached

    with Perf.timed("web_search_ms"):
        results = await _web_search_impl(query, count)

    _CACHE.set(key, results)
    return results


def web_search_stats() -> Dict[str, Any]:
    """Small, UI-friendly snapshot for dashboards/tests."""
    return Perf.snapshot()
