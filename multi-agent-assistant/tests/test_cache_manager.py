import time
from src.utils.cache_manager import TTLCache

def test_ttl_cache_basic():
    cache = TTLCache(ttl=1, maxsize=2)
    cache.set("a", 1)
    assert cache.get("a") == 1
    assert cache.get("b") is None

def test_ttl_cache_expiry():
    cache = TTLCache(ttl=0.1, maxsize=2)
    cache.set("a", 1)
    time.sleep(0.2)
    assert cache.get("a") is None

def test_ttl_cache_eviction():
    cache = TTLCache(ttl=10, maxsize=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    # 'a' should be evicted
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3
