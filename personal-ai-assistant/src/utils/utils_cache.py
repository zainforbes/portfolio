import time
_CACHE = {}

def cache_get(key, ttl=60):
    v = _CACHE.get(key)
    if not v: return None
    val, exp = v
    return val if exp > time.time() else None

def cache_set(key, val, ttl=60):
    _CACHE[key] = (val, time.time() + ttl)
