# Implement a 90S TTL for the LLM 
import hashlib, time
_CACHE = {}

def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def get(text: str, ttl: int = 90):
    k = _key(text)
    v = _CACHE.get(k)
    if not v: return None
    val, exp = v
    return val if exp > time.time() else None

def set_(text: str, value: str, ttl: int = 90):
    k = _key(text)
    _CACHE[k] = (value, time.time() + ttl)
