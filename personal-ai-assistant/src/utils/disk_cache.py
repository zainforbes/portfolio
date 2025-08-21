import json, os, time, hashlib
PATH = os.getenv("LLM_CACHE_PATH", "config/llm_cache.json")
TTL  = int(os.getenv("LLM_CACHE_TTL", "90"))

def _key(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _load() -> dict:
    try:
        with open(PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save(data: dict):
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(data, f)

def get(prompt: str):
    data = _load()
    v = data.get(_key(prompt))
    if not v:
        return None
    val, exp = v
    return val if exp > time.time() else None

def set(prompt: str, val: str, ttl: int = TTL):
    data = _load()
    data[_key(prompt)] = (val, time.time() + ttl)
    _save(data)