import time
from typing import Any, Optional

class TTLCache:
    """In-memory TTL cache (FIFO eviction)."""
    def __init__(self, ttl: int = 120, maxsize: int = 200):
        self.ttl = ttl
        self.maxsize = maxsize
        self._store: dict[str, tuple[Any, float]] = {}

    def get(self, key: str) -> Optional[Any]:
        item = self._store.get(key)
        if not item:
            return None
        val, expires_at = item
        if expires_at < time.time():
            self._store.pop(key, None)
            return None
        return val

    def set(self, key: str, val: Any) -> None:
        if len(self._store) >= self.maxsize:
            # naive eviction: pop oldest inserted
            self._store.pop(next(iter(self._store)))
        self._store[key] = (val, time.time() + self.ttl)

    def clear(self) -> None:
        self._store.clear()
