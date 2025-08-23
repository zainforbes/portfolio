import time
from typing import Dict, List

class Perf:
    counters: Dict[str, int] = {
        "web_search_calls": 0,
        "web_search_cache_hits": 0,
    }
    hist: Dict[str, List[float]] = {
        "web_search_ms": [],
    }

    @classmethod
    def timed(cls, bucket: str):
        class _T:
            def __enter__(self_inner):
                self_inner.t0 = time.perf_counter()
                return self_inner
            def __exit__(self_inner, *exc):
                dt_ms = (time.perf_counter() - self_inner.t0) * 1000
                cls.hist.setdefault(bucket, []).append(dt_ms)
                if len(cls.hist[bucket]) > 300:
                    cls.hist[bucket] = cls.hist[bucket][-300:]
        return _T()

    @classmethod
    def snapshot(cls) -> Dict[str, object]:
        # small, UI-friendly snapshot
        h = cls.hist.get("web_search_ms", [])
        avg = round(sum(h) / len(h), 1) if h else 0.0
        return {
            "counters": dict(cls.counters),
            "web_search_avg_ms": avg,
            "web_search_samples": len(h),
        }
