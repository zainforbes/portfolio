import time
from src.utils.performance_monitor import Perf

def test_perf_timed():
    bucket = "test_bucket"
    with Perf.timed(bucket):
        time.sleep(0.01)

    assert bucket in Perf.hist
    assert len(Perf.hist[bucket]) >= 1
    assert Perf.hist[bucket][0] >= 10  # at least 10ms

def test_perf_snapshot():
    # Ensure counters work
    Perf.counters["test_counter"] = 5
    snap = Perf.snapshot()
    assert snap["counters"]["test_counter"] == 5
    assert "web_search_avg_ms" in snap
