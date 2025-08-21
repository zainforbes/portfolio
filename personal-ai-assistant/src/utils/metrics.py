import time

def time_call(state, name: str, fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    ms = round((time.perf_counter() - t0) * 1000, 1)
    state.context.setdefault("metrics", {})[name] = ms
    return out

def mark(state, name: str, value):
    state.context.setdefault("metrics", {})[name] = value