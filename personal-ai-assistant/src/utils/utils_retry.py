import time
from typing import Callable, Tuple, Type

def retry(fn: Callable, retries: int = 2, backoff: float = 0.6,
          exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    def wrapped(*args, **kwargs):
        attempt, delay = 0, backoff
        while True:
            try:
                return fn(*args, **kwargs)
            except exceptions as e:
                attempt += 1
                if attempt > retries:
                    raise
                time.sleep(delay)
                delay *= 2
    return wrapped
