from functools import wraps
import time


def circuit(max_failures: int = 3, reset_timeout: int = 60):
    def decorator(fn):
        failures = 0
        opened_at = None

        @wraps(fn)
        def wrapper(*args, **kwargs):
            nonlocal failures, opened_at
            if opened_at and time.time() - opened_at < reset_timeout:
                raise RuntimeError("Circuit OPEN")
            try:
                result = fn(*args, **kwargs)
                failures = 0  # success resets
                opened_at = None
                return result
            except Exception as e:
                failures += 1
                if failures >= max_failures:
                    opened_at = time.time()
                raise e

        return wrapper

    return decorator
