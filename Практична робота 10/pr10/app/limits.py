"""Ковзне вікно обмеження частоти для кожного клієнта."""

import os
import threading
import time
from collections import defaultdict, deque

from dotenv import load_dotenv

load_dotenv()
PER_MINUTE = max(1, int(os.getenv("RATE_LIMIT_PER_MINUTE", "10")))
WINDOW_SECONDS = 60.0
_requests: dict[str, deque[float]] = defaultdict(deque)
_lock = threading.Lock()


class RateLimited(Exception):
    """Ліміт вичерпано; retry_after указує час до наступного дозволу."""

    def __init__(self, retry_after: float) -> None:
        super().__init__(f"ліміт вичерпано, повторіть через {retry_after:.1f} с")
        self.retry_after = max(0.1, retry_after)


def settings() -> dict:
    return {"per_minute": PER_MINUTE, "window_seconds": WINDOW_SECONDS}


def acquire(client: str) -> None:
    """Дозволити запит або відмовити з чесним часом очікування."""
    key = (str(client or "anonymous").strip() or "anonymous")[:128]
    now = time.monotonic()
    cutoff = now - WINDOW_SECONDS
    with _lock:
        history = _requests[key]
        while history and history[0] <= cutoff:
            history.popleft()
        if len(history) >= PER_MINUTE:
            retry_after = WINDOW_SECONDS - (now - history[0])
            raise RateLimited(retry_after)
        history.append(now)
        # Прибираємо порожні старі ключі, щоб словник не зростав без меж.
        if len(_requests) > 1000:
            for old_key in list(_requests):
                if not _requests[old_key]:
                    _requests.pop(old_key, None)
