"""
ratelimit.py

Request limits for the public demo (D-020). The demo answers with free-tier
API keys, so an unlimited public page lets one visitor (or a script) use
up the day's quota for everyone. Two limits, both in memory:

  - per client: at most PER_CLIENT_PER_MINUTE questions in any 60 seconds
  - global:     at most DAILY_LIMIT questions per UTC day across all clients

Both are set by environment variable; 0 turns a limit off:

  PITCHWISE_RATE_PER_MINUTE   default 6
  PITCHWISE_DAILY_LIMIT       default 500

In-memory state resets on restart and is per process, which is enough for
one free Hugging Face Space container. Several replicas would need a
shared store such as Redis.
"""

import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

WINDOW_SECONDS = 60


def _env_int(name, default):
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


class RateLimiter:
    def __init__(self, per_client_per_minute=None, daily_limit=None, clock=time.monotonic, today=None):
        self.per_client = (_env_int("PITCHWISE_RATE_PER_MINUTE", 6)
                           if per_client_per_minute is None else per_client_per_minute)
        self.daily_limit = _env_int("PITCHWISE_DAILY_LIMIT", 500) if daily_limit is None else daily_limit
        self._clock = clock
        self._today = today or (lambda: datetime.now(timezone.utc).date())
        self._lock = threading.Lock()
        self._hits = defaultdict(deque)
        self._day = None
        self._day_count = 0

    def check(self, client_id):
        """Record one request. Returns None if allowed, or a message saying
        why it was refused."""
        now = self._clock()
        with self._lock:
            day = self._today()
            if day != self._day:
                self._day, self._day_count = day, 0
                self._hits.clear()
            if self.daily_limit and self._day_count >= self.daily_limit:
                return ("The demo has reached its daily question limit, which protects its free-tier "
                        "API quota. Please try again tomorrow, or run Pitchwise locally.")
            if self.per_client:
                hits = self._hits[client_id]
                while hits and now - hits[0] >= WINDOW_SECONDS:
                    hits.popleft()
                if len(hits) >= self.per_client:
                    wait = int(WINDOW_SECONDS - (now - hits[0])) + 1
                    return f"Too many questions in a short time. Please wait about {wait} seconds."
                hits.append(now)
            self._day_count += 1
            return None


def client_id_from(headers, fallback_host):
    """The visitor's address. Behind Hugging Face's proxy the connecting
    host is the proxy, so the first X-Forwarded-For entry is used when
    present."""
    forwarded = headers.get("x-forwarded-for") if headers else None
    if forwarded:
        return forwarded.split(",")[0].strip()
    return fallback_host or "unknown"


limiter = RateLimiter()
