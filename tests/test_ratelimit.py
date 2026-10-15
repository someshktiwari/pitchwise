"""Offline tests for the public-demo request limits (ratelimit.py, D-020)."""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ratelimit


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_per_client_limit_and_window():
    clock = Clock()
    rl = ratelimit.RateLimiter(per_client_per_minute=2, daily_limit=0, clock=clock)
    assert rl.check("a") is None
    assert rl.check("a") is None
    assert "wait" in rl.check("a")
    assert rl.check("b") is None          # other clients are unaffected
    clock.t = 61
    assert rl.check("a") is None          # the window has moved on


def test_refused_requests_do_not_count_toward_the_daily_limit():
    clock = Clock()
    rl = ratelimit.RateLimiter(per_client_per_minute=1, daily_limit=2, clock=clock)
    assert rl.check("a") is None
    assert rl.check("a") is not None      # refused per client
    assert rl.check("b") is None          # still allowed: only 2 accepted so far
    assert "daily" in rl.check("c")


def test_daily_limit_resets_on_a_new_day():
    day = [date(2026, 10, 15)]
    rl = ratelimit.RateLimiter(per_client_per_minute=0, daily_limit=1, today=lambda: day[0])
    assert rl.check("a") is None
    assert rl.check("b") is not None
    day[0] = date(2026, 10, 16)
    assert rl.check("b") is None


def test_zero_disables_both_limits():
    rl = ratelimit.RateLimiter(per_client_per_minute=0, daily_limit=0)
    assert all(rl.check("a") is None for _ in range(100))


def test_client_id_prefers_forwarded_for():
    assert ratelimit.client_id_from({"x-forwarded-for": "1.2.3.4, 10.0.0.1"}, "10.0.0.1") == "1.2.3.4"
    assert ratelimit.client_id_from({}, "5.6.7.8") == "5.6.7.8"
    assert ratelimit.client_id_from(None, None) == "unknown"
