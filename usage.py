"""
usage.py

Token and cost accounting for every LLM call (DECISIONS.md D-012).

Each provider call reports its input and output tokens here. A *meter*
collects the calls made while answering one question, and `summarise()`
turns them into the totals that go into the answer's trace: tokens, cost at
list price, and one record per call (step, model, tokens, latency, and
whether the call failed and the fallback chain moved on).

The meter lives in a ContextVar, so code deep inside the LangGraph nodes
records into the meter of the request that called it, without passing a
meter object through every function. Concurrent requests (FastAPI runs each
in its own thread) each see their own meter.

Prices are list prices in US dollars per million tokens, from each
provider's published pricing on the date in PRICES_CHECKED. Pitchwise runs
on free tiers, so the money actually spent is zero; the list price answers
"what would this cost on a paid plan?".
"""

import contextvars
import time
from contextlib import contextmanager

PRICES_CHECKED = "2026-10-05"

# model id -> (USD per 1M input tokens, USD per 1M output tokens)
PRICES = {
    # Groq, console.groq.com/docs/models
    "qwen/qwen3.8-27b": (0.80, 4.00),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
    # Google, ai.google.dev/gemini-api/docs/pricing (paid tier, text input)
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.6-flash": (0.75, 3.75),  # rises to 1.50 / 7.50 on 2027-01-01
    "gemini-3.1-flash-lite": (0.25, 1.50),
    # OpenRouter ":free" variants cost nothing
    "minimax/minimax-m3:free": (0.0, 0.0),
}

_CALLS = contextvars.ContextVar("pitchwise_llm_calls", default=None)
_STEP = contextvars.ContextVar("pitchwise_llm_step", default="generate")


def cost_usd(model, input_tokens, output_tokens):
    """List-price cost of one call. None when the model has no known price,
    so an unpriced model shows up as unknown instead of silently as $0."""
    if model not in PRICES:
        return None
    price_in, price_out = PRICES[model]
    return ((input_tokens or 0) * price_in + (output_tokens or 0) * price_out) / 1_000_000


@contextmanager
def meter():
    """Collect every LLM call made inside this block. Re-entrant: if a meter
    is already active (agentic engine called from answer_question), the
    outer one keeps collecting and this block yields the same list."""
    existing = _CALLS.get()
    if existing is not None:
        yield existing
        return
    calls = []
    token = _CALLS.set(calls)
    try:
        yield calls
    finally:
        _CALLS.reset(token)


@contextmanager
def step(name):
    """Label the LLM calls made inside this block (plan, grade, rewrite,
    generate, judge)."""
    token = _STEP.set(name)
    try:
        yield
    finally:
        _STEP.reset(token)


def current_step():
    return _STEP.get()


def record(provider, model, input_tokens, output_tokens, latency_ms, ok=True, error=None):
    """Add one call to the active meter (no-op when no meter is active)."""
    calls = _CALLS.get()
    if calls is None:
        return
    calls.append({
        "step": _STEP.get(),
        "provider": provider,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd(model, input_tokens, output_tokens) if ok else 0.0,
        "latency_ms": latency_ms,
        "ok": ok,
        "error": (str(error)[:200] if error else None),
    })


class Timer:
    """Milliseconds since construction."""

    def __init__(self):
        self.start = time.perf_counter()

    def ms(self):
        return round((time.perf_counter() - self.start) * 1000)


def summarise(calls):
    """Totals for one question. Failed attempts (a provider error before the
    fallback chain moved on) are counted separately and cost nothing."""
    ok = [c for c in calls if c["ok"]]
    input_tokens = sum(c["input_tokens"] or 0 for c in ok)
    output_tokens = sum(c["output_tokens"] or 0 for c in ok)
    costs = [c["cost_usd"] for c in ok]
    by_step = {}
    for c in ok:
        s = by_step.setdefault(c["step"], {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        s["calls"] += 1
        s["input_tokens"] += c["input_tokens"] or 0
        s["output_tokens"] += c["output_tokens"] or 0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": None if any(x is None for x in costs) else round(sum(costs), 8),
        "tokens_reported": all(c["input_tokens"] is not None for c in ok),
        "failed_attempts": len(calls) - len(ok),
        "by_step": by_step,
        "calls": calls,
    }
