"""
Offline tests for token and cost accounting (usage.py, D-012) and for the
optional tracing layer staying out of the way when it is off (D-013).

The provider boundary (answer._call_provider) is replaced with a fake that
returns scripted text and token counts, so the real call_model,
generate_with_fallback, linear engine and LangGraph engine all run.

Run from the project root:  uv run pytest tests/ -q
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from langchain_core.documents import Document

import answer as base
import graph as g
import observability as obs
import usage


class FakeStore:
    def as_retriever(self, search_kwargs):
        class Retriever:
            def invoke(self, query):
                return [Document(page_content=f"chunk for {query}",
                                 metadata={"source": "kb.md", "header_1": query, "header_2": ""})]
        return Retriever()


def fake_provider(replies, fail_models=()):
    """A stand-in for answer._call_provider: replies are picked by which
    prompt is being sent; models in fail_models raise like a rate limit."""
    def call(provider, model, system_prompt, question, history=None, temperature=None):
        if model in fail_models:
            raise RuntimeError("429 rate limit")
        if system_prompt == g.PLAN_PROMPT:
            return replies["plan"], 100, 20
        if system_prompt == g.GRADE_PROMPT:
            return replies["grade"], 300, 10
        return replies.get("answer", "final answer"), 500, 50
    return call


DEFAULT = "Groq: qwen3.8-27b (fastest, default)"
QWEN = base.MODEL_OPTIONS[DEFAULT][1]


def test_cost_uses_list_prices_and_flags_unknown_models():
    # qwen3.8-27b: $0.80 / 1M input, $4.00 / 1M output
    assert usage.cost_usd(QWEN, 1_000_000, 0) == pytest.approx(0.80)
    assert usage.cost_usd(QWEN, 0, 1_000_000) == pytest.approx(4.00)
    assert usage.cost_usd("unknown-model", 10, 10) is None
    # every model the app can call has a price
    assert all(model in usage.PRICES for _, model in base.MODEL_OPTIONS.values())


def test_linear_engine_reports_tokens_and_cost(monkeypatch):
    monkeypatch.setattr(base, "_call_provider", fake_provider({}))
    _, _, trace = base.answer_question(FakeStore(), "q")
    u = trace["usage"]
    assert (u["input_tokens"], u["output_tokens"], u["total_tokens"]) == (500, 50, 550)
    assert u["cost_usd"] == pytest.approx((500 * 0.80 + 50 * 4.00) / 1_000_000)
    assert u["by_step"] == {"generate": {"calls": 1, "input_tokens": 500, "output_tokens": 50}}
    assert u["failed_attempts"] == 0
    json.dumps(trace["usage"])  # must be storable in a results row


def test_agentic_engine_labels_every_step(monkeypatch):
    replies = {"plan": json.dumps({"route": "multi", "sub_queries": ["a", "b"]}),
               "grade": json.dumps({"sufficient": True, "missing": ""})}
    monkeypatch.setattr(base, "_call_provider", fake_provider(replies))
    g._GRAPHS.clear()
    _, _, trace = g.run_agentic(FakeStore(), "q")
    u = trace["usage"]
    assert [c["step"] for c in u["calls"]] == ["plan", "grade", "generate"]
    assert u["input_tokens"] == 100 + 300 + 500 and u["output_tokens"] == 20 + 10 + 50
    assert set(u["by_step"]) == {"plan", "grade", "generate"}
    assert trace["llm_calls"] == 3


def test_failed_attempts_are_counted_but_cost_nothing(monkeypatch):
    # the default model rate-limits, the fallback chain moves on (D-008)
    monkeypatch.setattr(base, "_call_provider", fake_provider({}, fail_models={QWEN}))
    _, _, trace = base.answer_question(FakeStore(), "q")
    u = trace["usage"]
    assert u["failed_attempts"] == 1
    assert u["calls"][0]["ok"] is False and "429" in u["calls"][0]["error"]
    assert trace["answered_by"] != DEFAULT
    assert u["total_tokens"] == 550  # only the successful call counts


def test_meter_is_reentrant_and_inactive_outside_a_request():
    usage.record("groq", QWEN, 1, 1, 5)  # no meter: silently ignored
    with usage.meter() as outer:
        with usage.meter() as inner:
            usage.record("groq", QWEN, 10, 2, 5)
        assert inner is outer and len(outer) == 1


def test_tracing_is_a_no_op_without_keys():
    # CI and the offline tests run without Langfuse keys
    assert obs.ENABLED is False

    def f(x):
        return x

    assert obs.observe(name="x")(f) is f
    with obs.trace_attributes(trace_name="t"):
        obs.update_span(output=1)
        obs.update_generation(model="m")
    obs.flush()
