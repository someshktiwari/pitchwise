"""
Offline tests for the agentic engine (graph.py) and the linear engine's
return shape (answer.py). No API keys, no network, no embedding model:
the LLM and the vector store are replaced with scripted fakes.

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


def doc(name, text=None):
    return Document(page_content=text or f"{name} text",
                    metadata={"source": f"{name}.md", "header_1": name, "header_2": ""})


class FakeStore:
    """Mimics Chroma's as_retriever(...).invoke(query) -> list[Document]."""

    def __init__(self, by_query=None, default=None):
        self.by_query = by_query or {}
        self.default = default or [doc("misc")]

    def as_retriever(self, search_kwargs):
        store = self

        class Retriever:
            def invoke(self, query):
                return store.by_query.get(query, store.default)

        return Retriever()


@pytest.fixture
def llm(monkeypatch):
    """Script the replies per prompt and record every call in order."""
    script = {g.PLAN_PROMPT: [], g.GRADE_PROMPT: [], g.REWRITE_PROMPT: []}
    calls = []

    def fake_call_llm(system, user, temperature=0):
        calls.append(system)
        return script[system].pop(0)

    def fake_generate(question, context, history=None, model_label=None, allow_fallback=True):
        calls.append("GENERATE")
        return "answer", "fake-model"

    monkeypatch.setattr(g, "call_llm", fake_call_llm)
    monkeypatch.setattr(base, "generate_from_context", fake_generate)
    g._GRAPHS.clear()  # each test gets a freshly compiled graph for its own fake store
    return script, calls


def plan_json(route, subs):
    return json.dumps({"route": route, "sub_queries": subs})


def grade_json(ok, missing=""):
    return json.dumps({"sufficient": ok, "missing": missing})


def run(store, q):
    return g.run_agentic(store, q)


def test_1_simple_route_is_plan_then_generate(llm):
    script, calls = llm
    script[g.PLAN_PROMPT].append(plan_json("simple", ["Bradman Test average"]))
    answer, _, trace = run(FakeStore(), "What is Bradman's average?")
    assert calls == [g.PLAN_PROMPT, "GENERATE"]
    assert answer == "answer"
    assert trace["route"] == "simple" and trace["llm_calls"] == 2
    assert trace["answered_by"] == "fake-model"


def test_2_malformed_plan_fails_open_to_simple(llm):
    script, calls = llm
    script[g.PLAN_PROMPT].append("not json at all")
    _, _, trace = run(FakeStore(), "q")
    assert trace["route"] == "simple" and trace["sub_queries"] == ["q"]
    assert g.GRADE_PROMPT not in calls
    assert trace["steps"][0].endswith(":fallback")


def test_3_too_many_sub_queries_fails_open(llm):
    script, _ = llm
    script[g.PLAN_PROMPT].append(plan_json("multi", ["a", "b", "c", "d"]))
    _, _, trace = run(FakeStore(), "q")
    assert trace["route"] == "simple"


def test_4_retrieve_puts_original_first_dedupes_and_caps(llm):
    script, _ = llm
    script[g.PLAN_PROMPT].append(plan_json("multi", ["a", "b", "c"]))
    script[g.GRADE_PROMPT].append(grade_json(True))
    shared = doc("shared")
    store = FakeStore({
        "q": [doc("orig1"), shared, doc("orig2"), doc("orig3")],
        "a": [shared, doc("a1"), doc("a2"), doc("a3")],
        "b": [doc("b1"), doc("b2"), doc("b3"), doc("b4")],
        "c": [doc("c1"), doc("c2"), doc("c3"), doc("c4")],
    })
    _, docs, _ = run(store, "q")
    names = [d.metadata["header_1"] for d in docs]
    assert names[:4] == ["orig1", "shared", "orig2", "orig3"]  # original question first
    assert names.count("shared") == 1                         # duplicates removed
    assert len(docs) == g.MAX_CHUNKS                          # capped


def test_5_multi_sufficient_is_plan_grade_generate(llm):
    script, calls = llm
    script[g.PLAN_PROMPT].append(plan_json("multi", ["ODI powerplay", "T20I powerplay"]))
    script[g.GRADE_PROMPT].append(grade_json(True))
    _, _, trace = run(FakeStore(), "How many overs separate ODI and T20I powerplays?")
    assert calls == [g.PLAN_PROMPT, g.GRADE_PROMPT, "GENERATE"]
    assert trace["llm_calls"] == 3 and trace["rewrites"] == 0


def test_6_rewrite_loop_caps_at_two_then_generates(llm):
    script, calls = llm
    script[g.PLAN_PROMPT].append(plan_json("multi", ["a", "b"]))
    script[g.GRADE_PROMPT] += [grade_json(False, "x"), grade_json(False, "y"), grade_json(False, "z")]
    script[g.REWRITE_PROMPT] += ["query x", "query y"]
    _, _, trace = run(FakeStore(), "q")
    assert calls == [g.PLAN_PROMPT, g.GRADE_PROMPT, g.REWRITE_PROMPT, g.GRADE_PROMPT,
                     g.REWRITE_PROMPT, g.GRADE_PROMPT, "GENERATE"]
    assert trace["rewrites"] == 2 and trace["llm_calls"] == 7
    assert trace["sub_queries"][-2:] == ["query x", "query y"]


def test_7_rewrite_then_sufficient_stops_early(llm):
    script, calls = llm
    script[g.PLAN_PROMPT].append(plan_json("multi", ["a", "b"]))
    script[g.GRADE_PROMPT] += [grade_json(False, "x"), grade_json(True)]
    script[g.REWRITE_PROMPT].append("query x")
    _, _, trace = run(FakeStore(), "q")
    assert trace["rewrites"] == 1 and calls[-1] == "GENERATE" and trace["llm_calls"] == 5


def test_8_malformed_grade_fails_open_to_generate(llm):
    script, calls = llm
    script[g.PLAN_PROMPT].append(plan_json("multi", ["a", "b"]))
    script[g.GRADE_PROMPT].append("{broken")
    _, _, trace = run(FakeStore(), "q")
    assert g.REWRITE_PROMPT not in calls and trace["rewrites"] == 0


def test_9_duplicate_rewrite_falls_back_to_missing_phrase(llm):
    script, _ = llm
    script[g.PLAN_PROMPT].append(plan_json("multi", ["a", "b"]))
    script[g.GRADE_PROMPT] += [grade_json(False, "Fab Four nations"), grade_json(True)]
    script[g.REWRITE_PROMPT].append("a")  # the model repeats a query already tried
    _, _, trace = run(FakeStore(), "q")
    assert trace["sub_queries"][-1] == "Fab Four nations"


def test_10_parse_json_tolerates_fences_and_text():
    assert g.parse_json('{"a": 1}') == {"a": 1}
    assert g.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert g.parse_json('Sure! {"a": 1} hope that helps') == {"a": 1}
    assert g.parse_json("{broken") is None
    assert g.parse_json(None) is None


def test_11_linear_engine_returns_answer_docs_trace(monkeypatch):
    def fake_generate(question, context, history=None, model_label=None, allow_fallback=True):
        return "linear answer", "fake-model"

    monkeypatch.setattr(base, "generate_from_context", fake_generate)
    store = FakeStore({"q": [doc("one"), doc("two")]})
    answer, docs, trace = base.answer_question(store, "q")
    assert answer == "linear answer" and len(docs) == 2
    assert trace["engine"] == "linear" and trace["llm_calls"] == 1


def test_12_no_fallback_tries_only_one_model():
    order = base.build_try_order(model_label=None, allow_fallback=False)
    assert order == [base.FALLBACK_ORDER[0]]
    assert base.build_try_order("Gemini: 3.5-flash-lite", allow_fallback=False) == ["Gemini: 3.5-flash-lite"]
    assert len(base.build_try_order()) == len(base.FALLBACK_ORDER)
