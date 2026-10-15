"""
Offline tests for the D-018 / C-011 measurement fixes and the D-019
variance summary. No API keys, no network.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from langchain_core.documents import Document

import answer as base
from evaluation import eval as ev
from evaluation import variance
from evaluation.test import TestQuestion as Question


def d(text):
    return Document(page_content=text, metadata={})


# --- whole-word keyword matching (D-018) ---

@pytest.mark.parametrize("keyword, text, expected", [
    ("red", "He scored a century and retired.", False),   # substring only
    ("red", "The ball is red.", True),
    ("199", "The 1999 World Cup", False),
    ("199", "He was out for 199.", True),
    ("3,000", "first to reach 3,000 ODI runs", True),
    ("World Cup", "won the world cup in 2011", True),        # case-insensitive, multi-word
    ("Root", "Joe Root's average", True),                    # followed by an apostrophe
    ("Smith", "Smithson took a catch", False),
])
def test_keyword_in_matches_whole_words(keyword, text, expected):
    assert ev.keyword_in(keyword, text) is expected


def test_mrr_skips_substring_matches():
    docs = [d("He scored and retired."), d("A red ball is used.")]
    assert ev.calculate_mrr("red", docs) == 0.5


# --- the refusal rule (C-011) ---

def q(category):
    return Question(question="q", keywords=[], reference_answer="r", category=category)


@pytest.mark.parametrize("text", [
    base.REFUSAL_SENTENCE,
    '"I don’t have information about that in my knowledge base"',
    "  i don't have information about that in my knowledge base.  ",
])
def test_refusal_variants_are_recognised(text):
    assert ev.is_refusal(text)


def test_partial_refusal_is_not_the_fixed_sentence():
    assert not ev.is_refusal(base.REFUSAL_SENTENCE + " But Bradman averaged 99.94.")


def test_refusal_on_answerable_question_scores_1_by_rule():
    ruled = ev.refusal_rule(q("spanning"), base.REFUSAL_SENTENCE)
    assert (ruled.accuracy, ruled.completeness, ruled.relevance) == (1, 1, 1)


def test_refusal_on_out_of_scope_goes_to_the_judge():
    assert ev.refusal_rule(q("out_of_scope"), base.REFUSAL_SENTENCE) is None


def test_evaluate_answer_skips_the_judge_for_a_wrong_refusal(monkeypatch):
    monkeypatch.setattr(ev, "answer_question",
                        lambda *a, **k: (base.REFUSAL_SENTENCE, [], {"engine": "linear"}))

    def judge_must_not_run(*a, **k):
        raise AssertionError("judge called")

    monkeypatch.setattr(ev, "call_model", judge_must_not_run)
    trace = {}
    result, _, _ = ev.evaluate_answer(None, q("direct_fact"), trace_out=trace)
    assert result.accuracy == 1 and trace["scored_by"] == "rule"


def test_prompt_asks_for_the_refusal_sentence():
    prompt = base.SYSTEM_PROMPT_TEMPLATE.format(context="ctx", refusal=base.REFUSAL_SENTENCE)
    assert f'"{base.REFUSAL_SENTENCE}"' in prompt


# --- answer temperature (D-018) ---

def test_generate_from_context_passes_the_answer_temperature(monkeypatch):
    seen = {}

    def fake(system_prompt, question, history=None, model_label=None, allow_fallback=True, temperature=None):
        seen["temperature"] = temperature
        return "a", "m"

    monkeypatch.setattr(base, "generate_with_fallback", fake)
    monkeypatch.setattr(base, "ANSWER_TEMPERATURE", 0.0)
    base.generate_from_context("q", "ctx")
    assert seen["temperature"] == 0.0


def test_app_default_is_provider_temperature():
    assert base.ANSWER_TEMPERATURE is None


# --- variance summary (D-019) ---

def write_run(path, accuracies, category="spanning"):
    rows = [{"category": category, "answer_accuracy": a,
             "usage": {"cost_usd": 0.001}} for a in accuracies]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_variance_reports_mean_and_range(tmp_path):
    write_run(tmp_path / "a_r1.jsonl", [5, 4, 3])
    write_run(tmp_path / "a_r2.jsonl", [5, 5, 5])
    write_run(tmp_path / "b_r1.jsonl", [1, 1, 1])
    configs = variance.parse_configs([f"k=8={tmp_path}/a_r*.jsonl", f"low={tmp_path}/b_r*.jsonl"])
    assert list(configs) == ["k=8", "low"]  # label keeps its own "="
    s = variance.summarise(configs)
    assert s["k=8"]["runs"] == 2
    assert (s["k=8"]["min"], s["k=8"]["max"]) == (4.0, 5.0)
    assert s["k=8"]["correct"] == [2, 3]
    assert s["k=8"]["cost_per_1k"] == pytest.approx(1.0)
    assert not variance.overlaps(s["k=8"], s["low"])


def test_variance_category_filter(tmp_path):
    write_run(tmp_path / "r1.jsonl", [5, 5])
    with open(tmp_path / "r1.jsonl", "a", encoding="utf-8") as f:
        f.write("\n" + json.dumps({"category": "temporal", "answer_accuracy": 1}))
    s = variance.summarise({"x": [str(tmp_path / "r1.jsonl")]}, category="spanning")
    assert s["x"]["mean"] == 5.0
