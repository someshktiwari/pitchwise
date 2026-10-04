"""
evaluation/mechanism.py

Mechanism check for the agentic engine (DECISIONS.md D-010).

The eval's retrieval metrics (MRR, nDCG, keyword coverage) always score the
v1 retriever (top-k for the original question), whatever --engine is set to.
So a spanning question can show coverage=0% and still score 5/5 under the
agentic engine. That is either because the agentic retrieval found the facts
(the engine working as designed) or because the model answered from general
knowledge (a grounding leak).

This script tells the two apart. For each chosen question it runs the agentic
engine and measures keyword coverage on the FINAL context the answer was
generated from, next to the v1 coverage.

Run:  uv run python -m evaluation.mechanism            # Q66 Q67 Q76 Q79
      uv run python -m evaluation.mechanism 66 67      # specific questions
      uv run python -m evaluation.mechanism --spanning # every spanning question

Uses a non-default model for generation so it does not spend the pinned
eval model's daily quota.
"""

import sys

from answer import retrieve_context
from evaluation.test import load_tests
from graph import run_agentic
from ingest import ingest

DEFAULT_QUESTIONS = [66, 67, 76, 79]
MODEL = "Groq: gpt-oss-20b"


def coverage(keywords, docs):
    if not keywords:
        return None
    text = " ".join(d.page_content.lower() for d in docs)
    found = [k for k in keywords if k.lower() in text]
    return 100 * len(found) / len(keywords), [k for k in keywords if k not in found]


def main():
    tests = load_tests()
    args = sys.argv[1:]
    if "--spanning" in args:
        numbers = [i for i, t in enumerate(tests, 1) if t.category == "spanning"]
    else:
        numbers = [int(a) for a in args] or DEFAULT_QUESTIONS

    store = ingest()
    for n in numbers:
        t = tests[n - 1]
        v1_docs, _ = retrieve_context(store, t.question)
        answer, docs, trace = run_agentic(store, t.question, model_label=MODEL, allow_fallback=False)
        v1_cov, _ = coverage(t.keywords, v1_docs) or (None, [])
        ag_cov, missing = coverage(t.keywords, docs) or (None, [])
        print("=" * 80)
        print(f"Q{n} [{t.category}] {t.question}")
        print(f"  keywords: {t.keywords}")
        print(f"  v1 coverage: {v1_cov:.0f}%  ({len(v1_docs)} chunks)")
        print(f"  agentic coverage: {ag_cov:.0f}%  ({len(docs)} chunks)  missing: {missing}")
        print(f"  route={trace['route']} rewrites={trace['rewrites']} calls={trace['llm_calls']}")
        print(f"  steps: {trace['steps']}")
        print(f"  answer: {answer.strip()[:200]}")
    print("=" * 80)


if __name__ == "__main__":
    main()
