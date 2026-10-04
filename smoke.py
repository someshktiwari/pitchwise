"""
smoke.py

Live check of both engines on a few real questions from the test set:
5 spanning, 3 direct_fact, 2 out_of_scope. Uses real API keys from .env
(about 40 LLM calls in total). Not part of the offline test suite.

Run:  uv run python smoke.py
"""

import json

from answer import answer_question
from evaluation.test import load_tests
from ingest import ingest

PICK = {"spanning": 5, "direct_fact": 3, "out_of_scope": 2}


def main():
    tests = load_tests()
    chosen = []
    for category, n in PICK.items():
        chosen += [t for t in tests if t.category == category][:n]

    store = ingest()
    for t in chosen:
        print("=" * 80)
        print(f"[{t.category}] {t.question}")
        print(f"Reference: {t.reference_answer}")
        for engine in ("linear", "agentic"):
            text, docs, trace = answer_question(store, t.question, engine=engine)
            print(f"\n  {engine.upper()}: {text.strip()[:300]}")
            print(f"  trace: {json.dumps({k: trace[k] for k in ('route', 'sub_queries', 'rewrites', 'llm_calls', 'latency_ms', 'answered_by')})}")
    print("=" * 80)


if __name__ == "__main__":
    main()
