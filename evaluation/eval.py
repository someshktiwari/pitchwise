"""
evaluation/eval.py

Evaluates Pitchwise on two dimensions:
  - Retrieval quality (MRR, nDCG, keyword coverage) — did the retriever
    find chunks containing the expected keywords?
  - Answer quality (LLM-as-judge: accuracy, completeness, relevance) —
    is the generated answer actually correct?

Adapted to use answer.py's own multi-provider setup instead of a paid
third-party judge API. The retrieval-metric math (MRR, nDCG) follows a
standard information-retrieval evaluation approach. The judge step uses
our own multi-provider answer.py (Groq default, per DECISIONS.md D-006)
and parses a JSON response manually.

The out_of_scope test category exists specifically to measure the
hallucination gap discovered during manual testing (DECISIONS.md —
see the note on grounding failures): the system prompt says "if you
don't know, say so," but this wasn't being reliably followed. This
harness measures that gap directly rather than relying on spot-checking.
"""

import json
import math
import time
from pathlib import Path
from pydantic import BaseModel, Field

from evaluation.test import TestQuestion, load_tests
from answer import answer_question, retrieve_context, call_model, MODEL_OPTIONS
import observability as obs
import usage

# Judge model: Gemini 3.5-flash-lite — deliberately a different provider
# than the default answer-generation model (Groq qwen3.8-27b, per D-006).
# Two reasons: (1) avoids self-evaluation bias, where a model judging its
# own family's answers may rate them more favorably than an independent
# judge would; (2) splits the ~200+ calls a full evaluation run makes
# across two separate rate-limit budgets instead of concentrating them
# on one provider.
JUDGE_PROVIDER, JUDGE_MODEL = MODEL_OPTIONS["Gemini: 3.5-flash-lite"]


class RetrievalEval(BaseModel):
    """Evaluation metrics for retrieval performance."""

    mrr: float = Field(description="Mean Reciprocal Rank - average across all keywords")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain (binary relevance)")
    keywords_found: int
    total_keywords: int
    keyword_coverage: float


class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation of answer quality."""

    feedback: str
    accuracy: float = Field(description="1 (wrong) to 5 (perfectly accurate)")
    completeness: float = Field(description="1 (missing key info) to 5 (fully complete)")
    relevance: float = Field(description="1 (off-topic) to 5 (directly on-topic)")


def calculate_mrr(keyword: str, retrieved_docs: list) -> float:
    """Reciprocal rank for a single keyword (case-insensitive)."""
    keyword_lower = keyword.lower()
    for rank, doc in enumerate(retrieved_docs, start=1):
        if keyword_lower in doc.page_content.lower():
            return 1.0 / rank
    return 0.0


def calculate_dcg(relevances: list[int], k: int) -> float:
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)
    return dcg


def calculate_ndcg(keyword: str, retrieved_docs: list, k: int = 10) -> float:
    keyword_lower = keyword.lower()
    relevances = [1 if keyword_lower in doc.page_content.lower() else 0 for doc in retrieved_docs[:k]]
    dcg = calculate_dcg(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(vectorstore, test: TestQuestion, ndcg_k: int = 10, retriever_k=None) -> RetrievalEval:
    """Evaluate retrieval performance for one test question.

    ndcg_k is the cutoff used when computing nDCG (how many top results to
    consider) — unrelated to how many chunks the retriever actually
    fetches. retriever_k controls that directly, overriding the
    production default (D-007) — used for testing different k values.

    For out_of_scope questions (empty keywords list), retrieval isn't
    meaningfully scorable — there's nothing correct to retrieve — so this
    returns a zeroed result rather than dividing by zero.
    """
    if not test.keywords:
        return RetrievalEval(mrr=0.0, ndcg=0.0, keywords_found=0, total_keywords=0, keyword_coverage=0.0)

    retrieved_docs, _ = retrieve_context(vectorstore, test.question, k=retriever_k)

    mrr_scores = [calculate_mrr(kw, retrieved_docs) for kw in test.keywords]
    avg_mrr = sum(mrr_scores) / len(mrr_scores)

    ndcg_scores = [calculate_ndcg(kw, retrieved_docs, ndcg_k) for kw in test.keywords]
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)

    keywords_found = sum(1 for score in mrr_scores if score > 0)
    total_keywords = len(test.keywords)
    keyword_coverage = (keywords_found / total_keywords * 100) if total_keywords > 0 else 0.0

    return RetrievalEval(
        mrr=avg_mrr,
        ndcg=avg_ndcg,
        keywords_found=keywords_found,
        total_keywords=total_keywords,
        keyword_coverage=keyword_coverage,
    )


def _parse_judge_json(raw_text: str) -> dict:
    """Judge models sometimes wrap JSON in markdown fences or add stray
    text around it — strip fences and parse defensively rather than
    assuming a clean response."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())


def evaluate_answer(vectorstore, test: TestQuestion, retriever_k=None, engine="linear",
                    allow_fallback=True, trace_out=None) -> tuple[AnswerEval, str, list]:
    """Evaluate answer quality using LLM-as-a-judge.

    retriever_k overrides the retriever's default chunk count (D-007),
    same as evaluate_retrieval — used for testing different k values.

    Returns (AnswerEval, generated_answer, retrieved_docs).
    """
    generated_answer, retrieved_docs, trace = answer_question(
        vectorstore, test.question, k=retriever_k, engine=engine, allow_fallback=allow_fallback)
    if trace_out is not None:
        trace_out.clear()
        trace_out.update(trace)

    judge_system_prompt = (
        "You are an expert evaluator assessing the quality of answers. "
        "Evaluate the generated answer by comparing it to the reference answer. "
        "Only give 5/5 scores for perfect answers. "
        "Respond with ONLY a JSON object, no markdown fences, no other text, "
        'in exactly this shape: {"feedback": "...", "accuracy": <1-5>, '
        '"completeness": <1-5>, "relevance": <1-5>}'
    )

    judge_question = f"""Question:
{test.question}

Generated Answer:
{generated_answer}

Reference Answer:
{test.reference_answer}

Evaluate on three dimensions (1=very poor to 5=ideal):
1. Accuracy: factual correctness vs. the reference answer. If the answer is wrong, accuracy must be 1.
2. Completeness: does it cover all information from the reference answer?
3. Relevance: does it directly address the question without unnecessary extra information?

For out_of_scope questions specifically: if the reference answer says the knowledge
base doesn't cover this, a generated answer that correctly declines to answer
(rather than guessing from general knowledge) should score 5 on all dimensions.
A generated answer that confidently answers anyway, even if factually correct
in the real world, should score 1 on accuracy — it is not grounded in the
provided knowledge base, which is the actual failure being measured here."""

    # The judge's tokens are metered separately from the answer's, so a run
    # reports what answering costs and what grading costs (D-012).
    with usage.meter() as judge_calls, usage.step("judge"):
        raw_response = call_model(JUDGE_PROVIDER, JUDGE_MODEL, judge_system_prompt, judge_question)
    if trace_out is not None:
        trace_out["judge_usage"] = usage.summarise(judge_calls)
    parsed = _parse_judge_json(raw_response)
    answer_eval = AnswerEval(**parsed)

    return answer_eval, generated_answer, retrieved_docs


def evaluate_all_retrieval(vectorstore):
    """Evaluate retrieval for every test question, yielding progress as it
    goes — used by the dashboard (evaluator.py) to drive a progress bar.

    Retries each test on transient failures (dropped connections, etc.)
    up to 3 times; if a test still fails after retries, it's skipped
    (not counted in the running average) rather than crashing the whole
    dashboard run — same reasoning as the console evaluation's resilience
    (see DECISIONS.md C-002)."""
    tests = load_tests()
    total_tests = len(tests)
    for index, test in enumerate(tests):
        try:
            result = _retry(lambda: evaluate_retrieval(vectorstore, test))
        except Exception as e:
            print(f"  Skipping '{test.question}' after retries failed: {e}")
            progress = (index + 1) / total_tests
            continue
        progress = (index + 1) / total_tests
        yield test, result, progress


def evaluate_all_answers(vectorstore):
    """Evaluate answer quality for every test question, yielding progress
    as it goes — used by the dashboard (evaluator.py) to drive a progress
    bar.

    Retries each test on transient failures up to 3 times; if a test
    still fails after retries, it's skipped rather than crashing the
    whole dashboard run (see DECISIONS.md C-002)."""
    tests = load_tests()
    total_tests = len(tests)
    for index, test in enumerate(tests):
        try:
            result = _retry(lambda: evaluate_answer(vectorstore, test))[0]
        except Exception as e:
            print(f"  Skipping '{test.question}' after retries failed: {e}")
            progress = (index + 1) / total_tests
            continue
        progress = (index + 1) / total_tests
        yield test, result, progress


RESULTS_FILE = str(Path(__file__).parent / "results.jsonl")


def _retry(fn, max_attempts=3, delay_seconds=3):
    """Retry a call on transient failures (network drops, timeouts) rather
    than crashing the whole evaluation run. Not for genuine errors like bad
    API keys — those will exhaust retries and raise, which is correct."""
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if attempt < max_attempts:
                print(f"    (retry {attempt}/{max_attempts - 1} after error: {e})")
                time.sleep(delay_seconds)
    raise last_error


def _load_completed_questions():
    """Read any results already saved from a previous run, so a rerun
    after a crash skips questions already evaluated instead of redoing
    (and re-spending API calls on) work that already succeeded."""
    completed = set()
    if Path(RESULTS_FILE).exists():
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    completed.add(json.loads(line)["question"])
    return completed


def run_full_evaluation(vectorstore, resume=True, engine="linear", results_file=None, allow_fallback=True,
                        tests_file=None):
    """Run retrieval + answer evaluation across the full test set.

    Saves each result to evaluation/results.jsonl as soon as it's computed
    (not just at the end), and retries transient failures (e.g. dropped
    connections) rather than crashing the whole run — a single network
    hiccup partway through a 100+ question run should not cost the
    progress already made.

    If resume=True (default) and results.jsonl already has entries from a
    prior run, those questions are skipped rather than re-evaluated.
    """
    global RESULTS_FILE
    if results_file:
        RESULTS_FILE = str(results_file)  # one file per run, never mix two runs (D-010)
    print(f"Engine: {engine} | fallback: {'on' if allow_fallback else 'off (pinned model)'} | results: {RESULTS_FILE}")

    tests = load_tests(tests_file)
    print(f"Test set: {tests_file or 'evaluation/tests.jsonl'} ({len(tests)} questions)")
    already_done = _load_completed_questions() if resume else set()
    if already_done:
        print(f"Resuming: {len(already_done)} questions already completed, skipping those.\n")

    results_file = open(RESULTS_FILE, "a", encoding="utf-8")
    results = []

    for i, test in enumerate(tests):
        if test.question in already_done:
            continue

        print(f"[{i+1}/{len(tests)}] {test.category}: {test.question}")

        try:
            retrieval_result = _retry(lambda: evaluate_retrieval(vectorstore, test))
            trace = {}
            with obs.trace_attributes(session_id=Path(RESULTS_FILE).stem,
                                      metadata={"category": test.category}):
                answer_result, generated_answer, _ = _retry(lambda: evaluate_answer(
                    vectorstore, test, engine=engine, allow_fallback=allow_fallback, trace_out=trace))
        except Exception as e:
            print(f"  FAILED after retries: {e}\n")
            continue

        result_record = {
            "question": test.question,
            "category": test.category,
            "retrieval_mrr": retrieval_result.mrr,
            "retrieval_ndcg": retrieval_result.ndcg,
            "retrieval_keyword_coverage": retrieval_result.keyword_coverage,
            "answer_accuracy": answer_result.accuracy,
            "answer_completeness": answer_result.completeness,
            "answer_relevance": answer_result.relevance,
            "answer_feedback": answer_result.feedback,
            "generated_answer": generated_answer,
            "engine": engine,
            "route": trace.get("route"),
            "sub_queries": trace.get("sub_queries"),
            "rewrites": trace.get("rewrites"),
            "llm_calls": trace.get("llm_calls"),
            "latency_ms": trace.get("latency_ms"),
            "answered_by": trace.get("answered_by"),
            "routing_models": trace.get("routing_models"),
            "usage": trace.get("usage"),              # answering: tokens, list-price cost, per call
            "judge_usage": trace.get("judge_usage"),  # grading, kept separate
        }
        results_file.write(json.dumps(result_record, ensure_ascii=False) + "\n")
        results_file.flush()  # write to disk immediately, don't wait for buffer

        results.append(result_record)

        print(f"  Retrieval: MRR={retrieval_result.mrr:.2f} nDCG={retrieval_result.ndcg:.2f} coverage={retrieval_result.keyword_coverage:.0f}%")
        print(f"  Answer: accuracy={answer_result.accuracy:.1f} completeness={answer_result.completeness:.1f} relevance={answer_result.relevance:.1f}")
        u = trace.get("usage") or {}
        if u:
            cost = "n/a" if u.get("cost_usd") is None else f"${u['cost_usd']:.6f}"
            print(f"  Usage: {u.get('input_tokens')} in / {u.get('output_tokens')} out tokens, {cost} at list price")
        print()

    results_file.close()
    obs.flush()

    # Include previously-completed results (from a resumed run) in the summary
    all_results = results
    if already_done:
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            all_results = [json.loads(line) for line in f if line.strip()]

    print("=" * 70)
    print("SUMMARY BY CATEGORY")
    print("=" * 70)
    categories = sorted(set(r["category"] for r in all_results))
    for cat in categories:
        cat_results = [r for r in all_results if r["category"] == cat]
        avg_accuracy = sum(r["answer_accuracy"] for r in cat_results) / len(cat_results)
        avg_completeness = sum(r["answer_completeness"] for r in cat_results) / len(cat_results)
        avg_relevance = sum(r["answer_relevance"] for r in cat_results) / len(cat_results)
        print(f"{cat} (n={len(cat_results)}): accuracy={avg_accuracy:.1f} completeness={avg_completeness:.1f} relevance={avg_relevance:.1f}")

    overall_accuracy = sum(r["answer_accuracy"] for r in all_results) / len(all_results)
    overall_completeness = sum(r["answer_completeness"] for r in all_results) / len(all_results)
    overall_relevance = sum(r["answer_relevance"] for r in all_results) / len(all_results)
    print()
    print(f"OVERALL (n={len(all_results)}): accuracy={overall_accuracy:.1f}/5 completeness={overall_completeness:.1f}/5 relevance={overall_relevance:.1f}/5")

    metered = [r["usage"] for r in all_results if r.get("usage")]
    if metered:
        tokens = sum(u["total_tokens"] for u in metered) / len(metered)
        costs = [u["cost_usd"] for u in metered if u["cost_usd"] is not None]
        print(f"USAGE (n={len(metered)}): {tokens:.0f} tokens per question"
              + (f", ${sum(costs) / len(costs) * 1000:.3f} per 1,000 questions at list price" if costs else ""))

    return all_results


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.append(".")
    from ingest import ingest

    parser = argparse.ArgumentParser(description="Run the Pitchwise evaluation.")
    parser.add_argument("--engine", choices=["linear", "agentic"], default="linear")
    parser.add_argument("--results", help="results file for this run (default: evaluation/results.jsonl)")
    parser.add_argument("--pin-model", action="store_true",
                        help="disable provider fallback so the whole run uses one model (D-010)")
    parser.add_argument("--no-resume", action="store_true", help="ignore existing results in the file")
    parser.add_argument("--tests", help="test set file (default: evaluation/tests.jsonl, the frozen v2 set)")
    args = parser.parse_args()

    store = ingest()
    run_full_evaluation(store, resume=not args.no_resume, engine=args.engine,
                        results_file=args.results, allow_fallback=not args.pin_model,
                        tests_file=args.tests)
