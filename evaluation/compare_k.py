"""
evaluation/compare_k.py

Superseded for new runs by `evaluation.eval --k N --category spanning
--pin-model` (DECISIONS.md D-017), which pins the model and records results
in the same format as every other run. Kept for reference: the September
k comparison it was built for was never run to completion (D-007).

Tests different retriever k values (how many chunks are retrieved per
question) against the 'spanning' category specifically — the category
where the post-C-003 evaluation run showed a real gap: several
compound/comparison questions were correctly declined (no hallucination)
but only because retrieval didn't surface enough chunks to cover the full
question.

Scoped to the spanning category (20 questions) rather than the full
104-question set — this is specifically where the gap showed up, and
testing multiple k values against all 104 questions would multiply API
usage for categories where k=4 was already working fine.
"""

import sys
sys.path.append(".")

import json
from pathlib import Path
from evaluation.test import load_tests
from evaluation.eval import evaluate_retrieval, evaluate_answer
from ingest import ingest

K_VALUES_TO_TEST = [4, 6, 8]
RESULTS_FILE = str(Path(__file__).parent / "k_comparison_results.json")


def run_k_comparison():
    print("Building vector store...")
    vectorstore = ingest()

    all_tests = load_tests()
    spanning_tests = [t for t in all_tests if t.category == "spanning"]
    print(f"\nTesting {len(spanning_tests)} 'spanning' category questions across k = {K_VALUES_TO_TEST}\n")

    results_by_k = {}
    per_question_by_k = {}

    for k in K_VALUES_TO_TEST:
        print(f"{'=' * 70}")
        print(f"k = {k}")
        print(f"{'=' * 70}")

        mrr_scores = []
        ndcg_scores = []
        coverage_scores = []
        accuracy_scores = []
        completeness_scores = []
        relevance_scores = []
        per_question = []

        for i, test in enumerate(spanning_tests):
            retrieval_result = evaluate_retrieval(vectorstore, test, retriever_k=k)
            answer_result, generated_answer, _ = evaluate_answer(vectorstore, test, retriever_k=k)

            mrr_scores.append(retrieval_result.mrr)
            ndcg_scores.append(retrieval_result.ndcg)
            coverage_scores.append(retrieval_result.keyword_coverage)
            accuracy_scores.append(answer_result.accuracy)
            completeness_scores.append(answer_result.completeness)
            relevance_scores.append(answer_result.relevance)

            per_question.append({
                "question": test.question,
                "retrieval_coverage": retrieval_result.keyword_coverage,
                "answer_accuracy": answer_result.accuracy,
                "generated_answer": generated_answer,
            })

            print(f"  [{i+1}/{len(spanning_tests)}] coverage={retrieval_result.keyword_coverage:.0f}% "
                  f"accuracy={answer_result.accuracy:.1f} | {test.question[:60]}")

        per_question_by_k[k] = per_question

        avg = lambda lst: sum(lst) / len(lst)
        results_by_k[k] = {
            "mrr": avg(mrr_scores),
            "ndcg": avg(ndcg_scores),
            "coverage": avg(coverage_scores),
            "accuracy": avg(accuracy_scores),
            "completeness": avg(completeness_scores),
            "relevance": avg(relevance_scores),
        }

        print(f"\n  k={k} averages: MRR={results_by_k[k]['mrr']:.2f} "
              f"nDCG={results_by_k[k]['ndcg']:.2f} "
              f"coverage={results_by_k[k]['coverage']:.0f}% "
              f"accuracy={results_by_k[k]['accuracy']:.1f} "
              f"completeness={results_by_k[k]['completeness']:.1f} "
              f"relevance={results_by_k[k]['relevance']:.1f}\n")

    print(f"{'=' * 70}")
    print("SUMMARY — k comparison on 'spanning' category")
    print(f"{'=' * 70}")
    print(f"{'k':<6}{'MRR':<8}{'nDCG':<8}{'Coverage':<10}{'Accuracy':<10}{'Completeness':<14}{'Relevance':<10}")
    for k, r in results_by_k.items():
        print(f"{k:<6}{r['mrr']:<8.2f}{r['ndcg']:<8.2f}{r['coverage']:<10.0f}{r['accuracy']:<10.1f}{r['completeness']:<14.1f}{r['relevance']:<10.1f}")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "averages_by_k": results_by_k,
            "per_question_by_k": per_question_by_k,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nFull results saved to {RESULTS_FILE}")

    return results_by_k


if __name__ == "__main__":
    run_k_comparison()