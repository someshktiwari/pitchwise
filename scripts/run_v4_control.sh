#!/usr/bin/env bash
# The v4 control (D-019): k=4, k=8 and the agentic engine on the 32 v3
# `spanning` questions, three runs each, answers at temperature 0 and the
# D-018 / C-011 measurement fixes in place, all pinned to one model.
#
#   bash scripts/run_v4_control.sh            # all three runs
#   RUNS="2 3" bash scripts/run_v4_control.sh # only runs 2 and 3
#
# Each run writes its own results file (D-010). A run that stops part-way
# resumes where it left off when re-run. Summarise with:
#
#   uv run python -m evaluation.variance \
#       "k=4=evaluation/results_v4_linear_k4_r*.jsonl" \
#       "k=8=evaluation/results_v4_linear_k8_r*.jsonl" \
#       "agentic=evaluation/results_v4_agentic_r*.jsonl" --category spanning
set -euo pipefail

MODEL="${MODEL:-Groq: qwen3.8-27b (fastest, default)}"  # the model every v2/v3 run was pinned to (D-016)
RUNS="${RUNS:-1 2 3}"
COMMON=(--tests evaluation/tests_v3.jsonl --category spanning --pin-model --model "$MODEL")

for r in $RUNS; do
  uv run python -m evaluation.eval "${COMMON[@]}" --engine linear --k 4 \
    --results "evaluation/results_v4_linear_k4_r${r}.jsonl"
  uv run python -m evaluation.eval "${COMMON[@]}" --engine linear --k 8 \
    --results "evaluation/results_v4_linear_k8_r${r}.jsonl"
  uv run python -m evaluation.eval "${COMMON[@]}" --engine agentic \
    --results "evaluation/results_v4_agentic_r${r}.jsonl"
done
