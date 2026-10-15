"""
evaluation/variance.py

Summarises repeated runs of the same configuration (D-019): the mean and
range of each configuration's score across its runs, so a difference
between two configurations can be read against the run-to-run spread
rather than from one run each.

Reads the results files evaluation/eval.py wrote; makes no LLM calls.

Run:
  uv run python -m evaluation.variance \\
      "k=4=evaluation/results_v4_linear_k4_r*.jsonl" \\
      "k=8=evaluation/results_v4_linear_k8_r*.jsonl" \\
      "agentic=evaluation/results_v4_agentic_r*.jsonl" \\
      --category spanning

Each argument is LABEL=glob. The label is everything before the last "=",
so labels such as "k=8" work. Prints a Markdown table ready for
EVAL_RESULTS.md, then whether each pair of configurations' ranges overlap.
"""

import argparse
import glob
import json
from itertools import combinations
from pathlib import Path
from statistics import mean

CORRECT_THRESHOLD = 4.0  # same threshold as compare_runs.py


def load_rows(path, category=None):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if category:
        rows = [r for r in rows if r.get("category") == category]
    return rows


def run_stats(rows):
    """One run: mean accuracy, correct answers, n, cost per 1,000 questions."""
    acc = [r["answer_accuracy"] for r in rows]
    costs = [r["usage"]["cost_usd"] for r in rows
             if r.get("usage") and r["usage"].get("cost_usd") is not None]
    return {
        "n": len(rows),
        "accuracy": mean(acc) if acc else float("nan"),
        "correct": sum(1 for a in acc if a >= CORRECT_THRESHOLD),
        "cost_per_1k": mean(costs) * 1000 if costs else None,
        "ruled": sum(1 for r in rows if r.get("scored_by") == "rule"),
    }


def summarise(configs, category=None):
    """configs: {label: [paths]} -> {label: summary dict}."""
    out = {}
    for label, paths in configs.items():
        runs = [run_stats(load_rows(p, category)) for p in paths]
        runs = [r for r in runs if r["n"]]
        if not runs:
            continue
        accs = [r["accuracy"] for r in runs]
        costs = [r["cost_per_1k"] for r in runs if r["cost_per_1k"] is not None]
        out[label] = {
            "runs": len(runs),
            "n": sorted({r["n"] for r in runs}),
            "mean": mean(accs),
            "min": min(accs),
            "max": max(accs),
            "per_run": accs,
            "correct": [r["correct"] for r in runs],
            "cost_per_1k": mean(costs) if costs else None,
            "ruled": sum(r["ruled"] for r in runs),
        }
    return out


def overlaps(a, b):
    return a["min"] <= b["max"] and b["min"] <= a["max"]


def markdown(summary):
    lines = [
        "| Configuration | Runs | n | Mean accuracy | Range | Per run | Correct (≥4) per run | $ per 1,000 Q |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, s in summary.items():
        cost = "n/a" if s["cost_per_1k"] is None else f"${s['cost_per_1k']:.2f}"
        n = "/".join(str(x) for x in s["n"])
        lines.append(
            f"| {label} | {s['runs']} | {n} | {s['mean']:.2f} | {s['min']:.2f}–{s['max']:.2f} "
            f"| {', '.join(f'{a:.2f}' for a in s['per_run'])} "
            f"| {', '.join(str(c) for c in s['correct'])} | {cost} |")
    return "\n".join(lines)


def parse_configs(args):
    configs = {}
    for arg in args:
        label, _, pattern = arg.rpartition("=")
        if not label:
            raise SystemExit(f"Expected LABEL=glob, got {arg!r}")
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise SystemExit(f"No files match {pattern!r} for {label!r}")
        configs[label] = paths
    return configs


def main(argv=None):
    parser = argparse.ArgumentParser(description="Mean and range of repeated evaluation runs.")
    parser.add_argument("configs", nargs="+", help="LABEL=glob of results files, one per configuration")
    parser.add_argument("--category", help="only score this category (e.g. spanning)")
    args = parser.parse_args(argv)

    summary = summarise(parse_configs(args.configs), args.category)
    print(markdown(summary))
    print()
    for (la, a), (lb, b) in combinations(summary.items(), 2):
        gap = a["mean"] - b["mean"]
        verdict = "ranges overlap: not separable at this many runs" if overlaps(a, b) \
            else "ranges do not overlap"
        print(f"{la} vs {lb}: mean gap {gap:+.2f}; {verdict}")
    ruled = sum(s["ruled"] for s in summary.values())
    if ruled:
        print(f"\n{ruled} answer(s) were the fixed refusal on an answerable question, scored 1 by rule (C-011).")


if __name__ == "__main__":
    main()
