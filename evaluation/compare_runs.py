"""
evaluation/compare_runs.py

Side-by-side dashboard of saved evaluation runs (DECISIONS.md D-010).

evaluator.py re-runs the evaluation live (104 questions of LLM calls) and
only for the linear engine. This dashboard reads the results files that
evaluation/eval.py already wrote, so it makes no LLM calls and can compare
any runs: v1 linear against one or more v2 agentic runs.

Run:  uv run python -m evaluation.compare_runs
      uv run python -m evaluation.compare_runs L1=evaluation/results_linear_oct.jsonl A1=evaluation/results_agentic_r1.jsonl

Each argument is LABEL=path. With no arguments, the default runs below are
used if their files exist.
"""

import json
import sys
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.express as px

DEFAULT_RUNS = {
    "Linear (v1)": "evaluation/results_linear_oct.jsonl",
    "Agentic run 1": "evaluation/results_agentic_r1.jsonl",
    "Agentic run 2": "evaluation/results_agentic_r2.jsonl",
}
CATEGORY_ORDER = ["direct_fact", "spanning", "temporal", "out_of_scope"]
FAIL_THRESHOLD = 3.0  # an answer accuracy below this counts as a failed question
PORT = 7870  # own port, so it never collides with the chat app on 7860
CORRECT_THRESHOLD = 4.0  # an answer accuracy at or above this counts as correct, for cost per correct answer
DAILY_VOLUME = 10_000     # questions per day, for the monthly cost projection


def load_run(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def parse_args(argv):
    if not argv:
        return {label: p for label, p in DEFAULT_RUNS.items() if Path(p).exists()}
    runs = {}
    for arg in argv:
        label, _, path = arg.partition("=")
        runs[label] = path
    return runs


def summarise(runs):
    """One row per (run, category) plus an overall row per run."""
    rows = []
    for label, df in runs.items():
        groups = [(c, df[df["category"] == c]) for c in CATEGORY_ORDER if (df["category"] == c).any()]
        groups.append(("overall", df))
        for category, g in groups:
            row = {
                "Run": label,
                "Category": category,
                "n": len(g),
                "Accuracy": round(g["answer_accuracy"].mean(), 2),
                "Completeness": round(g["answer_completeness"].mean(), 2),
                "Failures (acc < 3)": int((g["answer_accuracy"] < FAIL_THRESHOLD).sum()),
            }
            if "llm_calls" in g and g["llm_calls"].notna().any():
                row["Avg LLM calls"] = round(g["llm_calls"].mean(), 1)
                row["Avg latency (s)"] = round(g["latency_ms"].mean() / 1000, 1)
            rows.append(row)
    return pd.DataFrame(rows)


def route_mix(runs):
    """How the agentic engine routed each category: simple vs multi, and rewrites used."""
    rows = []
    for label, df in runs.items():
        if "route" not in df or df["route"].isna().all():
            continue
        for category in CATEGORY_ORDER:
            g = df[df["category"] == category]
            if g.empty:
                continue
            rows.append({
                "Run": label,
                "Category": category,
                "multi route": int((g["route"] == "multi").sum()),
                "simple route": int((g["route"] == "simple").sum()),
                "questions with a rewrite": int((g["rewrites"].fillna(0) > 0).sum()),
            })
    return pd.DataFrame(rows)


def _metered(df):
    """Rows that recorded token usage (runs from D-012 on); older runs have none."""
    if "usage" not in df:
        return df.iloc[0:0]
    return df[df["usage"].apply(lambda u: isinstance(u, dict) and u.get("total_tokens") is not None)]


def cost_table(runs):
    """Tokens and list-price cost per question, per run and category (D-012).
    Cost per correct answer divides the run's total answering cost by the
    number of answers judged correct, so a cheap engine that answers badly
    doesn't look cheap. The judge's cost is shown separately: it is the cost
    of measuring, not of answering."""
    rows = []
    for label, df in runs.items():
        m = _metered(df)
        if m.empty:
            continue
        for category in CATEGORY_ORDER + ["overall"]:
            g = m if category == "overall" else m[m["category"] == category]
            if g.empty:
                continue
            u = g["usage"]
            costs = u.apply(lambda x: x.get("cost_usd"))
            known = costs.notna().all()
            per_q = costs.mean() if known else None
            correct = int((g["answer_accuracy"] >= CORRECT_THRESHOLD).sum())
            judge = g["judge_usage"].apply(lambda x: (x or {}).get("cost_usd")) if "judge_usage" in g else None
            rows.append({
                "Run": label,
                "Category": category,
                "n": len(g),
                "Input tokens / q": round(u.apply(lambda x: x["input_tokens"]).mean()),
                "Output tokens / q": round(u.apply(lambda x: x["output_tokens"]).mean()),
                "Cost / q ($)": None if per_q is None else round(per_q, 6),
                "Cost / 1k q ($)": None if per_q is None else round(per_q * 1000, 3),
                f"Cost / month at {DAILY_VOLUME:,}/day ($)": None if per_q is None else round(per_q * DAILY_VOLUME * 30, 2),
                "Cost / correct answer ($)": (round(costs.sum() / correct, 6) if known and correct else None),
                "Judge cost / q ($)": (round(judge.mean(), 6) if judge is not None and judge.notna().all() else None),
                "Failed attempts": int(u.apply(lambda x: x.get("failed_attempts", 0)).sum()),
            })
    return pd.DataFrame(rows)


def step_breakdown(runs):
    """Share of each run's tokens spent in each step (plan, grade, rewrite,
    generate): where the agentic engine's extra cost goes."""
    rows = []
    for label, df in runs.items():
        m = _metered(df)
        if m.empty:
            continue
        totals = {}
        for u in m["usage"]:
            for step_name, s in (u.get("by_step") or {}).items():
                t = totals.setdefault(step_name, {"calls": 0, "tokens": 0})
                t["calls"] += s["calls"]
                t["tokens"] += s["input_tokens"] + s["output_tokens"]
        all_tokens = sum(t["tokens"] for t in totals.values()) or 1
        for step_name, t in sorted(totals.items(), key=lambda kv: -kv[1]["tokens"]):
            rows.append({"Run": label, "Step": step_name, "Calls": t["calls"],
                         "Tokens / q": round(t["tokens"] / len(m)),
                         "Share of tokens": f"{100 * t['tokens'] / all_tokens:.0f}%"})
    return pd.DataFrame(rows)


def accuracy_chart(chart_data):
    """Grouped bars: one bar per run, side by side within each category."""
    fig = px.bar(chart_data, x="Category", y="Accuracy", color="Run", barmode="group",
                 text="Accuracy", category_orders={"Category": CATEGORY_ORDER},
                 title="Answer accuracy by category (1-5)")
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(yaxis_range=[1, 5.4], height=420, legend_title_text="")
    return fig


def build(run_paths):
    runs = {label: load_run(p) for label, p in run_paths.items()}
    summary = summarise(runs)
    chart_data = summary[summary["Category"] != "overall"][["Run", "Category", "Accuracy"]]
    routes = route_mix(runs)
    costs = cost_table(runs)
    steps = step_breakdown(runs)
    unmetered = [label for label, df in runs.items() if _metered(df).empty]

    with gr.Blocks(title="Pitchwise: Linear vs Agentic") as demo:
        gr.Markdown("# 🏏 Pitchwise Evaluation: Linear vs Agentic")
        gr.Markdown(
            "Saved runs of the same 104-question test set, each with one pinned generation model "
            "and the same LLM judge. Scores are 1-5. Files: "
            + ", ".join(f"`{label}` = `{p}`" for label, p in run_paths.items())
        )
        gr.Plot(accuracy_chart(chart_data))
        gr.Markdown("## Summary")
        gr.Dataframe(summary, interactive=False, wrap=True)
        if not routes.empty:
            gr.Markdown("## How the agentic engine routed questions")
            gr.Dataframe(routes, interactive=False, wrap=True)
        if not costs.empty:
            gr.Markdown("## Cost and tokens (list prices; Pitchwise itself runs on free tiers)")
            gr.Dataframe(costs, interactive=False, wrap=True)
            gr.Markdown("## Where the tokens go, by step")
            gr.Dataframe(steps, interactive=False, wrap=True)
        if unmetered:
            gr.Markdown("*No token usage recorded for: " + ", ".join(unmetered)
                        + " (runs from before token accounting, DECISIONS.md D-012).*")
    return demo


if __name__ == "__main__":
    paths = parse_args(sys.argv[1:])
    if not paths:
        sys.exit("No results files found. Pass LABEL=path arguments.")
    build(paths).launch(inbrowser=True, server_port=PORT, theme=gr.themes.Soft(primary_hue=gr.themes.colors.green))
