# EVAL_RESULTS.md — Pitchwise

A summary of the evaluation work behind Pitchwise, in plain terms. For the
full reasoning behind every decision referenced here, see
[`DECISIONS.md`](./DECISIONS.md).

---

## How this RAG pipeline was validated, and how it improved

Two rounds of validation happened, at two different points in the build,
using two different levels of rigor — worth being upfront about that rather
than implying uniform treatment throughout.

### Round 1 — Chunking & embedding model (small-scale, manual)

Before the full evaluation harness existed, chunking strategy and embedding
model choice were validated with a hand-verified 8-query test set run
against 4 embedding models. This surfaced a real bug: three of four models
failed to answer "What is Don Bradman's exact career Test batting average?"
correctly — not because of model quality, but because the knowledge base's
"Career Statistics" section didn't repeat Bradman's name, making the chunk
ambiguous once embedded independently. Restructuring all 17 documents so
every section self-anchors with its subject's name fixed this across every
model tested.

**Full reasoning:** [`DECISIONS.md` D-001, D-002, D-003](./DECISIONS.md)

### Round 2 — Full evaluation harness (104 questions, systematic)

A proper evaluation harness was then built: **104 test questions** across
four categories, each checked on two independent metrics — **retrieval
quality** (MRR, nDCG, keyword coverage) and **answer quality** (an
independent LLM judge scoring accuracy, completeness, and relevance, 1–5).

| Category | Count | What it tests |
|---|---|---|
| `direct_fact` | 59 | Single, unambiguous facts |
| `spanning` | 20 | Questions needing 2+ facts combined |
| `temporal` | 10 | Recency-sensitive / retirement-status questions |
| `out_of_scope` | 15 | Questions the knowledge base does **not** cover — tests whether the system correctly declines rather than hallucinates |

---

## The headline finding: a real grounding gap, found and closed

Manual testing surfaced a case where Pitchwise confidently answered a
question about cricket nicknames using general model knowledge, not its own
knowledge base — despite an explicit system-prompt instruction to decline
when unsure. Running the full harness confirmed this wasn't a one-off:

| | `out_of_scope` accuracy | Overall accuracy (n=104) |
|---|---|---|
| **Before fix** | **3.3 / 5** | 4.6 / 5 |
| **After fix — Run 1** | **5.0 / 5** | — |
| **After fix — Run 2** | **5.0 / 5** | — |

The system prompt was rewritten from a single soft suggestion ("if you
don't know, say so") into explicit, repeated grounding rules — a required
exact refusal phrase, and an instruction not to name any entity absent from
the retrieved context. The fix was verified **twice, independently** — not
just once — since LLM outputs carry natural run-to-run variance, and a
result that holds twice is real evidence, not a lucky run.

**Full reasoning:** [`DECISIONS.md` C-003](./DECISIONS.md)

---

## An honest side effect, also found by the same two runs

The stricter grounding prompt didn't come free. The `spanning` category
(compound/comparison questions) dipped slightly across the same two runs
(~4.35 → ~4.10). Inspecting the actual failures split them into two
different problems:

- **Genuine retrieval gaps** — some compound questions need more than the
  default 4 retrieved chunks to answer fully; the model now correctly
  declines instead of guessing, which is *correct* behavior given
  incomplete context, but means real answers are being missed.
- **Over-conservative refusal** — a couple of cases where the retriever
  found everything needed, but the model still declined simple combination
  or inference over the retrieved facts. This is a prompt-strictness side
  effect, not a retrieval problem — raising `k` won't fix it.

This is a genuinely useful finding, not a flaw to hide: it shows the
evaluation harness catching a second-order effect of its own first fix, and
gives a concrete, evidence-backed next step rather than a vague "could be
improved."

**Full reasoning, including the next step already built (`compare_k.py`):**
[`DECISIONS.md` D-007](./DECISIONS.md)

---

## v2: linear vs agentic engine (October 2026)

v2 added a LangGraph agentic engine (plan, grade, capped rewrite) beside the
v1 linear pipeline ([`DECISIONS.md` D-009](./DECISIONS.md)). It was measured
on the same 104 questions, in the same week, with one generation model pinned
for every run (qwen3.8-27b on Groq, no fallback) and the same judge
(Gemini 3.5 Flash Lite) ([`DECISIONS.md` D-010](./DECISIONS.md)).

- **L1**: linear engine, one run (the baseline)
- **A1, A2**: agentic engine, two independent runs

### Answer accuracy by category (1-5)

| Category | n | L1 linear | A1 agentic | A2 agentic |
|---|---|---|---|---|
| `direct_fact` | 59 | 5.0 | 5.0 | 5.0 |
| `spanning` | 20 | **4.2** | **5.0** | **5.0** |
| `temporal` | 10 | 4.6 | 4.6 | 4.5 |
| `out_of_scope` | 15 | 4.7 | 5.0 | 4.9 |
| Overall | 104 | 4.8 | 5.0 | 4.9 |

The pass rule was fixed before the runs: `spanning` at least 0.3 above the
baseline in **both** agentic runs, `direct_fact` not below 4.9, at most one
`out_of_scope` failure per run, and `temporal` within 0.3 of the baseline.
Both runs pass.

### Where the spanning gain came from

L1 failed 4 of the 20 `spanning` questions (each scored 1); A1 and A2 failed
none. The eval's retrieval metrics always score the v1 retriever, so
`evaluation/mechanism.py` re-ran those 4 questions through the agentic engine
and measured keyword coverage on the context the answer was actually
generated from:

| Q | Question (short) | v1 coverage | Agentic coverage | Why it was fixed |
|---|---|---|---|---|
| 66 | The 7,000-run / 200-wicket players | 0% | 100% | Planner's standalone rewrite retrieved the right chunk |
| 67 | World Cup vs T20 World Cup founding years | 50% | 100% | Split into two sub-queries |
| 76 | Ashes vs World Test Championship format | 50% | 100% | Split into two sub-queries |
| 79 | The "Fab Four" nations | 100% | 100% | Not the engine: identical context; the model answered instead of refusing |

**3 of the 4 fixes come from retrieval**, which is what the engine was built
to change. The fourth (Q79) is an over-cautious refusal in the generation
step; the same context was refused again by a different model in the
mechanism check, so it is counted as model variance, not as an engine gain.

### What it costs

| | Linear | Agentic (A2) |
|---|---|---|
| LLM calls per question, average | 1 | 2.2 |
| LLM calls per `spanning` question | 1 | 3.2 |
| `spanning` questions routed to `multi` | n/a | 16 of 20 |
| Questions that used a rewrite | n/a | 2 of 104 |

Simple questions take 2 calls (plan + generate) and skip grading, which is
why `direct_fact` stayed at 5.0.

### Remaining failures, reported as found

- **Q83 (temporal), failed in all three runs.** "Has Ben Stokes retired from
  any international format?" The answer is in the knowledge base, but under
  a `## Personal` heading that never mentions retirement, so neither engine
  retrieves it. The agentic engine routes it as `simple`, which skips
  grading, so it never notices the gap. This is a knowledge-base structure
  issue of the kind D-003 fixed elsewhere, and is part of v3.
- **Q90 (out_of_scope), partial in A2 (scored 3).** Asked about "Master
  Blaster, Little Master, Chase Master", A2 attributed "Chase Master" to
  Tendulkar; the knowledge base says Kohli. A1 answered it correctly from
  context, and L1 answered it from general knowledge. Part of the question
  *is* in the knowledge base, so it is also mislabelled as out of scope; the
  test set will be corrected in v3.
- **Q78 was not a failure.** D-007 had filed the powerplay question as an
  over-cautious refusal. The agentic grader showed the ODI powerplay length
  is simply not in the knowledge base, so refusing was correct
  ([`DECISIONS.md` C-004](./DECISIONS.md)). In A2 the engine answered the
  part it could (T20I: 6 overs) and said the ODI figure was not available.

### Reproduce

```bash
uv run python -m evaluation.eval --pin-model --results evaluation/results_linear_oct.jsonl
uv run python -m evaluation.eval --engine agentic --pin-model --results evaluation/results_agentic_r1.jsonl
uv run python -m evaluation.eval --engine agentic --pin-model --results evaluation/results_agentic_r2.jsonl
uv run python -m evaluation.mechanism            # final-context coverage for Q66, Q67, Q76, Q79
uv run python -m evaluation.compare_runs         # dashboard comparing the saved runs, no LLM calls
```

---

## Screenshots

**Chat interface** (`app.py`) — a live conversation, showing the retrieved
source chunks alongside the answer:

![Pitchwise chat interface](./screenshots/chat-interface.png)

**Evaluation dashboard** (`evaluator.py`) — color-coded retrieval and answer
metrics, broken down by category:

![Pitchwise evaluation dashboard](./screenshots/evaluation-dashboard.png)

---

## Try it yourself

```bash
uv run python app.py                    # chat interface
uv run python evaluator.py               # visual evaluation dashboard
uv run python -m evaluation.eval         # full 104-question console run
```

---
*Author: Somesh Kant Tiwari*
*Last updated: October 2026*