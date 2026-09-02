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
*Last updated: September 2026*