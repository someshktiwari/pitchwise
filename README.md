# 🏏 Pitchwise

A Retrieval-Augmented Generation (RAG) assistant that answers cricket
questions — players, formats, tournaments, and the laws of the game —
grounded strictly in a curated knowledge base, not general model knowledge.

Built end-to-end: document ingestion, chunking, embeddings, vector search,
multi-provider LLM generation with automatic fallback, a chat interface,
and a 104-question evaluation harness that found and fixed a real
grounding gap. Every architectural decision — and every mistake found and
corrected along the way — is dated and reasoned in
[`DECISIONS.md`](./DECISIONS.md).

---

## What it does

Ask a question. Pitchwise retrieves the relevant chunks from its own
knowledge base, generates an answer strictly grounded in that retrieved
context, and shows you exactly which chunks it used — so nothing is a black
box.

![Pitchwise chat interface](./screenshots/chat-interface.png)

If the knowledge base doesn't cover something, Pitchwise says so instead of
guessing — a behavior that was tested for, found to be failing, fixed, and
verified with real data. Details in [`EVAL_RESULTS.md`](./EVAL_RESULTS.md).

---

## Architecture

```
knowledge-base/          17 curated markdown documents
       │
       ▼
   ingest.py              chunk (header-aware) → embed (local, free) → vector store
       │
       ▼
   answer.py               retrieve → build prompt → generate (multi-provider, with fallback)
       │
       ├──▶ app.py              Gradio chat interface
       └──▶ evaluation/         104-question eval harness (retrieval + LLM-judge)
```

**Multi-provider by design:** Pitchwise can generate answers via Gemini,
Groq, or OpenRouter — chosen after directly measuring latency across 7
model/provider combinations (Groq came out fastest by a wide margin). If
the selected provider fails or hits a rate limit, it silently falls back to
the next-fastest option rather than surfacing an error. Full reasoning:
[`DECISIONS.md` D-006, D-008](./DECISIONS.md).

---

## Evaluation

A 104-question harness across four categories (direct facts, multi-fact
compound questions, temporal/recency questions, and out-of-scope questions
designed to test grounding) — each scored on retrieval quality and answer
quality independently.

The headline result: a real hallucination gap was found through targeted
testing (`out_of_scope` accuracy: **3.3/5**), fixed with a rewritten system
prompt, and the fix was verified **twice, independently** (**5.0/5 in both
runs**).

![Pitchwise evaluation dashboard](./screenshots/evaluation-dashboard.png)

Full results, methodology, and an honestly-reported side effect of the fix:
**[`EVAL_RESULTS.md`](./EVAL_RESULTS.md)**

---

## Running it

```bash
uv sync                                  # install dependencies

uv run python app.py                     # chat interface
uv run python evaluator.py                # visual evaluation dashboard
uv run python -m evaluation.eval          # full 104-question console run
```

Requires free API keys for Gemini, Groq, and OpenRouter in a `.env` file
(see `.env.example`). All embeddings run locally — no key needed for
ingestion.

---

## Project structure

```
pitchwise/
├── ingest.py              document loading, chunking, embedding, vector store
├── answer.py               retrieval + multi-provider generation
├── app.py                   Gradio chat interface
├── evaluator.py              visual evaluation dashboard
├── knowledge-base/           17 curated markdown documents
├── evaluation/
│   ├── test.py                test question schema + loader
│   ├── eval.py                 retrieval + LLM-judge evaluation logic
│   ├── compare_k.py             retriever k-value comparison tooling
│   └── tests.jsonl               104 test questions
├── DECISIONS.md               every architecture decision, dated and reasoned
└── EVAL_RESULTS.md            evaluation methodology and results, in plain terms
```

---

## Why this exists

Built as a hands-on project to demonstrate practical RAG engineering:
chunking strategy trade-offs, embedding model selection under real
constraints, multi-provider LLM architecture, and — the part most RAG demos
skip — a real evaluation harness that found an actual bug rather than
existing for show.

## Future plans

Pitchwise is presented here as a working, evaluated project — not a
finished product. Concrete next steps, in rough priority order:

- **FastAPI service layer** — expose retrieval and generation as a proper
  REST API (`answer.py`'s functions are already decoupled from the Gradio
  UI, so this is additive, not a rewrite), enabling programmatic access and
  a cleaner separation between backend and interface.
- **A stronger knowledge base** — expand beyond the current 17 curated
  documents with more players, tournaments, and historical depth, and use
  the evaluation harness to verify retrieval quality holds as the knowledge
  base grows (see [`DECISIONS.md`](./DECISIONS.md) D-001 and D-005 for why
  this matters — chunk count and re-embedding cost both scale with content
  size).
- **Live deployment** — the architecture was deliberately built
  deployment-ready from the start (no disk persistence, no external
  embedding API dependency — see [`DECISIONS.md`](./DECISIONS.md) D-002,
  D-005), specifically so this is a configuration step rather than a
  rework. Hugging Face Spaces is the leading candidate given the free tier
  and native Gradio support.
- **Retriever `k` tuning** — `evaluation/compare_k.py` is already built to
  test different retriever settings against the specific gap the
  evaluation harness surfaced (see [`EVAL_RESULTS.md`](./EVAL_RESULTS.md)
  and [`DECISIONS.md`](./DECISIONS.md) D-007) — running it to completion
  and applying the result is the most immediate of these next steps.

---
*Author: Somesh Kant Tiwari*