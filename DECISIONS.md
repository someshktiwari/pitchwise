# DECISIONS.md — Pitchwise

Every architectural and implementation decision made in building Pitchwise,
a RAG-based cricket knowledge assistant. Each entry covers the decision, the
alternatives considered, the reasoning, and the trade-offs explicitly accepted.

---

## Table of Contents

**Part I — Architectural Decisions**
D-001 through D-008 — ingestion and generation pipeline decisions.
D-009 through D-011 — v2: agentic engine, evaluation changes, service layer.
D-012 through D-013 — v3: token and cost accounting, request tracing.
D-014 through D-017 — v3: knowledge base expansion, saved index, evaluation model choice, k=8 control.
D-018 — measurement limits kept fixed for the v3 runs, and what changes after them.

**Part II — Corrections**
Every bug fixed, every incorrect assumption corrected, every change made during
the build — with the reasoning behind each correction.

# Part I — Architectural Decisions

---

## D-001 · Chunking Strategy: MarkdownHeaderTextSplitter over RecursiveCharacterTextSplitter

**Decision:** documents are split using `MarkdownHeaderTextSplitter` (splitting
on `#`/`##` header boundaries), not `RecursiveCharacterTextSplitter` (splitting
by character count with paragraph/sentence fallback).

**Alternatives considered:**
- `RecursiveCharacterTextSplitter` — the default approach in most RAG
  tutorials; splits by character count (1000 chars, 200 overlap), falling
  back through paragraph → sentence → word boundaries
- `CharacterTextSplitter` — simpler single-separator splitting, rejected
  early as strictly weaker than the recursive version with no offsetting
  benefit

**Why this:**
The knowledge base is structured markdown with meaningful headers — each `##`
section (e.g. "Career Statistics," "Format Rules") represents one coherent
fact-category. Tested both directly: `RecursiveCharacterTextSplitter` produced
38 chunks (avg 780 characters, min 222, max 984); `MarkdownHeaderTextSplitter`
produced 81 chunks (avg 336 characters, min 49, max 705). The markdown-based
chunks align exactly to document sections, so a query about "Test cricket's
format rules" retrieves a chunk that is *only* about format rules, rather than
a chunk that blends format rules with adjacent unrelated content because a
1000-character cutoff landed mid-section.

**Trade-offs accepted:**
- More than double the chunk count (81 vs 38) — at this knowledge base's
  scale (17 documents), the cost is negligible; would need reassessment at a
  much larger scale
- Smaller chunks carry less standalone context — a question spanning two
  subsections may need the retriever to return multiple chunks rather than
  one, which is normal RAG behavior but worth naming as the actual cost of
  choosing focus over breadth per chunk

---

## D-002 · Embedding Model: all-MiniLM-L6-v2, Chosen on Architecture, Not Just Benchmarks

**Decision:** `all-MiniLM-L6-v2` (HuggingFace, local, free, 384 dimensions) is
the embedding model used in the ingestion pipeline.

**Alternatives considered and tested directly** (all four run against the same
81 chunks, compared on load/embed time and retrieval quality against 8
hand-verified test queries):
- `all-mpnet-base-v2` — larger HuggingFace model, 768 dimensions
- `BAAI/bge-small-en-v1.5` — competitive small HuggingFace model, 384
  dimensions
- `gemini-embedding-001` — Google's hosted embedding API, 3072 dimensions,
  tops the MTEB multilingual leaderboard

**Why this:**
Timing was a non-factor once models were cached locally (9.6s–11.1s across
all three HuggingFace options for 81 chunks — noise-level difference). The
real decision came down to retrieval quality vs. architectural fit.

An initial 8-query retrieval test showed a real quality gap: all three local
HuggingFace models failed to retrieve Don Bradman's exact career average
(99.94) from his "Career Statistics" section, instead retrieving his
unrelated biographical "Career Overview" section. Only `gemini-embedding-001`
retrieved the correct chunk. Investigating this (see D-003) revealed the
cause was not embedding model capability but chunk self-containment — the
stats section didn't repeat Bradman's name or any anchoring context, making
its embedding ambiguous. After restructuring the knowledge base to fix this
(D-003), all four models — including all three free local ones — retrieved
the correct chunk.

With that gap closed, `all-MiniLM-L6-v2` performs at parity with
`gemini-embedding-001` on 6 of 7 hard test queries, differing only on a
negation-style query that no model handled reliably (see D-004). Given parity
in retrieval quality, the deciding factor became architecture:
`all-MiniLM-L6-v2` runs locally with no API key, no network dependency, and
no rate limits — directly relevant given Pitchwise's ingestion pipeline is
designed to rebuild the vector store fresh on every app startup, a
deliberate choice (since superseded by D-015, which saves the index) made so the project can be deployed to a free-tier host
(e.g. Hugging Face Spaces) later without any rework, even though it
currently runs locally and is presented via GitHub. A hosted embedding API
would make every single app boot dependent on an external network call
succeeding within a free-tier quota — a real reliability risk this
architecture doesn't need to accept once quality is no longer the
deciding factor.

**Trade-offs accepted:**
- Forgoes `gemini-embedding-001`'s deeper multilingual/semantic capability,
  which is unneeded for a small, English-only, well-structured knowledge base
- The quality-parity finding is specific to this knowledge base's size and
  structure and shouldn't be assumed to generalize to a larger or
  less-curated document set

Note: temperature=0 is set on ChatGoogleGenerativeAI but gemini-3.6-flash ignores it (fixed sampling defaults) — kept in code to document intent, no functional effect.

---

## D-003 · Content Restructuring: Every Section Header and Opening Line Repeats the Subject Name

**Decision:** every `##` section across all 17 knowledge base documents was
rewritten so the section header and its first sentence explicitly name the
subject (player, format, or tournament), rather than relying on the
document's title or a preceding section for context.

**Example — before:**
```
# Don Bradman
## Career Overview
Don Bradman is universally regarded as...
## Career Statistics (career totals, retired 1948)
- Tests: 6,996 runs in 52 matches, batting average 99.94
```
**Example — after:**
```
# Don Bradman
## Career Overview
Don Bradman is universally regarded as...
## Don Bradman's Career Statistics (career totals, retired 1948)
- Tests: Don Bradman scored 6,996 runs in 52 matches, batting average 99.94
```

**Why this:**
Root-caused via the embedding model comparison in D-002. Because chunking
splits documents at header boundaries (D-001), a chunk containing only
"Career Statistics... Tests: 6,996 runs..." carries no explicit link back to
"Don Bradman" in its own text — the connection lived only in the sibling
"Career Overview" chunk or the document's H1 title, neither of which travels
with the stats chunk once it's embedded and retrieved independently. Three of
four tested embedding models treated this chunk as too generic to surface for
a query naming Bradman directly. Verified the fix by re-running the identical
retrieval test after restructuring: all four models, including the three
that previously failed, correctly retrieved the anchored chunk.

**Trade-offs accepted:**
- Minor content redundancy (the subject's name now appears more often than
  strict prose style would otherwise call for) — an explicit trade of
  writing elegance for retrieval robustness, judged worthwhile since chunks
  are consumed by an embedding model and an LLM, not read as continuous
  prose by a human
- This is a manual, per-document fix; a knowledge base that grows
  significantly larger would need this discipline enforced by a content
  template or a linting step rather than applied by hand each time

---

## D-004 · Known Limitation: Negation-Style Queries Are Not Reliably Handled

**Decision:** no fix attempted; documented as an accepted limitation of the
retrieval approach.

**What's actually true:** the query "Which cricket format does *not* have a
follow-on rule?" was tested against all four embedding models. Three of four
retrieved Test cricket's own follow-on section — the format that *does* have
the rule — rather than a correct answer. Only one model retrieved a
not-clearly-better alternative (the Laws of Cricket document, also not a
direct answer).

**Why this happens:** embedding-based retrieval works by semantic
similarity — it finds chunks that are topically close to the query. A
negation ("does NOT have X") is topically *closest* to the exact chunk that
discusses X, since that chunk shares the most vocabulary and concepts with
the query. The embedding has no mechanism to represent logical negation; it
can only represent "this is about follow-on rules," not "this is the absence
of follow-on rules elsewhere."

**Considered and rejected:** restructuring content to include explicit
negative statements (e.g., adding "T20I cricket does not have a follow-on
rule" to the T20I document) — rejected because it doesn't scale; every
format would need explicit negative statements about every rule it lacks,
multiplying content size for marginal, narrow-case benefit.

**Status: unfixed, deliberately.** This is a known, general limitation of
semantic-similarity retrieval, not specific to Pitchwise's implementation,
and is worth stating plainly rather than obscuring behind cherry-picked test
queries.

---

## D-005 · No Local-Only Persistence, Despite the Re-Embedding Cost

> **Superseded by D-015 (October 2026)** once the knowledge base grew past
> what is cheap to re-embed on every start.

**Decision:** the vector store is rebuilt from scratch on every run, in every
environment. The project currently runs locally and is presented via GitHub
rather than deployed — but the design deliberately doesn't rely on disk
persistence, so deploying to a free-tier host later (e.g. Hugging Face
Spaces) requires no architectural changes. No `persist_directory` is used,
even though local development could support disk persistence.

**Alternatives considered:**
- Persist locally to disk during development (faster iteration), rebuild
  fresh only in production — the two environments would diverge in setup,
  but each would be individually reasonable
- Use `gemini-embedding-001` locally (where a persisted store makes an API
  dependency less costly, since it only runs once) and `all-MiniLM-L6-v2` in
  production — combining D-002's production reasoning with faster local
  iteration

**Why rejected:**
Different embedding models produce vectors in different, incomparable
embedding spaces. If local development used one model and production used
another, retrieval behavior tested and tuned locally would not necessarily
match what actually runs in production — the two environments would
effectively be testing two different systems. That inconsistency is a worse
trade than the cost it would save.

**Why the cost is acceptable:** re-embedding all 81 chunks with
`all-MiniLM-L6-v2` takes approximately 10 seconds, measured directly during
the D-002 embedding comparison. At the knowledge base's current size, this
is a trivial one-time cost on every app boot or script run, not a real
performance problem.

**Trade-offs accepted:**
- Every `ingest.py` run, in any environment, always pays the ~10 second
  embedding cost, even during rapid local iteration
- Revisit if the knowledge base grows large enough that 10 seconds becomes
  a meaningful delay (e.g. minutes) — at that point, local-only persistence
  with the same embedding model used in production would be worth adding
  back, since it would no longer create a cross-environment inconsistency

---

## D-006 · Default Provider/Model: Groq (qwen3.8-27b), Chosen from a Full Speed Comparison

**Decision:** `answer.py`'s default is Groq running `qwen/qwen3.8-27b`, not
Gemini, despite Gemini being Pitchwise's primary provider elsewhere
(embeddings-adjacent work).

**What's actually true:** the same question, with identical retrieved
context, was timed against 7 model/provider combinations in a single run:

| Rank | Model | Time |
|---|---|---|
| 1 | Groq — qwen3.8-27b | 0.4s |
| 2 | Groq — gpt-oss-20b | 0.6s |
| 3 | Groq — gpt-oss-120b | 0.9s |
| 4 | Gemini — 3.5-flash-lite | 1.2s |
| 5 | OpenRouter — minimax-m3:free | 4.3s |
| 6 | OpenRouter — glm-5.2:free | 4.5s |
| 7 | Gemini — 3.6-flash | 5.1s |

All seven produced a correct, grounded answer (99.94) using the same
retrieved context — this is a speed comparison among already-correct
answers, not a correctness filter.

**Found by:** an initial two-point comparison (Gemini 3.6-flash vs. Groq
gpt-oss-120b) showed a large gap — 13.8s vs. 1.6s in that run. Rather than
generalizing from two data points, the comparison was widened to 7
combinations across all three providers, run together in one pass for a fair
comparison against identical context. Gemini 3.6-flash's own timing varied
between runs (13.8s in the first isolated test, 5.1s in the combined run),
which is itself informative: single-sample latency numbers for hosted LLM
APIs are noisy and shouldn't be treated as stable benchmarks.

**Why Groq, and why qwen3.8-27b specifically:** Groq's three tested models
occupied the top three speed positions outright, consistent with Groq's
infrastructure being purpose-built for low-latency inference. Among Groq's
own models, qwen3.8-27b was fastest in this run; gpt-oss-120b and
gpt-oss-20b remain reasonable fallbacks within the same provider if
qwen3.8-27b's free-tier limits are hit.

**Why not Gemini 3.5-flash-lite, given it was competitive (1.2s):** close
enough that this is a soft preference, not a hard rejection — Flash-Lite
remains a reasonable secondary option, and the finding that it is ~4x faster
than 3.6-flash is independently useful (see note below). Groq was chosen as
default primarily because its three models clustered fastest and most
consistently, and secondarily to keep Gemini's role focused on
embeddings-adjacent work and multi-provider comparison rather than being
overloaded as both.

**Why not OpenRouter as default:** both tested free models were
meaningfully slower (4.3–4.5s) than every Groq option and Gemini's Lite
variant, plausibly due to OpenRouter's request-routing/aggregation layer
across free-tier models. OpenRouter remains wired in as a selectable
provider for its model variety, not removed.

**Related finding, not a separate decision:** `gemini-3.6-flash`'s slowness
relative to `gemini-3.5-flash-lite` (5.1s vs. 1.2s, same provider) supports
the theory that 3.6-flash performs more internal reasoning by default,
overhead unnecessary for a straightforward factual RAG lookup. Worth knowing
independently of the Groq-vs-Gemini choice: if Gemini is ever selected
explicitly (e.g. for its stronger multi-step reasoning on a harder query),
Flash-Lite is the better default variant, not 3.6-flash.

**Trade-offs accepted:**
- Groq's free-tier rate limits are tighter in absolute terms than Gemini's
  for some models; acceptable at demo-level traffic, worth reassessing if
  usage grows
- Latency numbers here are single-run measurements, not averaged over
  multiple trials — sufficient to establish clear rank ordering (the gaps
  are large) but not precise enough to claim, e.g., "exactly 8x faster"
  with confidence
- Gemini and OpenRouter remain fully wired in as selectable options, so the
  multi-provider architecture and comparison capability built earlier in the
  project is preserved — this is a default change, not a removal

Note: conversation history is stored in one provider-agnostic format and converted fresh per call to whichever model is selected — switching models mid-conversation works, but context window limits differ across free-tier models and could theoretically be hit on a very long conversation (not a concern at Pitchwise's current knowledge-base scale).

---

## D-007 · Retriever k=4, Stated Explicitly, Deferred to the Evaluation Harness

**Decision:** the retriever returns 4 chunks per question (`search_kwargs={"k": 4}`),
matching Chroma's own default — but stated explicitly in code rather than
left as an unstated default.

**Alternatives considered:** none tested directly. A higher k (e.g. 6-8)
would surface more context per question at the cost of a longer, more
diluted prompt; a lower k (e.g. 2) would tighten focus at the risk of
missing relevant chunks for questions spanning multiple document sections.

**Why not tested now:** unlike chunking strategy (D-001) and embedding model
(D-002), which were evaluated against hand-verified test queries with a
clear right answer, tuning k has no obvious verification method without a
proper accuracy metric across many questions — eyeballing a handful of
queries risks the same weak-methodology trap the embedding comparison
deliberately avoided by testing 8 queries across 4 models rather than
trusting one or two. A meaningful k comparison needs the same kind of
ground-truth evaluation set that the evaluation harness (see Part I,
evaluation-related entries) is intended to provide.

**Status: deferred, not decided.** k=4 is a reasonable, stated default, not
a tuned one. Revisit once the evaluation harness exists and retrieval
quality can be measured systematically rather than eyeballed.

**Update, post C-003 (system prompt fix):** the evaluation harness now
exists and was run twice after the prompt fix, giving real evidence to
revisit this decision with. The `spanning` category (compound/comparison
questions, the type most likely to need more than 4 chunks to answer fully)
showed the only meaningful score movement across the two runs. Inspecting
the failures revealed **two distinct failure types**, not one:

- **Type 1 — genuine retrieval gaps** (e.g. "which three players have
  7,000+ runs and 200+ wickets," "compare the founding years of both World
  Cups"): `keyword_coverage` measured 0–50% on these — the retriever
  genuinely didn't surface enough chunks to answer fully, and the (now
  correctly grounded) model declines rather than guess. This is the
  scenario this entry originally anticipated, and raising k is the direct,
  likely fix.
- **Type 2 — over-conservative refusal despite full retrieval** (e.g. "how
  many overs separate ODI and T20I powerplays," "which nations make up the
  Fab Four"; the powerplay case was later found to be a correct refusal,
  see C-004): `keyword_coverage` measured 100% on these — the retriever
  found everything needed, but the stricter grounding prompt (C-003) still
  caused the model to decline simple arithmetic on two retrieved numbers,
  or to withhold a well-known fact (a player's nationality) because the
  specific retrieved chunk didn't restate it explicitly. **Raising k will
  not fix this** — it's a prompt-strictness side effect of C-003, not a
  retrieval problem, and needs a separate, more surgical instruction (e.g.
  explicitly permitting combination/inference over multiple context items,
  while still prohibiting facts absent from the context entirely).

`evaluation/compare_k.py` was built to test k=4/6/8 against the `spanning`
category directly and separate these two effects with real data, rather
than assuming which k value is "enough." Not yet run to completion — this
remains the concrete next step, not a completed decision.

**Update (October 2026):** the k question was answered by D-017 instead:
a k=8 run through the main evaluation harness on the v3 `spanning`
questions. `compare_k.py` is kept but was not used for it.

---

## D-008 · Silent Automatic Fallback Across Providers

**Decision:** if the selected model/provider fails (rate limit, API error,
or any other exception), `answer_question()` silently retries with the next
option in `FALLBACK_ORDER` — fastest-first, per D-006's measured ranking —
until one succeeds or every option has failed.

**Alternatives considered:**
- Surface the error directly to the user, let them choose whether to retry
  or pick a different model manually

**Why silent fallback:** Pitchwise's primary near-term purpose is being
demoed live, often by someone (an interviewer) encountering it cold via a
resume link. A visible error on a rate-limited free-tier model reads as the
project being broken, even though the underlying system has multiple
working alternatives available immediately. Silently recovering preserves a
working, responsive experience for that audience.

**Trade-offs accepted:**
- The user has no visibility into which model actually answered a given
  question unless they check logs (since v2, the trace shown under each
  answer names the model that answered, D-009) — acceptable for a demo-facing product at
  this scale, but would need to change for a context where users need to
  know which model handled their request (e.g. for cost attribution or
  reproducibility in a production setting)
- If every provider fails simultaneously (e.g. all API keys are invalid),
  the eventual error message concatenates only the *last* provider's error,
  which may not clearly explain that every option was actually tried —
  acceptable for now, worth revisiting if debugging a full outage ever
  becomes difficult in practice

---

## D-009 · v2 Agentic Engine: an Explicit LangGraph Graph, Not a ReAct Agent

**Date:** October 2026

**Decision:** add a second engine, `engine="agentic"` (`graph.py`), built as an
explicit LangGraph `StateGraph`: plan → retrieve → (grade → rewrite loop) →
generate. The v1 pipeline stays as `engine="linear"`, and both engines share
the same retrieval function and the same grounded generation prompt.

**Why:** the evaluation localised the weakness. `spanning` (compound)
questions scored 4.1/5 (September v1 run) against 5.0/5 for direct facts, and D-007's failure
analysis split those failures into retrieval gaps (keyword coverage 0–50%)
and over-conservative refusals. Query decomposition plus a sufficiency check
targets the first type directly: each needed fact gets its own retrieval, and
a missing fact triggers a targeted re-retrieval.

**Alternatives considered:**
- *Raise k (e.g. k=8)* — the cheapest fix for retrieval gaps, with no extra
  LLM calls. Planned as a control for the v2 evaluation (if k=8 matches the
  agentic engine, the graph is not earning its cost). **Not run:** the free
  Groq quota covered the three planned runs (L1, A1, A2) but not a fourth.
  It remains the open comparison: without it, the results show the agentic
  engine beats k=4, not that it beats a simpler fix.
- *Prebuilt ReAct agent with a retriever tool* (`create_react_agent`) — the
  model decides when to retrieve and when to stop. Rejected: retrieval could
  be skipped entirely (breaking the grounding guarantee from C-003), the
  number of calls is unbounded, and branches are hard to test in isolation.

**Design rules:**
- *Always retrieve, original question first.* The agentic context is a
  superset of what v1 would retrieve (before the 10-chunk cap).
- *Simple questions skip grading.* They cost 2 LLM calls (plan, generate),
  which protects the categories v1 already answers at 5.0/5.
- *Fail open.* Unparseable planner output → treat as simple with the original
  question; unparseable grader output → treat as sufficient. Either way the
  engine degrades to v1 behaviour rather than erroring or guessing.
- *Bounded cost.* At most 2 rewrites (worst case 7 LLM calls); LangGraph's
  recursion limit (20) is only a safety net above that cap.
- *One grading call over all chunks*, not one per chunk, to keep cost flat.
- *Temperature 0 for the routing calls* (plan, grade, rewrite) for repeatable
  decisions; the generation call is left exactly as in v1.

**State design:** `trace` and `sub_queries` use an `operator.add` reducer
(they accumulate across steps); `docs` deliberately does not — `retrieve`
rebuilds the full merged set on each pass, so overwriting avoids the kind of
silent duplication found in C-001.

**Trade-offs accepted:**
- Every question now costs at least 2 LLM calls instead of 1, and multi
  questions up to 7, so latency rises. Measured per category in the v2 eval.
- The planner can misroute. The route is recorded for every eval question,
  so misroutes are visible rather than hidden.

**Status (October 2026):** measured. On the 104-question harness, `spanning`
went from 4.2/5 (October linear baseline) to 5.0/5 in both agentic runs, with
`direct_fact` unchanged at 5.0/5; 3 of the 4 fixed failures were traced to
retrieval. Full results: [`EVAL_RESULTS.md`](./EVAL_RESULTS.md). The routing
calls in those runs were not pinned to one model; see C-005. The k=8
control was run in v3, on the expanded knowledge base (D-017).

---

## D-010 · Evaluation Runs Can Pin One Model and Use Their Own Results File

**Decision:** `evaluation/eval.py` gains `--engine`, `--results` and
`--pin-model`. With `--pin-model`, provider fallback (D-008) is disabled for
the run, and every result row records which model answered. (As first built,
this applied only to the answer step, not to the agentic engine's plan,
grade and rewrite calls; corrected in C-005, and rows now also record the
routing model.)

**Why:** silent fallback is right for a demo but wrong for measurement. If the
default provider rate-limits mid-run and another model answers, the run
compares two models instead of two engines. A separate results file per run
prevents two runs from being merged by the resume logic (C-002).

**Trade-offs accepted:** a pinned run is more likely to hit rate limits; the
existing retry logic (C-002) and resume make that a delay, not a lost run.

---

## D-011 · Service Layer: FastAPI + Gradio in One Container

**Decision:** `api.py` serves `POST /ask` and `GET /health` with Pydantic
request/response models, and mounts the Gradio UI at `/` on the same app. The
`Dockerfile` runs it with uvicorn on port 7860, the port Hugging Face Docker
Spaces expect.

**Choices inside it:**
- *Plain `def` endpoints, not `async def`.* The LLM SDK calls block; FastAPI
  runs `def` endpoints in a thread pool, whereas blocking inside `async def`
  would stall the event loop for every request.
- *One vector store for API and UI*, loaded once at startup (built on every
  start under D-005; loaded from the saved index since D-015).
- *503 when every provider fails* (D-008); 400 for an unknown model; 422 from
  Pydantic for invalid input; questions capped at 500 characters to protect
  free-tier quotas on a public demo. *(October 2026: the question cap could
  be bypassed through the conversation history, which had no limit. The
  API now rejects more than 50 history messages or a message over 4,000
  characters, and both engines send only the last 8 messages, each cut to
  2,000 characters, to the model, which covers the chat UI too.)*
- *Keys only at run time* (`--env-file` / Space secrets), never in the image.
- *The embedding model is baked into the image* so a cold start doesn't
  download it.

**Trade-offs accepted:** `sentence-transformers` pulls a full PyTorch build,
so the image is large (several GB including CUDA libraries the CPU-only
deployment doesn't use). A CPU-only PyTorch install would cut that
substantially; deferred until the deployment is live.

---

## D-012 · Token and Cost Accounting on Every LLM Call

**Date:** October 2026

**Decision:** every LLM call records its input and output tokens, latency and
list-price cost (`usage.py`). Each answer's trace carries the totals, a
per-step breakdown (plan, grade, rewrite, generate) and one record per call,
including failed attempts the fallback chain moved past. Evaluation rows
store the answering usage and the judge's usage separately, and
`evaluation/compare_runs.py` reports tokens and cost per question, per 1,000
questions, per month at 10,000 questions a day, and per correct answer.

**Why:** v2 could only say the agentic engine makes "2.2 LLM calls per
question against 1". Calls are a poor unit of cost: a grading call reads
every retrieved chunk, a planning call reads only the question, and output
tokens cost several times more than input tokens. Tokens times price is
what a team would actually pay, and cost per *correct* answer stops a cheap
engine that answers badly from looking cheap.

**How it works:**
- *Tokens come from the provider's response* (`usage.prompt_tokens` /
  `completion_tokens` for Groq and OpenRouter, `usage_metadata` for Gemini),
  not from a local tokenizer, so the counts are what each provider bills.
  If a provider omits them, the record says `tokens_reported: false` rather
  than guessing.
- *A meter per request, held in a `ContextVar`.* Calls deep inside the
  LangGraph nodes record into the meter of the request that started them,
  without a meter object threaded through every function; concurrent API
  requests each get their own. A `step` context labels each call.
- *Prices are list prices* (`usage.PRICES`, with the date they were
  checked). Pitchwise runs on free tiers, so actual spend is $0; the list
  price answers "what would this cost on a paid plan?". A model without a
  price shows cost as unknown (`None`), never as $0.
- *Failed attempts cost nothing but are counted*, so a run that leaned on
  the fallback chain is visible in its usage. In a pinned evaluation run
  there is no fallback: a rate-limited call fails the question, and the
  harness retries it from scratch, so the stored row shows only the
  attempt that succeeded. Those failed calls are visible in Langfuse
  (D-013) as generations at level ERROR.

**Trade-offs accepted:** results rows are larger (one record per call).
Runs made before this change have no usage, and the dashboard says so
instead of showing zeros. Prices change; the table is dated and must be
re-checked before quoting projections.

---

## D-013 · Optional Request Tracing with Langfuse

**Date:** October 2026

**Decision:** when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set,
each question becomes one Langfuse trace: a span for the request, a span per
graph node, a retriever span per search (query and chunks returned) and a
generation per LLM call with its prompt, output, model, tokens and cost.
Traces are tagged with the engine; evaluation runs group their traces under
a session named after the results file. In an evaluation run, the search
that computes the retrieval metrics and the judge's call run outside the
answer, so each appears as its own small trace beside the answer trace;
the scores are attached to the answer trace. Without the keys, `observability.py`
makes every hook a no-op and never imports Langfuse.

**Why Langfuse:** an open-source, OpenTelemetry-based tracing tool built for
LLM applications, with a free cloud tier and a self-hosted option. It shows
the per-step tokens, cost and latency of a single request in one view,
which is what debugging the agentic engine needs (why did this question
take 7 calls?), and it is a tool teams already use, unlike a home-made
dashboard.

**Scores on traces:** evaluation runs attach the judge's accuracy,
completeness and relevance (with the judge's feedback as a comment) and the
keyword coverage to the trace that produced each answer, using the trace id
recorded in the answer's trace dict. In Langfuse that turns "which answers
scored 2 or less, and what did the engine do?" into a filter and a click.

**Why optional:** the offline tests, CI and anyone cloning the repo must
work without another account. Accounting (D-012) does not depend on
tracing: token and cost numbers are always recorded in the trace and in
evaluation rows. `PITCHWISE_TRACING=off` forces tracing off even with keys
present; the test suite sets it so tests never send traces.

**Verified offline:** with an in-memory span exporter, one agentic question
produced one trace whose spans nest as request → node → generation, with
model, token counts and cost on each generation.

**Trade-offs accepted:** prompts and retrieved context are sent to Langfuse
when tracing is on, which is fine for a public cricket knowledge base but
would need masking for private data. Spans are exported in the background;
batch jobs and the API flush on exit.

---

## D-014 · Knowledge Base Expansion: Curated Core plus Wikipedia Articles

**Date:** October 2026

**Decision:** keep the 17 hand-written documents as a curated core and add
about 130 Wikipedia articles (players, national teams, tournaments, grounds,
rules and the game), fetched and converted by `scripts/build_kb.py` from the
list in `knowledge-base/wikipedia-sources.txt`, into
`knowledge-base/wikipedia/<category>/`. A new test set,
`evaluation/tests_v3.jsonl`, covers the new content; the 104-question v2 set
stays frozen so the v2 results remain reproducible.

**Why:** 17 documents and 81 chunks made retrieval easy: few near-duplicate
chunks compete for the top 4 places. A knowledge base with thousands of
chunks tests whether retrieval, the grounding prompt and the agentic engine
still hold up when the right chunk has many plausible neighbours. That is
the condition a real deployment faces.

**How the conversion keeps D-001 to D-003 true at scale:**
- *Headings name their subject* ("## Brian Lara: Early life"), and every
  Wikipedia chunk starts with its heading, because the splitter removes
  headings from chunk text and article bodies rarely repeat the name. This
  is the D-003 fix (the Bradman "Career Statistics" bug) done by code
  instead of by hand.
- *Long sections are split again.* all-MiniLM-L6-v2 reads at most 256 word
  pieces (about 1,000 characters) and ignores the rest, so Wikipedia
  sections longer than 1,000 characters are split into overlapping pieces
  (150 characters), each keeping its heading. Every curated section is
  shorter than that (the longest is 708 characters), so curated chunks are
  split on headings only, exactly as before.
- *Reference sections are dropped* (References, External links, See also and
  similar), and each article is capped at 20,000 characters, cut only at a
  section boundary, so one very long article can't dominate the index.
- *No subject exists twice.* The script refuses any article whose resolved
  title is a curated subject, so the knowledge base never holds two versions
  of the same facts.

**What the first build produced (October 2026):** 132 of the 133 listed
articles (one title, "James Anderson", was a disambiguation page and is now
"James Anderson (cricketer)"), so 149 documents and 3,077 chunks: 81 curated
and 2,996 from Wikipedia, against 81 in v2. With that article added and the
clean-ups below, the knowledge base used by the v3 runs is 150 documents and
3,106 chunks (81 curated, 3,025 from Wikipedia). Two clean-ups came from reading
the output rather than trusting it:
- *Table-only sections were dropped.* The plain-text extract leaves out
  tables, so some sections became a heading and nothing else ("**Finals.**",
  "Last updated 25 October 2025."). 34 such chunks would have competed for
  retrieval with no facts in them; `ingest.py` now drops any Wikipedia chunk
  with under 40 characters of real text.
- *Markup debris was removed.* Two articles began with URL-encoded template
  links, and several kept citation markers ("[62]") or footnote lines.
  `build_kb.clean_line` removes them (four-digit years in brackets, such as
  "[2003]", are kept), and the existing files were cleaned with the same
  function; a word-level diff against the originals confirmed only debris
  was removed.

**The v3 test set** (`evaluation/tests_v3.jsonl`, 177 questions, built and
checked by `evaluation/build_tests_v3.py`): 114 direct_fact, 32 spanning,
16 temporal and 15 out_of_scope (after C-009). It keeps the v2 questions except where the
new knowledge base changed the right answer: seven out-of-scope questions
became answerable and were relabelled or reworded, one was dropped (the
"fastest T20I century", where the only claim in the knowledge base is a 2017
record that no longer stands), and the Ben Stokes questions follow C-007.
New out-of-scope questions name a subject the knowledge base *mentions*
(Nathan Lyon, Lasith Malinga, Major League Cricket) but ask for a fact it
does not contain, a harder test of grounding than a subject it never
mentions. Every keyword is checked against its source document.

**Licensing:** Wikipedia text is CC BY-SA 4.0. Every generated file records
its title, the exact revision used (an `oldid` link), the retrieval date and
the licence in front matter, which `ingest.py` keeps as chunk metadata.

**Trade-offs accepted:**
- Wikipedia changes; the revision recorded in each file makes the knowledge
  base reproducible, and re-running the script is a deliberate update.
- About half of the v2 `out_of_scope` questions become answerable (MS Dhoni,
  Rohit Sharma, Muttiah Muralitharan, the Big Bash League, the
  Duckworth–Lewis–Stern method and others). They are re-checked and
  relabelled in the v3 test set rather than silently scored as failures.
- A bigger index means more retrieval competition for the original
  questions; the v3 re-baseline measures whether that costs accuracy.

---

## D-015 · Save the Index and Reuse It Until the Knowledge Base Changes

**Date:** October 2026 · **Supersedes:** D-005

**Decision:** `ingest()` saves the built Chroma index under `.index/` (or
`PITCHWISE_INDEX_DIR`), in a folder named after a fingerprint: a SHA-256 hash
of every knowledge-base file plus the embedding model and chunking
settings. On start, a saved index with a matching fingerprint is loaded;
anything else is rebuilt. The Docker image builds the index at image build
time, so a cold start on a hosting tier loads it instead of embedding.

**Why D-005 no longer holds:** D-005 rebuilt the index on every start
because 81 chunks took a few seconds to embed. With the Wikipedia articles
there are thousands of chunks; embedding them on a 2-vCPU free host on
every cold start would add a minute or more before the first answer.

**Why a fingerprint and not a timestamp:** the index is correct only for
the exact files and settings it was built from. Hashing the content means
an edit anywhere in the knowledge base, a new embedding model or a changed
chunk size forces a rebuild, and nothing else does.

*(October 2026: the fingerprint first covered the chunk size and overlap but
not the minimum-text filter, so changing that filter changed the index
without changing its fingerprint, and a stale saved index would have been
reused. The fingerprint now includes every chunking setting.)*

**Why a folder per fingerprint:** Chroma keeps one client open per folder
inside a process. Rebuilding in place would delete files an open client
still uses. Building into a new folder and removing the old ones afterwards
avoids that, and an interrupted build never replaces a good index.

**Trade-offs accepted:** two processes building the same new index at the
same moment could collide; in practice the app and an evaluation run share
one index that is built once. `persist_directory=None` still builds in
memory, which the tests use.

---

## D-016 · v3 Evaluation Runs Stay on Groq's Free Tier; Paid Routes Are Opt-In

**Date:** October 2026

**Decision:** the v3 runs (linear and two agentic on the v3 test set, plus
the k=8 control on its `spanning` questions, D-017) use
the same pinned model and provider as v2, qwen3.8-27b on Groq's free tier,
run day by day as the daily token limit allows. A paid route to the same
model (OpenRouter) was added as an option but is not used for these runs.

**Why:** the three runs need roughly 1.5 million tokens against Groq's free
limit of 200,000 a day for this model, so they take about a week; Groq's
paid tier was not accepting upgrades at the time. The alternative was the
same open-weight model through OpenRouter for under a dollar, finishing in
a day. Staying on Groq keeps v3 identical to v2 in model *and* provider, so
the only differences between v2 and v3 are the knowledge base and the test
set. A run that spans several days is still one model throughout, because
`--pin-model` makes a rate-limited question fail and be retried the next
day instead of falling back to another model.

**What was built for the option, and kept:**
- `evaluation.eval --model` chooses the answer model; with `--pin-model` it
  is the only model used, for routing calls too (C-005).
- Paid routes live in `EVAL_ONLY_MODEL_OPTIONS`, not in `MODEL_OPTIONS`, so
  the chat UI, the public API and automatic fallback can never spend money
  on them; a test enforces this.
- Prices are keyed by provider and model, because the same model costs
  different amounts from different providers. The first version keyed them
  by model alone, so adding OpenRouter's price for qwen3.8-27b silently
  replaced Groq's; a test caught it before any run.

**Status (October 2026):** done. All four v3 runs (linear, the k=8 control
and two agentic) finished by October 9 on Groq's free tier, every row
answered by qwen3.8-27b. Their result rows record about 0.76 million
answering tokens and 0.23 million judge tokens, less than the estimate
above; rows count successful calls only, so rate-limited retries are not
included.

**Trade-off accepted:** results arrive in about a week rather than a day,
and the shared daily limit means the app should be tried on a different
model (each Groq model has its own free allowance) while the runs are in
progress.

---

## D-017 · The k=8 Control: Does Retrieving More Chunks Do What the Agent Does?

**Date:** October 2026

**Decision:** alongside the v3 runs, run the linear engine at k=8 on the 32
`spanning` questions of the v3 test set, pinned to the same model, with
`evaluation.eval --k 8 --category spanning --pin-model`. Compare it with the
`spanning` rows of the v3 linear (k=4) and agentic runs, and report the
result in EVAL_RESULTS whichever way it goes.

**Why:** D-009 named "raise k" as the cheapest alternative to the agentic
engine, and the v2 evaluation never ran it, so the v2 results only show that
the agent beats k=4. With about 3,100 chunks the question matters more: more
chunks could recover the missing facts, or could add near-duplicate noise.
If k=8 matches the agent on `spanning` at one LLM call per question instead
of about three, the graph is not earning its cost.

**Why this way, not `compare_k.py`:** the eval harness already pins the
model (D-010, C-005), records tokens and cost per question (D-012), traces
and scores in Langfuse (D-013), and writes rows `compare_runs.py` can read.
`--k` and `--category` were added to it, and the retrieval metrics use the
same k as the answers, so a k=8 run reports k=8 keyword coverage. Limiting
the run to `spanning` makes it 32 questions, under a fifth of a full run's
177; at twice the retrieved context of a k=4 call, it costs roughly a third
of the tokens of the v3 linear run.

**Trade-off accepted:** one k=8 run on one category is a control, not a
sweep; k=6 is not tested. The comparison with the k=4 linear run is across
two runs, which carry run-to-run variance (v2's two agentic runs differed by
up to 0.13 per category).

**Status (October 2026):** measured. On the 32 `spanning` questions, k=8
scored 4.31 against 3.62 at k=4, and the agent 4.44 and 4.50 in its two runs
(after C-011): ahead of k=8 in both, by one or two correct answers, at about
2.3 times k=8's tokens. Raising k recovers most of the gain; the agent adds
a small, consistent amount at about twice the cost per correct compound
answer. Full results: [`EVAL_RESULTS.md`](./EVAL_RESULTS.md).

---

## D-018 · Measurement Limits Kept Fixed for the v3 Runs

**Date:** October 2026

**Decision:** three weaknesses in how runs are measured were found by a
code review while the v3 runs were in progress. They are left unchanged
until every v3 run is finished, because changing them mid-way would make
the v3 linear run, which is already complete, not comparable with the
agentic runs. They are fixed together afterwards.

1. **Answers are generated at the provider's default temperature.** The
   answer call sends no temperature, so Groq uses its default (1.0), while
   the plan, grade and rewrite calls run at 0 (D-009) and the judge runs at
   0. Every v2 and v3 run shares this, so comparisons are like with like,
   but each answer carries sampling noise. This is part of why each
   agentic configuration is run twice. *After v3:* evaluation runs generate
   at temperature 0.
2. **Keyword matching is by substring, not whole word.** The retrieval
   metrics (MRR, nDCG, keyword coverage) count a keyword as found inside
   any longer word: "red" appears in 1,550 chunks as a substring ("scored",
   "retired") and in 22 as a word; "199" matches "1999". In the v3 test set,
   79 of the 162 answerable questions have a keyword that also matches
   inside other words; for 7 the effect is large (the ball-colour, Ashes
   urn, team-size and boundary-six questions, among others). This inflates
   the retrieval metrics for those questions only. The judge's scores,
   which are the headline results, do not use keywords. *After v3:* match
   whole words, make generic keywords specific, and recompute the
   retrieval metrics of every saved run, which needs no LLM calls.
3. **nDCG is computed against the retrieved list only.** Its ideal ranking
   is built from the chunks that were retrieved, so it measures whether the
   relevant chunks found are ranked first, not whether relevant chunks were
   missed. Keyword coverage measures that. *After v3:* document the
   distinction in the evaluation notes rather than change the metric.

**Why wait:** the v3 linear run is complete and the agentic runs started
on the same code. Fixing any of these now would change two things between
the runs being compared.

---

# Part II — Corrections

> Every significant bug, incorrect assumption, and deliberate change made during
> the build. It exists so that every line of code can be defended — including
> the lines that were wrong first.

---

## C-001 · Silent Chunk Duplication on Repeated ingest() Calls

**Date:** September 2026

**The assumption:** calling `ingest()` more than once in the same process
(e.g. once in an earlier notebook cell, then again to build a fresh store
for a later test) is safe and simply rebuilds the vector store from scratch.

**What's actually true:** `build_vector_store()` called `Chroma.from_documents()`
against the same `collection_name` without first clearing any existing
collection. A second `ingest()` call in the same process appended a full
second copy of every chunk into the same in-memory collection rather than
replacing it — 81 chunks became 162 silently, no error raised.

**Found by:** running a fallback test that called `ingest()` again in a
notebook session that had already built a store earlier. The printed chunk
count (162) didn't match the expected 81, and was noticed only because the
ingestion function's own print statement reports the count on every run —
this was a stated invariant that quietly stopped holding.

**The fix:** `build_vector_store()` now explicitly attempts to delete any
existing collection with the same name before building a fresh one,
wrapped in a try/except since there may be no existing collection to
delete on a genuinely first run.

**Why this matters beyond the test:** the same failure mode would occur in
production if `ingest()` were ever called more than once per app lifetime
(e.g. a future hot-reload or manual refresh trigger) — doubled chunks would
silently degrade retrieval quality (duplicate near-identical results
crowding out distinct ones) without any visible error, exactly the kind of
failure that's hard to catch without a stated invariant to check against.

---

## C-002 · Transient Network Failure Crashed a 104-Question Evaluation Run

**Date:** September 2026

**The assumption:** a full evaluation run across the test set would either
succeed question-by-question or fail cleanly, and any failure would be a
real model or code error worth stopping for.

**What's actually true:** partway through a 104-question run (at question
40), a dropped connection (`Server disconnected without sending a response`)
to a provider's API crashed the entire process. This was a transient
network failure, not a model or logic error — but because the evaluation
loop had no retry logic and saved no results incrementally, all 39 already-
completed, successful evaluations were lost along with the crash.

**Found by:** the run's traceback pointed to an `httpx.RemoteProtocolError`
several layers below the actual evaluation code, with no application-level
handling in between.

**The fix:** the evaluation loop now (1) retries each retrieval/answer call
up to 3 times with a short delay on failure, distinguishing transient
errors from genuine ones by simply attempting again rather than assuming
either way, and (2) writes each result to `evaluation/results.jsonl`
immediately as it completes, rather than holding all results in memory
until the end. A rerun after any future crash skips questions already
present in `results.jsonl` instead of re-doing (and re-spending API calls
on) work that already succeeded.

**Why this matters beyond this one run:** any evaluation or batch-processing
loop that makes many sequential external API calls will eventually hit a
transient failure at that provider's scale of usage. Treating a single
dropped connection as a reason to discard all prior progress is a fragile
default; retry-with-backoff plus incremental persistence is the correct
baseline for this class of task, not an optional enhancement.

---

## C-003 · System Prompt Update to Address a Measured Grounding Gap

**Date:** September 2026

**The assumption:** a single soft instruction in the system prompt ("if you
don't know the answer, say so") would be sufficient to make the assistant
decline to answer questions the knowledge base doesn't cover.

**What's actually true:** manual testing surfaced a case where the
assistant confidently answered a question about cricket nicknames not
present in the knowledge base, using general model knowledge instead of
declining. Running the full evaluation set's `out_of_scope` category (15
questions specifically designed to test this) confirmed it was not a
one-off: that category scored 3.3/5 average accuracy, meaningfully below
the 4.6/5 overall average, with roughly half of out-of-scope questions
answered from general knowledge rather than correctly declined.

**Found by:** the manual test surfaced the failure first; the evaluation
harness's `out_of_scope` category then measured how frequently it occurred,
rather than relying on the single manually-found example.

**The fix:** the system prompt was rewritten from a single soft suggestion
into explicit, repeated grounding rules: an instruction to answer strictly
from the provided context and not from outside knowledge, an exact required
refusal phrase for when the context is insufficient, and an explicit
instruction not to name entities absent from the retrieved context. The
evaluation set was then rerun to measure whether the rewritten prompt
closed the gap.

**Why this is logged as a correction and not just a prompt tweak:** the
original prompt's failure was not visible from casual use — most questions
are answerable from the knowledge base and behaved correctly regardless.
The gap was only found through deliberate adversarial testing (asking about
things known to be absent) and only measured systematically through the
evaluation harness's dedicated category for it. This is the kind of failure
an evaluation harness exists to catch — a system that looks correct in
casual use but has a measurable, specific weakness that only shows up under
targeted testing.

**Verified fix, reproduced across two independent runs:** after rewriting
the prompt, the full 104-question evaluation was run twice, independently.
`out_of_scope` accuracy went from 3.3/5 (pre-fix) to **5.0/5 in both
post-fix runs** — every one of the 15 out-of-scope questions correctly
declined rather than answered from general knowledge, in both runs. This
consistency across two separate runs is meaningful: LLM outputs carry
natural run-to-run variance (observed elsewhere in this project, e.g.
D-006's latency measurements), so a fix that holds at 5.0/5 twice is
stronger evidence than a single favorable run would be.

**A related but distinct finding, surfaced by the same two runs:** the
`spanning` category (compound/comparison questions) showed a small dip
across the same two runs (approximately 4.35 → 4.10 average accuracy) —
not a regression of the grounding fix itself, but a side effect worth
tracking separately. See the update to D-007 below.

---

## C-004 · Misdiagnosed Refusal: the ODI Powerplay Length Is Not in the Knowledge Base

**What D-007 said:** the powerplay question ("How many overs separate ODI
and T20I powerplays?") was filed as a Type 2 failure: an over-conservative
refusal despite full retrieval, because `keyword_coverage` was 100%.

**What the v2 agentic engine showed:** on this question the grade step
reported `missing: ODI powerplay overs` three times in a row, across the
original retrieval and two rewritten queries (10 chunks in the end), and
the engine then generated a refusal:

```
plan:multi(2) -> retrieve:3q/7chunks -> grade:missing(ODI powerplay overs)
-> rewrite:ODI powerplay overs count -> retrieve:4q/8chunks -> grade:missing(...)
-> rewrite:ODI powerplay length -> retrieve:5q/10chunks -> grade:missing(...) -> generate
```

Checking the source confirmed it: `t20i-cricket.md` states the 6-over
powerplay, but `odi-cricket.md` only says fielding restrictions apply "in
different phases of the innings" and never gives an over count. The
question has no answer in the knowledge base, so refusing was correct.

**Root cause of the misdiagnosis:** this question's keywords are
`["powerplay", "ODI", "T20I"]`. Those words appear in the retrieved chunks
whether or not the needed number does, so 100% keyword coverage did not
mean the facts were present. Keyword coverage is only as good as the
keywords; for questions that combine numbers, the keywords should be the
numbers themselves.

**What changed:**
- The README and `graph.py` demo example now use a question the knowledge
  base can answer ("Compare the founding years of the Cricket World Cup and
  the T20 World Cup").
- The test question and its keywords are left unchanged for now, so v2
  runs stay comparable with the v1 baseline. Rewriting under-specified
  keywords (and adding the ODI powerplay fact, if wanted) is part of the
  v3 knowledge-base expansion.

**Why it's worth logging:** the agentic engine's grade step turned out to
be a diagnostic tool as well as a retrieval step. It names the missing
fact, which made a wrong conclusion in this document checkable in minutes.

---

## C-005 · `--pin-model` Did Not Pin the Agentic Engine's Routing Calls

**What D-010 said:** with `--pin-model`, provider fallback is disabled for the
whole run, so a run measures exactly one model.

**What the code did:** the answer step (`generate`) was pinned, but the plan,
grade and rewrite calls went through `graph.call_llm`, which always used the
default fallback chain. Found in a post-run review of the code against this
document, not by a failing run.

**Effect on the published v2 runs (A1, A2):** the fallback chain starts with
the same model the runs were pinned to (qwen3.8-27b on Groq), so routing
calls used it unless it failed. It did fail with rate-limit errors near the
end of A1, when the free daily token limit ran out, so some late A1 routing
calls may have been answered by the next model in the chain. Which model
handled routing was not recorded, so this cannot be checked after the fact.
Every answer in all three runs was generated by qwen3.8-27b (recorded per
row). The linear baseline (L1) is unaffected: it makes no routing calls. The
A1 and A2 results agree on every category within 0.13, which suggests no
large effect, but the claim "one model per run" is narrowed to "one answer
model per run" for those runs.

**Fix:**
- `call_llm` now takes the run's `model_label` and `allow_fallback`, so a
  pinned run pins plan, grade and rewrite too.
- The trace records `routing_models` (the models that handled plan, grade
  and rewrite), and `evaluation/eval.py` writes it to every result row.
- `tests/test_graph.py` test 13 checks that a pinned run passes the pin to
  every LLM call; it fails against the old code.

**Side effect, intended:** in the app, choosing a model in the UI or the API
now applies to the routing calls as well as the answer (with fallback, as
before), instead of routing always starting from the default model.

---

## C-006 · Four Curated Sections Missed the D-003 Restructuring

**What D-003 said:** every section header and opening line repeats its
subject's name, so a chunk is unambiguous once separated from its document.

**What was found:** four player files (Ben Stokes, Virat Kohli, Kane
Williamson, Jasprit Bumrah) still had a bare `## Personal` section whose
text never named the player. The cost was measured: "Has Ben Stokes retired
from any international format?" (Q83) failed in every recorded run, v1 and
v2, linear and agentic, because the retirement fact sat in that unnamed
chunk.

**Fix:** the four headings and opening lines now name the player
(for example "## Ben Stokes's Honours and Retirement from ODI and T20I
Cricket"), with no facts changed. This alters four curated chunks, so the v2
results are not re-scored with it; the v3 test runs include the fix.

**Prevention:** the Wikipedia conversion (D-014) builds every heading as
"Subject: Section" and prefixes every chunk with it, so this mistake cannot
recur in generated documents.

---

## C-007 · A Curated Document Went Out of Date: Ben Stokes Retired in 2026

**What was found:** after the Wikipedia articles were added, the knowledge
base contradicted itself. The curated Ben Stokes document (written from
mid-2026 information) said he "continues to play and captain in Test
cricket"; Joe Root's article (September 2026 revision) said Root had been
reappointed Test captain in 2026, and Brendon McCullum's said he had been
sacked as Test coach in July 2026.

**Checked against two independent sources:** Ben Stokes's last Test was the
third Test against New Zealand (25–29 June 2026), in a home series England
lost 2–1, and he retired from international cricket; Joe Root was
reappointed Test captain from the August 2026 series against Pakistan.

**Fix:** the Stokes document now says he captained England's Test side from
April 2022 until retiring from international cricket in 2026, and names
Root as his successor. Only those facts changed. In the v3 test set, the
three Stokes captaincy and retirement questions (v2 Q21, Q83, Q88) have
updated reference answers; v2 Q88 now asks who succeeded him, which needs
both the Stokes document and Root's article.

**Why it matters:** curated documents carry an "as of" date and go stale
without anyone noticing; a second, independently dated source exposed it.
A knowledge base needs a way to find its own contradictions, and this one
was found by reading, not by a check. An automated check for conflicting
claims about the same subject is a future step.

---

## C-008 · A v2 Keyword That Could Never Match

**What was found:** while checking every v3 keyword against the knowledge
base, v2 Q15 ("How many Test runs did Sachin Tendulkar score?") turned out
to list its answer twice, as "15,921" and "15921". The documents only use
"15,921", so keyword coverage for Q15 could never exceed 67%, and that is
what every v1 and v2 run reported.

**Fix:** the v3 set drops the unmatched form. The v2 set is left as it was
so the v2 results stay reproducible; the effect on v2's averages is one
question's retrieval coverage, not any answer score.

## C-009 · Two v3 Test Questions Were Wrong, Found by the First v3 Run

**What was found:** reading the v3 linear run question by question, not
just its averages, showed two failures that were the test set's fault:
- *Q95, "What is Rohit Sharma's career Test batting average?"*, was kept as
  out of scope, but Rohit Sharma's article says he "finished his Test career
  having played 67 Tests and making 4,301 runs at 40.57". The model answered
  40.57, correctly from the context, and the judge scored it 1 because the
  label said it should decline. The search used to re-check the v2
  out-of-scope questions had looked for "Rohit" within 60 characters of
  "average" and missed this sentence.
- *Q122, "Who was the first player to score a double century in ODI
  cricket?"*, has two answers in the knowledge base: Belinda Clark's
  article (1997, correct) and the curated Sachin Tendulkar document ("the
  first player to score a double century in ODI cricket", 2010, true only
  of men's ODIs). The model answered Tendulkar.

**Fix:** Q95 is now a direct_fact question with the answer from the
article; Q122 asks for the first woman to score an ODI double century. Both
were fixed before the agentic runs started, and the two questions were
re-run in the linear run, so every v3 run uses the same test set. The
curated Tendulkar sentence is left unchanged until the v3 runs finish, so
the knowledge base stays identical across them; it is a known inaccuracy
to correct afterwards (a second instance of the C-007 pattern: a curated
claim that a larger knowledge base contradicts).

**Lesson:** an out-of-scope label is a claim that the knowledge base does
*not* contain something, which is harder to verify than a keyword check.
Absence was checked with narrow search patterns; the model and the judge
together caught what the patterns missed.

## C-010 · A Query Rewrite Added Nothing Once the First Search Filled the Cap

**What D-009 said:** when the grader names a missing fact, the engine writes
a new search query for it and retrieves again.

**What the code did:** `retrieve` searched the original question, then the
planner's sub-queries, then the rewrite queries, and kept the first 10
unique chunks. The original question and two sub-queries return up to 12
chunks, so whenever they already filled the cap, the rewrite query's
chunks were cut and the grader saw the same 10 chunks again. Each such
rewrite cost two LLM calls (rewrite and grade) and changed nothing. Found
in a code review and reproduced with the real graph and a scripted store:
two rewrites, `retrieve:3q/10chunks`, `4q/10chunks`, `5q/10chunks`, and none
of the rewrite queries' chunks in the final context. Where the first search
returned fewer than 10 unique chunks, as in C-004's trace (7 chunks), the
rewrites did add chunks, which is why the bug did not show in that example.

**Fix:** rewrite queries are searched right after the original question,
newest first, before the planner's sub-queries. The original question's
chunks still come first, so the agentic context still contains everything
the linear engine would retrieve; the planner's sub-queries take the
remaining places. `tests/test_graph.py` test 14 fills the cap on the first
search and checks the rewrite's chunk reaches the answer; it fails against
the old code.

**Effect on published results:** v2's agentic runs used rewrites on 2 of
104 questions each, so the v2 numbers barely touch this path and are left
as they are. The v3 agentic run 1 had started before the fix; its 5
questions that used a rewrite were re-run with it (the other 172 never reach
the changed code), and none of their scores changed. Run 2 ran with the fix
throughout. The v3 linear run makes no rewrites and is unaffected.

## C-011 · The Judge Scored Some Wrong Refusals as Correct

**What was found:** reading the v3 runs question by question, the judge had
given 5 out of 5 to five answers that were exactly the fixed refusal
sentence ("I don't have information about that in my knowledge base.") on
questions the knowledge base answers: two in the linear run, three in the
second agentic run. In another run, the identical refusal to the same
question was scored 1. The judge's feedback on the 5s called declining "an
appropriate response when lacking the data", which is the out-of-scope rule
in its prompt applied to an answerable question.

**Why it matters:** these were the largest single differences between runs.
With them scored as wrong, the two agentic runs agree to within 0.01
overall instead of 0.08, and the linear run's `temporal` 5.0 becomes 4.75
(its India captaincy answer was a refusal). The conclusions do not change:
on `spanning`, the agent is still ahead of k=8 in both runs, by a smaller
margin in run 2 (0.19 instead of 0.31).

**Fix, for now:** EVAL_RESULTS counts the five as wrong (accuracy 1) and
shows the as-judged figure beside each corrected one; the results files
keep the judge's original scores, so the correction is reproducible from
them. One case is left as judged: "How many ODI runs has Kane Williamson
scored?" asks for a total the knowledge base does not give, so refusing is
correct. Every v2 run is unaffected apart from that question.

**Fix, after v3 (with D-018):** the harness scores the exact refusal
sentence on an answerable question as 1 by rule, without asking the judge.
The fixed refusal sentence introduced in C-003 is what makes this check
exact. Gemini 3.5 Flash Lite also ignores the temperature setting (it uses
fixed sampling), so the judge itself is not deterministic; a rule removes
it from the one case that matters most.

**Lesson:** an LLM judge needs the same scrutiny as the system it grades.
These scores were caught only because the runs were read question by
question and compared with each other.

---
*Author: Somesh Kant Tiwari*
*Last updated: October 2026*