# DECISIONS.md — Pitchwise

Every architectural and implementation decision made in building Pitchwise,
a RAG-based cricket knowledge assistant. Each entry covers the decision, the
alternatives considered, the reasoning, and the trade-offs explicitly accepted.

---

## Table of Contents

**Part I — Architectural Decisions**
D-001 through D-008 — ingestion and generation pipeline decisions.
D-009 through D-011 — v2: agentic engine, evaluation changes, service layer.

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
deliberate choice made so the project can be deployed to a free-tier host
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
  question unless they check logs — acceptable for a demo-facing product at
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
questions scored 4.1/5 against 5.0/5 for direct facts, and D-007's failure
analysis split those failures into retrieval gaps (keyword coverage 0–50%)
and over-conservative refusals. Query decomposition plus a sufficiency check
targets the first type directly: each needed fact gets its own retrieval, and
a missing fact triggers a targeted re-retrieval.

**Alternatives considered:**
- *Raise k (e.g. k=8)* — the cheapest fix for retrieval gaps, with no extra
  LLM calls. Kept as the control in the v2 evaluation: if k=8 matches the
  agentic engine, the graph is not earning its cost.
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

**Status:** built and covered by offline tests; accuracy comparison against
the v1 baseline not yet run. No v2 accuracy claims until it is.

---

## D-010 · Evaluation Runs Can Pin One Model and Use Their Own Results File

**Decision:** `evaluation/eval.py` gains `--engine`, `--results` and
`--pin-model`. With `--pin-model`, provider fallback (D-008) is disabled for
the run, and every result row records which model answered.

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
- *One vector store for API and UI*, built once at startup (D-005).
- *503 when every provider fails* (D-008); 400 for an unknown model; 422 from
  Pydantic for invalid input; questions capped at 500 characters to protect
  free-tier quotas on a public demo.
- *Keys only at run time* (`--env-file` / Space secrets), never in the image.
- *The embedding model is baked into the image* so a cold start doesn't
  download it.

**Trade-offs accepted:** `sentence-transformers` pulls a full PyTorch build,
so the image is large (several GB including CUDA libraries the CPU-only
deployment doesn't use). A CPU-only PyTorch install would cut that
substantially; deferred until the deployment is live.

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
*Author: Somesh Kant Tiwari*
*Last updated: October 2026*