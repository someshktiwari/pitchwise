"""
answer.py

Takes a user's question, retrieves relevant chunks from the vector store
built by ingest.py, and generates a grounded answer using an LLM.

Implements the decisions recorded in DECISIONS.md:
  - D-006: Groq (qwen3.8-27b) is the default provider/model, chosen from a
    measured speed comparison across 7 provider/model combinations
  - Silent automatic fallback across providers if the selected one fails
    or is rate-limited, in order of measured speed

Supports Gemini, Groq, and OpenRouter — the user (or app.py's UI) can select
any of them explicitly; if unspecified, the fastest default is used.

v2 (D-009): answer_question() takes engine="linear" (this file's original
pipeline) or engine="agentic" (graph.py). Both share retrieve_context() and
generate_from_context(), and both return (answer_text, docs, trace).
"""

import os
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from openai import OpenAI

load_dotenv(override=True)

import observability as obs  # optional Langfuse tracing (D-013)
import usage                 # token and cost accounting (D-012)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

RETRIEVER_K = 4  # chunks retrieved per question — Chroma's default, stated
                 # explicitly rather than left implicit; revisit once the
                 # evaluation harness can measure retrieval quality directly

SYSTEM_PROMPT_TEMPLATE = """
You are Pitchwise, a cricket assistant that answers ONLY using the context provided below, retrieved from a curated knowledge base.

Rules:
- Base your answer STRICTLY on the provided context. Do not use any outside knowledge, even if you are confident it is correct.
- If the context does not contain enough information to answer the question, respond exactly: "I don't have information about that in my knowledge base." Do not guess, speculate, or answer from general knowledge instead.
- Do not mention players, teams, tournaments, or facts that do not appear in the context below.
- Keep answers concise and directly grounded in the retrieved context.

Context:
{context}
"""

# Model choices exposed to the UI, and the order automatic fallback tries
# them in — fastest first, per DECISIONS.md D-006's measured comparison.
MODEL_OPTIONS = {
    "Groq: qwen3.8-27b (fastest, default)": ("groq", "qwen/qwen3.8-27b"),
    "Groq: gpt-oss-20b": ("groq", "openai/gpt-oss-20b"),
    "Groq: gpt-oss-120b": ("groq", "openai/gpt-oss-120b"),
    "Gemini: 3.5-flash-lite": ("gemini", "gemini-3.5-flash-lite"),
    "OpenRouter: minimax-m3": ("openrouter", "minimax/minimax-m3:free"),
    "Gemini: 3.6-flash": ("gemini", "gemini-3.6-flash"),
    "Gemini: 3.1-flash-lite": ("gemini", "gemini-3.1-flash-lite"),
}

DEFAULT_MODEL_LABEL = "Groq: qwen3.8-27b (fastest, default)"

# Fallback tries every option in MODEL_OPTIONS order if the selected one fails
FALLBACK_ORDER = list(MODEL_OPTIONS.keys())

# Paid routes the evaluation can opt into with `evaluation.eval --model`
# (none of the published runs uses one). They are deliberately not in
# MODEL_OPTIONS, so the UI, the public API and automatic fallback can never
# spend money on them (D-016).
EVAL_ONLY_MODEL_OPTIONS = {
    # The same open-weight model as the Groq default, on OpenRouter's paid route
    "OpenRouter: qwen3.8-27b (paid)": ("openrouter", "qwen/qwen3.8-27b"),
}
ALL_MODEL_OPTIONS = {**MODEL_OPTIONS, **EVAL_ONLY_MODEL_OPTIONS}


@obs.observe(name="retrieve", as_type="retriever")
def retrieve_context(vectorstore, question, k=None):
    """Search the vector store for chunks relevant to the question.
    Returns the retrieved Document objects and a formatted context string.

    k defaults to RETRIEVER_K (the production default, D-007) but can be
    overridden — used for testing different retriever k values against
    the evaluation harness without changing the app's actual default."""
    retriever_k = k if k is not None else RETRIEVER_K
    retriever = vectorstore.as_retriever(search_kwargs={"k": retriever_k})
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    obs.update_span(input={"query": question, "k": retriever_k},
                    output=[f"{d.metadata.get('source', '').split('/')[-1]} | "
                            f"{d.metadata.get('header_2') or d.metadata.get('header_1') or ''}" for d in docs])
    return docs, context


def _history_to_langchain_messages(history):
    """Convert Gradio's history (list of {"role": ..., "content": ...}
    dicts) into LangChain message objects, for providers called via
    LangChain (currently just Gemini)."""
    messages = []
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        elif turn["role"] == "assistant":
            messages.append(AIMessage(content=turn["content"]))
    return messages


def _history_to_openai_messages(history):
    """Convert Gradio's history into the plain role/content dict format
    used by OpenAI-compatible endpoints (Groq, OpenRouter)."""
    return [{"role": turn["role"], "content": turn["content"]} for turn in history]


def call_gemini(model, system_prompt, question, history=None, temperature=None):
    """Call a Gemini model via LangChain. Returns (text, input_tokens,
    output_tokens); token counts are None if the response has no usage.
    Gemini's response.content can be a nested list-of-dicts structure
    (seen during exploration, tied to its extended reasoning mode) rather
    than a plain string, so this normalizes it either way."""
    # temperature=0 by default, as before; callers may override (D-009)
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=0 if temperature is None else temperature,
        google_api_key=GEMINI_API_KEY,
    )

    messages = [SystemMessage(content=system_prompt)]
    if history:
        messages.extend(_history_to_langchain_messages(history))
    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages)
    meta = getattr(response, "usage_metadata", None) or {}
    tokens = (meta.get("input_tokens"), meta.get("output_tokens"))

    content = response.content
    if isinstance(content, str):
        return (content, *tokens)
    # nested format: list of dicts, each possibly containing a "text" field
    if isinstance(content, list):
        parts = [item.get("text", "") for item in content if isinstance(item, dict)]
        return ("".join(parts), *tokens)
    return (str(content), *tokens)


def call_openai_compatible(base_url, api_key, model, system_prompt, question, history=None, temperature=None):
    """Call a Groq or OpenRouter model — both expose an OpenAI-compatible
    endpoint, so a single function handles both via a different base_url.
    Returns (text, input_tokens, output_tokens) from the response's usage
    block (None if the provider doesn't send one)."""
    client = OpenAI(api_key=api_key, base_url=base_url)

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(_history_to_openai_messages(history))
    messages.append({"role": "user", "content": question})

    # Only send temperature when a caller asks for one, so the default
    # generation path behaves exactly as it did in v1 (D-009).
    kwargs = {"temperature": temperature} if temperature is not None else {}
    response = client.chat.completions.create(model=model, messages=messages, **kwargs)
    u = getattr(response, "usage", None)
    return (response.choices[0].message.content,
            getattr(u, "prompt_tokens", None), getattr(u, "completion_tokens", None))


def _call_provider(provider, model, system_prompt, question, history=None, temperature=None):
    """Route to the correct provider's call function.
    Returns (text, input_tokens, output_tokens)."""
    if provider == "gemini":
        return call_gemini(model, system_prompt, question, history=history, temperature=temperature)
    elif provider == "groq":
        return call_openai_compatible(GROQ_BASE_URL, GROQ_API_KEY, model, system_prompt, question,
                                      history=history, temperature=temperature)
    elif provider == "openrouter":
        return call_openai_compatible(OPENROUTER_BASE_URL, OPENROUTER_API_KEY, model, system_prompt, question,
                                      history=history, temperature=temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}")


@obs.observe(name="llm-call", as_type="generation")
def call_model(provider, model, system_prompt, question, history=None, temperature=None):
    """Call one model and return its text. Every call, successful or not, is
    recorded in the active usage meter (D-012) and, when tracing is on, as a
    Langfuse generation with its tokens and list-price cost (D-013)."""
    timer = usage.Timer()
    try:
        text, input_tokens, output_tokens = _call_provider(
            provider, model, system_prompt, question, history=history, temperature=temperature)
    except Exception as e:
        usage.record(provider, model, None, None, timer.ms(), ok=False, error=e)
        obs.update_generation(name=usage.current_step(), model=model, level="ERROR",
                              status_message=str(e)[:500])
        raise
    usage.record(provider, model, input_tokens, output_tokens, timer.ms())
    cost = usage.cost_usd(provider, model, input_tokens, output_tokens)
    obs.update_generation(
        name=usage.current_step(),
        model=model,
        input=[{"role": "system", "content": system_prompt}, *(history or []),
               {"role": "user", "content": question}],
        output=text,
        metadata={"provider": provider},
        usage_details={k: v for k, v in (("input", input_tokens), ("output", output_tokens)) if v is not None},
        cost_details=({"total": cost} if cost is not None else None),
    )
    return text


def build_try_order(model_label=None, allow_fallback=True):
    """The order in which models are tried: the requested model first (if
    any), then the rest of FALLBACK_ORDER (D-008). With allow_fallback=False
    only the first model is tried, so an evaluation run measures exactly one
    model instead of silently mixing providers (D-010)."""
    try_order = []
    if model_label and model_label in ALL_MODEL_OPTIONS:
        try_order.append(model_label)
    for label in FALLBACK_ORDER:
        if label not in try_order:
            try_order.append(label)
    return try_order if allow_fallback else try_order[:1]


def generate_with_fallback(system_prompt, question, history=None, model_label=None,
                           allow_fallback=True, temperature=None):
    """Call the first model in the try-order that succeeds. Returns
    (answer_text, label_that_answered). Raises RuntimeError if every
    option fails."""
    last_error = None
    for label in build_try_order(model_label, allow_fallback):
        provider, model = ALL_MODEL_OPTIONS[label]
        try:
            text = call_model(provider, model, system_prompt, question,
                              history=history, temperature=temperature)
            return text, label
        except Exception as e:
            last_error = e
            continue  # silently try the next option in the fallback chain

    raise RuntimeError(f"All providers failed. Last error: {last_error}")


def generate_from_context(question, context, history=None, model_label=None, allow_fallback=True):
    """The v1 generation step, unchanged: the grounded system prompt over a
    context string, through the fallback chain. Shared by both engines so
    any difference between them comes from retrieval, not wording (D-009).
    Returns (answer_text, label_that_answered)."""
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    return generate_with_fallback(system_prompt, question, history=history,
                                  model_label=model_label, allow_fallback=allow_fallback)


def format_context(docs):
    """Join retrieved chunks into the context string the prompt expects."""
    return "\n\n".join(doc.page_content for doc in docs)


def answer_question(vectorstore, question, history=None, model_label=None, k=None,
                    engine="linear", allow_fallback=True):
    """Answer a question with one of two engines; see _answer_question.
    Wraps the call in one Langfuse trace when tracing is on (D-013)."""
    with obs.trace_attributes(trace_name="pitchwise-answer", tags=[engine],
                              metadata={"engine": engine, "model_label": model_label or "default",
                                        "pinned": str(not allow_fallback)}):
        return _answer_question(vectorstore, question, history=history, model_label=model_label,
                                k=k, engine=engine, allow_fallback=allow_fallback)


@obs.observe(name="answer_question")
def _answer_question(vectorstore, question, history=None, model_label=None, k=None,
                     engine="linear", allow_fallback=True):
    """Answer a question with one of two engines.

    engine="linear"  — v1 pipeline: retrieve k chunks, generate once.
    engine="agentic" — v2 LangGraph engine (graph.py): plan, retrieve,
                       grade, capped query rewrite, generate (D-009).

    Accepts prior conversation history (list of {"role", "content"} dicts,
    Gradio's format) so follow-up questions can resolve pronouns.

    Returns (answer_text, docs, trace). docs is the list of Document objects
    used as context; trace is a dict describing what the engine did
    (route, sub-queries, rewrites, LLM calls, latency, model that answered,
    and token usage with list-price cost, D-012).
    """
    obs.update_span(input={"question": question, "engine": engine})
    if engine == "agentic":
        from graph import run_agentic  # imported lazily: langgraph only needed for this engine
        return run_agentic(vectorstore, question, history=history,
                           model_label=model_label, allow_fallback=allow_fallback)
    if engine != "linear":
        raise ValueError(f"Unknown engine: {engine}")

    start = time.perf_counter()
    with usage.meter() as calls:
        docs, context = retrieve_context(vectorstore, question, k=k)
        with usage.step("generate"):
            answer_text, answered_by = generate_from_context(question, context, history=history,
                                                             model_label=model_label,
                                                             allow_fallback=allow_fallback)
    trace = {
        "engine": "linear",
        "route": None,
        "sub_queries": [question],
        "rewrites": 0,
        "llm_calls": 1,
        "latency_ms": round((time.perf_counter() - start) * 1000),
        "answered_by": answered_by,
        "routing_models": [],
        "steps": [f"retrieve:{len(docs)}", "generate"],
        "usage": usage.summarise(calls),
    }
    obs.update_span(output={"answer": answer_text, "trace": {k: v for k, v in trace.items() if k != "usage"}})
    return answer_text, docs, trace


if __name__ == "__main__":
    # Quick manual test: run `python answer.py` to sanity-check the full
    # retrieve + generate pipeline without starting the full app.
    import sys
    sys.path.append(".")
    from ingest import ingest

    store = ingest()
    test_question = "What is Don Bradman's exact career Test batting average?"
    answer_text, docs, trace = answer_question(store, test_question)
    print(f"\nQ: {test_question}")
    print(f"A: {answer_text}")
    print(f"Trace: {trace}")
    print(f"\nSources used ({len(docs)} chunks):")
    for doc in docs:
        print(f"- {doc.metadata.get('source')}")