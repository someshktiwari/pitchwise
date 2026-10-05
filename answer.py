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
    """Call a Gemini model via LangChain, returning a plain string.
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

    content = response.content
    if isinstance(content, str):
        return content
    # nested format: list of dicts, each possibly containing a "text" field
    if isinstance(content, list):
        parts = [item.get("text", "") for item in content if isinstance(item, dict)]
        return "".join(parts)
    return str(content)


def call_openai_compatible(base_url, api_key, model, system_prompt, question, history=None, temperature=None):
    """Call a Groq or OpenRouter model — both expose an OpenAI-compatible
    endpoint, so a single function handles both via a different base_url."""
    client = OpenAI(api_key=api_key, base_url=base_url)

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(_history_to_openai_messages(history))
    messages.append({"role": "user", "content": question})

    # Only send temperature when a caller asks for one, so the default
    # generation path behaves exactly as it did in v1 (D-009).
    kwargs = {"temperature": temperature} if temperature is not None else {}
    response = client.chat.completions.create(model=model, messages=messages, **kwargs)
    return response.choices[0].message.content


def call_model(provider, model, system_prompt, question, history=None, temperature=None):
    """Route to the correct provider's call function. Returns a plain
    string answer regardless of which provider handled it."""
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


def build_try_order(model_label=None, allow_fallback=True):
    """The order in which models are tried: the requested model first (if
    any), then the rest of FALLBACK_ORDER (D-008). With allow_fallback=False
    only the first model is tried, so an evaluation run measures exactly one
    model instead of silently mixing providers (D-010)."""
    try_order = []
    if model_label and model_label in MODEL_OPTIONS:
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
        provider, model = MODEL_OPTIONS[label]
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
    """Answer a question with one of two engines.

    engine="linear"  — v1 pipeline: retrieve k chunks, generate once.
    engine="agentic" — v2 LangGraph engine (graph.py): plan, retrieve,
                       grade, capped query rewrite, generate (D-009).

    Accepts prior conversation history (list of {"role", "content"} dicts,
    Gradio's format) so follow-up questions can resolve pronouns.

    Returns (answer_text, docs, trace). docs is the list of Document objects
    used as context; trace is a dict describing what the engine did
    (route, sub-queries, rewrites, LLM calls, latency, model that answered).
    """
    if engine == "agentic":
        from graph import run_agentic  # imported lazily: langgraph only needed for this engine
        return run_agentic(vectorstore, question, history=history,
                           model_label=model_label, allow_fallback=allow_fallback)
    if engine != "linear":
        raise ValueError(f"Unknown engine: {engine}")

    start = time.perf_counter()
    docs, context = retrieve_context(vectorstore, question, k=k)
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
    }
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