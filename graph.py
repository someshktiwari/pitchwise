"""
graph.py

Pitchwise v2 agentic engine, built with LangGraph (DECISIONS.md D-009).

The v1 pipeline retrieves 4 chunks for the question and generates once. That
works for single facts but misses context on compound questions: the
`spanning` category scored 4.1/5 against 5.0 for direct facts, and the
failures split into retrieval gaps and over-cautious refusals (D-007).

This engine adds three steps around the unchanged v1 retrieval and prompt:

    START -> plan -> retrieve --(simple)--> generate -> END
                        |
                     (multi)
                        v
                      grade --(sufficient, or 2 rewrites used)--> generate
                        |
                (missing, < 2 rewrites)
                        v
                     rewrite -> retrieve  (loop)

- plan:     one LLM call. Classifies the question as simple or multi and,
            for multi, splits it into 2-3 standalone sub-queries.
- retrieve: no LLM. Retrieves for the original question first, then for each
            sub-query; merges, removes duplicate chunks, caps the total.
- grade:    one LLM call over all chunks: is the context sufficient for the
            original question, and if not, which fact is missing?
- rewrite:  one LLM call: a new search query aimed at the missing fact.
- generate: the v1 grounded prompt (answer.py), unchanged.

Design rules:
- Fail open. If plan or grade returns something unparseable, the graph falls
  back to v1 behaviour instead of crashing or guessing.
- Bounded cost. At most 2 rewrites, so the worst case is 7 LLM calls.
- Simple questions skip grading entirely (2 LLM calls), which protects the
  direct-fact category that v1 already answers at 5.0/5.
"""

import json
import operator
import time
from typing import Annotated, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

import answer as base  # accessed as base.x so tests can patch the seams

MAX_SUB_QUERIES = 3
MAX_REWRITES = 2
MAX_CHUNKS = 10
HISTORY_TURNS = 4

PLAN_PROMPT = """You route cricket questions for a retrieval system over a cricket knowledge base.
Return ONLY a JSON object, no other text: {"route": "simple" | "multi", "sub_queries": ["..."]}

- "simple": the question can be answered from one fact or one section (one player, one format,
  one tournament). sub_queries = [the question, rewritten to be standalone].
- "multi": the question needs facts about two or more different players, formats, tournaments,
  or numbers (comparisons, "which X and Y", arithmetic across facts).
  sub_queries = 2 or 3 short standalone search queries, one per needed fact.
- Resolve pronouns using the conversation history, so every sub-query stands alone.
- When unsure, choose "simple".

Example simple: "What is Don Bradman's Test batting average?"
  -> {"route": "simple", "sub_queries": ["Don Bradman Test batting average"]}
Example multi: "How many overs separate ODI and T20I powerplays?"
  -> {"route": "multi", "sub_queries": ["ODI powerplay overs", "T20I powerplay overs"]}"""

GRADE_PROMPT = """You check whether retrieved context contains enough information to answer a question.
Return ONLY a JSON object, no other text: {"sufficient": true | false, "missing": "..."}

- Judge only against the QUESTION. Ignore context that is not needed.
- "sufficient" is true only if every fact needed is present in the context, including every
  number that would have to be combined or compared.
- If not sufficient, "missing" is a short search phrase naming the absent fact
  (for example "T20I powerplay overs"). If sufficient, "missing" is ""."""

REWRITE_PROMPT = """You write one search query for a cricket knowledge base.
Given a question, the fact that is still missing, and the queries already tried,
return ONLY the new query as plain text: a few words aimed at the missing fact,
different from every query already tried. No quotes, no explanation."""


class RAGState(TypedDict):
    question: str
    history: list
    model_label: Optional[str]
    allow_fallback: bool
    route: str
    sub_queries: Annotated[list, operator.add]  # plan adds the first ones, rewrite appends
    docs: list                                  # overwritten: retrieve rebuilds the full set
    sufficient: Optional[bool]
    missing: str
    rewrites: int
    llm_calls: int
    trace: Annotated[list, operator.add]        # one entry appended per step
    routing_models: Annotated[list, operator.add]  # model used by each plan/grade/rewrite call
    answer: Optional[str]
    answered_by: Optional[str]


def call_llm(system, user, temperature=0, model_label=None, allow_fallback=True):
    """Utility call for plan, grade and rewrite, at temperature 0 for
    repeatable routing decisions. Uses the same model choice as the answer:
    the selected model first, then fallback, or exactly one model when the
    run is pinned (allow_fallback=False, D-010 / C-005).
    Returns (text, label_of_the_model_that_answered)."""
    return base.generate_with_fallback(system, user, temperature=temperature,
                                       model_label=model_label, allow_fallback=allow_fallback)


def parse_json(raw):
    """Parse a JSON object from a model reply. Tolerates ``` fences and text
    around the object. Returns None if nothing parseable is found."""
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        cleaned = parts[1] if len(parts) > 1 else cleaned
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    try:
        return json.loads(cleaned.strip())
    except Exception:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except Exception:
            return None
    return None


def _doc_key(doc):
    m = doc.metadata
    return (m.get("source"), m.get("header_1"), m.get("header_2"), doc.page_content)


def build_graph(vectorstore):
    """Compile the agentic graph. The vector store is closed over by the
    retrieve node rather than stored in state, so state stays plain data."""

    def route_llm(state, system, user):
        return call_llm(system, user, model_label=state["model_label"],
                        allow_fallback=state["allow_fallback"])

    def plan(state):
        user = json.dumps({"question": state["question"],
                           "history": state["history"][-HISTORY_TURNS:]}, default=str)
        raw, used = route_llm(state, PLAN_PROMPT, user)
        data = parse_json(raw)
        valid = (
            isinstance(data, dict)
            and data.get("route") in ("simple", "multi")
            and isinstance(data.get("sub_queries"), list)
            and 1 <= len(data["sub_queries"]) <= MAX_SUB_QUERIES
            and all(isinstance(q, str) and q.strip() for q in data["sub_queries"])
        )
        if valid:
            route, subs = data["route"], [q.strip() for q in data["sub_queries"]]
        else:
            route, subs = "simple", [state["question"]]  # fail open to v1 behaviour
        return {"route": route, "sub_queries": subs,
                "llm_calls": state["llm_calls"] + 1, "routing_models": [used],
                "trace": [f"plan:{route}({len(subs)})" + ("" if valid else ":fallback")]}

    def retrieve(state):
        # The original question always goes first, so the agentic context is a
        # superset of what v1 would have retrieved (before the cap).
        queries = [state["question"]]
        for q in state["sub_queries"]:
            if q not in queries:
                queries.append(q)
        seen, merged = set(), []
        for q in queries:
            docs, _ = base.retrieve_context(vectorstore, q)
            for d in docs:
                key = _doc_key(d)
                if key not in seen:
                    seen.add(key)
                    merged.append(d)
        merged = merged[:MAX_CHUNKS]
        return {"docs": merged, "trace": [f"retrieve:{len(queries)}q/{len(merged)}chunks"]}

    def grade(state):
        user = json.dumps({"question": state["question"],
                           "context": base.format_context(state["docs"])})
        raw, used = route_llm(state, GRADE_PROMPT, user)
        data = parse_json(raw)
        if isinstance(data, dict) and isinstance(data.get("sufficient"), bool):
            sufficient, missing, note = data["sufficient"], str(data.get("missing") or ""), ""
        else:
            sufficient, missing, note = True, "", ":fallback"  # fail open: generate as v1 would
        return {"sufficient": sufficient, "missing": missing,
                "llm_calls": state["llm_calls"] + 1, "routing_models": [used],
                "trace": [("grade:ok" if sufficient else f"grade:missing({missing})") + note]}

    def rewrite(state):
        user = json.dumps({"question": state["question"], "missing": state["missing"],
                           "tried": state["sub_queries"]})
        raw, used = route_llm(state, REWRITE_PROMPT, user)
        new = (raw or "").strip().strip('"').strip()
        if not new or new in state["sub_queries"] or new == state["question"]:
            new = state["missing"] or state["question"]
        return {"sub_queries": [new], "rewrites": state["rewrites"] + 1,
                "llm_calls": state["llm_calls"] + 1, "routing_models": [used],
                "trace": [f"rewrite:{new}"]}

    def generate(state):
        text, answered_by = base.generate_from_context(
            state["question"], base.format_context(state["docs"]),
            history=state["history"], model_label=state["model_label"],
            allow_fallback=state["allow_fallback"])
        return {"answer": text, "answered_by": answered_by,
                "llm_calls": state["llm_calls"] + 1, "trace": ["generate"]}

    def route_after_retrieve(state):
        return "grade" if state["route"] == "multi" else "generate"

    def route_after_grade(state):
        if state["sufficient"]:
            return "generate"
        return "rewrite" if state["rewrites"] < MAX_REWRITES else "generate"

    g = StateGraph(RAGState)
    g.add_node("plan", plan)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("rewrite", rewrite)
    g.add_node("generate", generate)

    g.add_edge(START, "plan")
    g.add_edge("plan", "retrieve")
    g.add_conditional_edges("retrieve", route_after_retrieve, ["grade", "generate"])
    g.add_conditional_edges("grade", route_after_grade, ["rewrite", "generate"])
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)
    return g.compile()


_GRAPHS = {}


def get_graph(vectorstore):
    """Compile once per vector store and reuse it across requests."""
    key = id(vectorstore)
    if key not in _GRAPHS:
        _GRAPHS[key] = build_graph(vectorstore)
    return _GRAPHS[key]


def run_agentic(vectorstore, question, history=None, model_label=None, allow_fallback=True):
    """Run the agentic engine. Same return shape as answer.answer_question:
    (answer_text, docs, trace)."""
    start = time.perf_counter()
    final = get_graph(vectorstore).invoke({
        "question": question,
        "history": history or [],
        "model_label": model_label,
        "allow_fallback": allow_fallback,
        "route": "simple",
        "sub_queries": [],
        "docs": [],
        "sufficient": None,
        "missing": "",
        "rewrites": 0,
        "llm_calls": 0,
        "trace": [],
        "routing_models": [],
        "answer": None,
        "answered_by": None,
    }, config={"recursion_limit": 20})  # safety net above MAX_REWRITES, not the cap itself
    trace = {
        "engine": "agentic",
        "route": final["route"],
        "sub_queries": final["sub_queries"],
        "rewrites": final["rewrites"],
        "llm_calls": final["llm_calls"],
        "latency_ms": round((time.perf_counter() - start) * 1000),
        "answered_by": final["answered_by"],
        "routing_models": sorted(set(final["routing_models"])),
        "steps": final["trace"],
    }
    return final["answer"], final["docs"], trace


if __name__ == "__main__":
    # Manual check: `uv run python graph.py` prints the graph and answers one
    # multi-part question with its trace.
    from ingest import ingest

    store = ingest()
    print(get_graph(store).get_graph().draw_mermaid())
    q = "Compare the founding years of the Cricket World Cup and the T20 World Cup."
    text, docs, trace = run_agentic(store, q)
    print(f"\nQ: {q}\nA: {text}\nTrace: {json.dumps(trace, indent=2)}")
