"""
api.py

FastAPI service for Pitchwise (DECISIONS.md D-011). One process serves:

  POST /ask     JSON API: answer a question with either engine
  GET  /health  liveness + proof the vector store loaded
  GET  /docs    auto-generated OpenAPI docs (FastAPI built-in)
  /             the Gradio chat UI from app.py, mounted on the same app

Run locally:   uv run uvicorn api:app --port 7860
In Docker:     see Dockerfile (same command, host 0.0.0.0)

The vector store is loaded once, when app.py is imported (D-015: the saved
index is reused until the knowledge base changes), and shared by the API
and the UI.

Endpoints are plain `def`, not `async def`: the LLM SDK calls are blocking,
and FastAPI runs `def` endpoints in a thread pool. Blocking calls inside an
`async def` endpoint would stall the event loop for every other request.
"""

from contextlib import asynccontextmanager
from typing import Literal, Optional

import gradio as gr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import app as ui  # builds the vector store on import and exposes build_demo()
import observability as obs
from answer import MODEL_OPTIONS, answer_question

MAX_QUESTION_CHARS = 500  # protects free-tier API quotas on a public demo


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    history: list[Turn] = Field(default_factory=list)
    engine: Literal["linear", "agentic"] = "linear"
    model_label: Optional[str] = None


class Source(BaseModel):
    source: str
    header: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    trace: dict


@asynccontextmanager
async def lifespan(_app):
    yield
    obs.flush()  # send any buffered Langfuse traces before the process exits (D-013)


api = FastAPI(
    lifespan=lifespan,
    title="Pitchwise API",
    description="Grounded cricket Q&A with a linear RAG engine and a LangGraph agentic engine.",
    version="2.0.0",
)


@api.get("/health")
def health():
    return {
        "status": "ok",
        "chunks": ui.vectorstore._collection.count(),
        "engines": ["linear", "agentic"],
        "models": list(MODEL_OPTIONS.keys()),
        "tracing": "langfuse" if obs.ENABLED else "off",
    }


@api.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")
    if req.model_label is not None and req.model_label not in MODEL_OPTIONS:
        raise HTTPException(status_code=400, detail=f"Unknown model_label. Choose one of: {list(MODEL_OPTIONS)}")

    try:
        answer, docs, trace = answer_question(
            ui.vectorstore,
            req.question,
            history=[t.model_dump() for t in req.history],
            model_label=req.model_label,
            engine=req.engine,
        )
    except RuntimeError as e:
        # Every provider in the fallback chain failed (D-008).
        raise HTTPException(status_code=503, detail=f"All model providers are unavailable: {e}")

    sources = [
        Source(
            source=d.metadata.get("source", "").split("/")[-1],
            header=d.metadata.get("header_2") or d.metadata.get("header_1") or "",
            snippet=d.page_content[:220],
        )
        for d in docs
    ]
    return AskResponse(answer=answer, sources=sources, trace=trace)


# Mount the chat UI last, so the API routes above take precedence over "/".
app = gr.mount_gradio_app(api, ui.build_demo(), path="/", theme=ui.CRICKET_THEME, css=ui.CUSTOM_CSS)
