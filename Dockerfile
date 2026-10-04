# Pitchwise: FastAPI + Gradio in one container (DECISIONS.md D-011).
# Build:  docker build -t pitchwise .
# Run:    docker run -p 7860:7860 --env-file .env pitchwise
# API keys are passed at run time only; they are never copied into the image.

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN pip install --no-cache-dir uv

# Dependencies first, so code changes don't invalidate this (slow) layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Bake the embedding model into the image so a cold start doesn't download it (D-002).
RUN uv run --no-sync python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

# Run as a non-root user (also what Hugging Face Spaces expects).
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uv", "run", "--no-sync", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
