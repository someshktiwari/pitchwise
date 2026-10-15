---
title: Pitchwise
emoji: 🏏
colorFrom: green
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Grounded cricket Q&A — linear RAG vs a LangGraph agent
---

# 🏏 Pitchwise

A cricket assistant that answers only from its own knowledge base (17
curated documents and 133 Wikipedia articles), with a linear RAG engine and
a LangGraph agentic engine you can switch between, and the retrieved
sources and a trace of each answer shown beside it.

This Space runs the Docker image from the GitHub repository; the code,
design decisions and evaluation results are there:
**https://github.com/someshktiwari/pitchwise**

- Chat UI: this page
- JSON API: `POST /ask`, `GET /health`, OpenAPI docs at `/docs`

The demo answers on free-tier model APIs, so questions are rate-limited per
visitor and per day. If a limit is reached, run it locally from the repo.
