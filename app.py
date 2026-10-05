"""
app.py

Gradio chat interface for Pitchwise. Builds the vector store once at
startup (ingest.py) and wires user messages through the retrieval +
generation pipeline (answer.py).

Uses gr.Blocks rather than gr.ChatInterface for full layout control —
specifically, to place the model-selection dropdown directly beside the
input box and to show which knowledge base chunks were used to ground
each answer, via a two-column layout (chat on the left, retrieved
context on the right).

Startup behavior: ingest() loads the saved vector index, or builds and
saves it when the knowledge base has changed (DECISIONS.md D-015). Embeddings
are local (D-002), so no embedding API key is needed.
"""

import gradio as gr
from ingest import ingest
from answer import answer_question, MODEL_OPTIONS, DEFAULT_MODEL_LABEL

print("Starting Pitchwise — building vector store...")
vectorstore = ingest()
print("Ready.")

# --- Cricket-themed styling ---
# Pitch green as the primary accent, warm red for highlights (the ball),
# stone/cream neutrals evoking a cricket pitch's natural tones.
CRICKET_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.green,
    secondary_hue=gr.themes.colors.red,
    neutral_hue=gr.themes.colors.stone,
    font=["Inter", "system-ui", "sans-serif"],
)

CUSTOM_CSS = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}
"""


def format_sources(docs):
    """Render the retrieved chunks as a readable sources panel, so the
    user can see exactly which knowledge base sections grounded the
    answer — not just trust it blindly."""
    if not docs:
        return "*No sources retrieved yet — ask a question to see them here.*"

    parts = ["### Sources used\n"]
    for doc in docs:
        header = doc.metadata.get("header_2") or doc.metadata.get("header_1", "Unknown section")
        doc_type = doc.metadata.get("doc_type", "")
        source_file = doc.metadata.get("source", "").split("/")[-1]
        parts.append(f"**{header}** · _{doc_type} / {source_file}_")
        parts.append(f"> {doc.page_content[:220]}{'...' if len(doc.page_content) > 220 else ''}")
        parts.append("")  # blank line between entries
    return "\n\n".join(parts)


ENGINE_CHOICES = {
    "Linear RAG (v1)": "linear",
    "Agentic RAG (v2, LangGraph)": "agentic",
}
DEFAULT_ENGINE_LABEL = "Linear RAG (v1)"


def format_usage(u):
    """'2,310 in / 85 out, $0.002188 at list price' (D-012)."""
    if not u:
        return "n/a"
    cost = "price unknown" if u.get("cost_usd") is None else f"${u['cost_usd']:.6f} at list price"
    failed = f", {u['failed_attempts']} failed attempt(s) before fallback" if u.get("failed_attempts") else ""
    return f"{u['input_tokens']:,} in / {u['output_tokens']:,} out, {cost}{failed}"


def format_trace(trace):
    """Render what the engine did for this answer: route, sub-queries,
    rewrites, LLM calls, latency, and which model answered (D-009)."""
    if not trace:
        return ""
    lines = [
        "### How this answer was produced\n",
        f"- **Engine:** {trace.get('engine')}",
        f"- **Route:** {trace.get('route') or 'n/a'}",
        f"- **Search queries:** {', '.join(trace.get('sub_queries') or [])}",
        f"- **Rewrites:** {trace.get('rewrites', 0)}",
        f"- **LLM calls:** {trace.get('llm_calls')}",
        f"- **Latency:** {trace.get('latency_ms')} ms",
        f"- **Answered by:** {trace.get('answered_by')}",
        f"- **Routing model:** {', '.join(trace.get('routing_models') or []) or 'n/a'}",
        f"- **Tokens:** {format_usage(trace.get('usage'))}",
        f"- **Steps:** {' → '.join(trace.get('steps') or [])}",
    ]
    return "\n".join(lines)


def respond(message, history, model_label, engine_label=DEFAULT_ENGINE_LABEL):
    """Handles one turn: retrieve + generate an answer, update the chat
    history, and return the sources panel content for the same turn."""
    engine = ENGINE_CHOICES.get(engine_label, "linear")
    trace = None
    try:
        answer_text, docs, trace = answer_question(vectorstore, message, history=history,
                                                   model_label=model_label, engine=engine)
    except RuntimeError as e:
        # Every provider in the fallback chain failed (DECISIONS.md D-008) —
        # this is the one case where we do surface an error, since silent
        # fallback has been fully exhausted at this point.
        answer_text = f"Sorry, all available models are currently unavailable. ({e})"
        docs = []

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer_text},
    ]
    sources = format_sources(docs)
    if trace:
        sources = sources + "\n\n" + format_trace(trace)
    return history, sources


def build_demo():
    """Build the Gradio UI. Exposed as a function so api.py can mount the
    same interface inside the FastAPI service."""
    with gr.Blocks(title="Pitchwise") as demo:
        gr.Markdown("# 🏏 Pitchwise\nA RAG-powered cricket assistant, grounded in a curated knowledge base — not general model knowledge.")

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Conversation", height=500)

                with gr.Row():
                    message_box = gr.Textbox(
                        placeholder="Ask about players, formats, tournaments, or the laws of cricket...",
                        show_label=False,
                        scale=4,
                    )
                    model_dropdown = gr.Dropdown(
                        choices=list(MODEL_OPTIONS.keys()),
                        value=DEFAULT_MODEL_LABEL,
                        label="Model",
                        scale=2,
                    )

                engine_radio = gr.Radio(
                    choices=list(ENGINE_CHOICES.keys()),
                    value=DEFAULT_ENGINE_LABEL,
                    label="Engine",
                )

                gr.Examples(
                    examples=[
                        "What is Don Bradman's exact career Test batting average?",
                        "Which player was named Player of the Match in the 2026 T20 World Cup final?",
                        "How many overs are bowled per day in a Test match?",
                        "Compare the founding years of the Cricket World Cup and the T20 World Cup.",
                    ],
                    inputs=message_box,
                )

            with gr.Column(scale=1):
                sources_panel = gr.Markdown(
                    value="*No sources retrieved yet — ask a question to see them here.*",
                    label="Sources",
                )

        def submit_and_clear(message, history, model_label, engine_label):
            new_history, sources = respond(message, history, model_label, engine_label)
            return new_history, sources, ""  # clear the textbox after submit

        message_box.submit(
            submit_and_clear,
            inputs=[message_box, chatbot, model_dropdown, engine_radio],
            outputs=[chatbot, sources_panel, message_box],
        )

    return demo


if __name__ == "__main__":
    build_demo().launch(theme=CRICKET_THEME, css=CUSTOM_CSS)
