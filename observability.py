"""
observability.py

Optional request tracing with Langfuse (DECISIONS.md D-013).

When LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set, every question
becomes one Langfuse trace: a span for the request, a span per graph node
(plan, retrieve, grade, rewrite, generate) and a generation per LLM call
with its model, prompt, output, token usage and cost. Without the keys,
everything here is a no-op and Langfuse is never imported, so the app,
the tests and CI run exactly as before with no extra dependency at runtime.

Token and cost accounting (usage.py) does not depend on this module: the
numbers in each answer's trace and in every evaluation row are recorded
whether or not Langfuse is on.
"""

import os
from contextlib import nullcontext

from dotenv import load_dotenv

load_dotenv(override=True)

# PITCHWISE_TRACING=off forces tracing off even when keys are present
# (the test suite sets it, so tests never send traces).
ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
               and os.getenv("PITCHWISE_TRACING", "on").lower() != "off")

if ENABLED:
    from langfuse import get_client
    from langfuse import observe as _observe
    from langfuse import propagate_attributes as _propagate_attributes


def observe(name=None, as_type=None, capture_input=False, capture_output=False):
    """Decorator: a Langfuse span or generation around the function, or the
    function unchanged when tracing is off. Inputs and outputs are not
    captured automatically (arguments include the vector store); callers set
    what matters with update_span / update_generation."""
    if not ENABLED:
        return lambda func: func
    return _observe(name=name, as_type=as_type,
                    capture_input=capture_input, capture_output=capture_output)


def update_span(**fields):
    if ENABLED:
        get_client().update_current_span(**fields)


def update_generation(**fields):
    if ENABLED:
        get_client().update_current_generation(**fields)


def trace_attributes(**fields):
    """Context manager that sets trace-level fields (trace_name, tags,
    metadata, session_id) for everything inside it."""
    if not ENABLED:
        return nullcontext()
    return _propagate_attributes(**fields)


def flush():
    """Send buffered traces. Called at the end of batch jobs (evaluation,
    mechanism check) and on API shutdown; the SDK also flushes in the
    background."""
    if ENABLED:
        get_client().flush()
