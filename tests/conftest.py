"""Shared test setup: tests never send Langfuse traces, even when a
developer's .env holds Langfuse keys (observability.py, D-013)."""

import os

os.environ["PITCHWISE_TRACING"] = "off"
