"""
evaluation/test.py

Defines the TestQuestion schema and loads tests.jsonl.
"""

import json
from pathlib import Path
from pydantic import BaseModel, Field

TEST_FILE = str(Path(__file__).parent / "tests.jsonl")


class TestQuestion(BaseModel):
    """A test question with expected keywords and reference answer."""

    question: str = Field(description="The question to ask the RAG system")
    keywords: list[str] = Field(description="Keywords that must appear in retrieved context")
    reference_answer: str = Field(description="The reference answer for this question")
    category: str = Field(description="Question category: direct_fact, spanning, temporal, out_of_scope")


def load_tests() -> list[TestQuestion]:
    """Load test questions from tests.jsonl."""
    tests = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                tests.append(TestQuestion(**data))
    return tests