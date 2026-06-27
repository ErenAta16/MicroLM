"""Unit tests for Qwen teacher output → M1 schema mapping."""

from __future__ import annotations

from hollow_chains.data.teacher_hf import qwen_to_schema
from hollow_chains.metrics.parse import parse_trace
from hollow_chains.metrics.semantic import extract_answer


def test_qwen_to_schema_well_formed_and_correct() -> None:
    """Canned Qwen-style output maps to a valid trace with matching gold."""
    raw = "<think>2 plus 3 is 5</think> " "The answer is 5."
    trace = qwen_to_schema(raw, gold="5", task_type="arithmetic")
    assert trace is not None

    parsed = parse_trace(trace)
    assert parsed.well_formed is True
    assert extract_answer(parsed, "arithmetic") == "5"


def test_qwen_to_schema_think_tag_variant() -> None:
    """Native Qwen think tags are also accepted."""
    raw = "2 plus 3 is 5\nThe answer is 5."
    trace = qwen_to_schema(raw, gold="5", task_type="arithmetic")
    assert trace is not None
    parsed = parse_trace(trace)
    assert parsed.well_formed is True
    assert extract_answer(parsed, "arithmetic") == "5"
