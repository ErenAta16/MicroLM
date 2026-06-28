"""Unit tests for Qwen teacher output mapping and correctness filter."""

from __future__ import annotations

from hollow_chains.data.teacher_hf import is_correct, map_qwen_output, qwen_to_schema
from hollow_chains.metrics.parse import parse_trace
from hollow_chains.metrics.semantic import extract_answer


def test_map_qwen_output_with_think_tags() -> None:
    raw = "<think>2+3=5</think> The answer is 5."
    thought, answer = map_qwen_output(raw)
    assert thought == "2+3=5"
    assert answer == "The answer is 5."


def test_map_qwen_output_without_think_tags() -> None:
    raw = "2 plus 3 is 5\nThe answer is 5."
    thought, answer = map_qwen_output(raw)
    assert thought == raw
    assert answer == raw


def test_map_qwen_output_strips_redacted_im_end() -> None:
    raw = "<think>step</think> 42<|im_end|>"
    thought, answer = map_qwen_output(raw)
    assert thought == "step"
    assert answer == "42"


def test_is_correct_numeric_robust() -> None:
    assert is_correct("The answer is 5.", "5")
    assert is_correct("The answer is 5.", "5.")
    assert not is_correct("The answer is 6.", "5")


def test_qwen_to_schema_well_formed_and_correct() -> None:
    raw = "<think>2+3=5</think> The answer is 5."
    trace = qwen_to_schema(raw, gold="5", task_type="arithmetic")
    assert trace is not None

    parsed = parse_trace(trace)
    assert parsed.well_formed is True
    assert extract_answer(parsed, "arithmetic") == "5"
    assert is_correct(raw, "5")
