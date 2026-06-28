"""Unit tests for validated generation recipe prompt formats."""

from __future__ import annotations

from hollow_chains.data.sft_format import (
    REASONING_SYSTEM,
    format_instruct_prompt,
    format_reasoning_eval_prompt,
)
from hollow_chains.metrics.parse import TAG_BEGIN_THOUGHT


def test_reasoning_eval_prompt_exact() -> None:
    question = "What is 2 + 3?"
    expected = (
        f"[SYSTEM]: {REASONING_SYSTEM}\n\n"
        f"[USER]: {question}\n\n"
        f"[ASSISTANT]: {TAG_BEGIN_THOUGHT}\n"
    )
    assert format_reasoning_eval_prompt(question) == expected


def test_instruct_eval_prompt_exact() -> None:
    question = "What is 2 + 3?"
    expected = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{question}\n\n### Response:\n"
    )
    assert format_instruct_prompt(question) == expected
