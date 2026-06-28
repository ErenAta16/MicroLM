"""Validated reasoning SFT / eval prompt formats (shared between train and eval)."""

from __future__ import annotations

from hollow_chains.metrics.parse import (
    TAG_BEGIN_SOLUTION,
    TAG_BEGIN_THOUGHT,
    TAG_END_SOLUTION,
    TAG_END_THOUGHT,
)

REASONING_SYSTEM = (
    "Your role as an assistant involves thoroughly exploring questions through "
    "a systematic long thinking process before providing the final precise and "
    "accurate solutions."
)

INSTRUCT_ALPACA_PREFIX = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
)


def format_reasoning_eval_prompt(question: str) -> str:
    """SupraLabs *-Reasoning eval / SFT prompt (open thought tag in prompt)."""
    return (
        f"[SYSTEM]: {REASONING_SYSTEM}\n\n"
        f"[USER]: {question}\n\n"
        f"[ASSISTANT]: {TAG_BEGIN_THOUGHT}\n"
    )


def format_sft_prompt(question: str) -> str:
    """Masked SFT prompt — matches eval reasoning format."""
    return format_reasoning_eval_prompt(question)


def format_sft_target(thought: str, answer: str, eos: str = "") -> str:
    """SFT completion target (loss applied here)."""
    return (
        f"{thought}\n"
        f"{TAG_END_THOUGHT}{TAG_BEGIN_SOLUTION}\n"
        f"{answer}\n"
        f"{TAG_END_SOLUTION}{eos}"
    )


def format_instruct_prompt(question: str) -> str:
    """SupraLabs *-Instruct Alpaca prompt."""
    return f"{INSTRUCT_ALPACA_PREFIX}{question}\n\n### Response:\n"


def schema_trace_from_parts(thought: str, answer: str) -> str:
    """Build a well-formed M1 trace from thought and answer text."""
    return (
        f"{TAG_BEGIN_THOUGHT} {thought} {TAG_END_THOUGHT} "
        f"{TAG_BEGIN_SOLUTION} {answer} {TAG_END_SOLUTION}"
    )
