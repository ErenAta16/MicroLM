"""Lightweight task loaders for evaluation prompts and gold answers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple


class TaskSample(NamedTuple):
    """A single task prompt with its ground-truth answer."""

    id: str
    prompt: str
    gold: str


_BUILTIN_ARITHMETIC: list[TaskSample] = [
    TaskSample(
        id="arith_001",
        prompt="Tom has 5 apples. He buys 3 more. How many apples does Tom have?",
        gold="8",
    ),
    TaskSample(
        id="arith_002",
        prompt="A store has 24 candies. They sell 9. How many remain?",
        gold="15",
    ),
    TaskSample(
        id="arith_003",
        prompt="Lisa runs 4 miles each day for 5 days. How many miles total?",
        gold="20",
    ),
    TaskSample(
        id="arith_004",
        prompt="There are 17 birds on a tree. 6 fly away. How many stay?",
        gold="11",
    ),
    TaskSample(
        id="arith_005",
        prompt="A baker makes 12 pies and sells 7. How many pies are left?",
        gold="5",
    ),
]

_BUILTIN_SYMBOLIC: list[TaskSample] = [
    TaskSample(
        id="sym_001",
        prompt="Simplify the expression: 2x + 3x",
        gold="5x",
    ),
    TaskSample(
        id="sym_002",
        prompt="Evaluate: 3 * (4 + 2)",
        gold="18",
    ),
    TaskSample(
        id="sym_003",
        prompt="What is the derivative of x^2?",
        gold="2x",
    ),
]


def load_arithmetic_samples() -> list[TaskSample]:
    """Return built-in multi-step arithmetic word problems with known answers.

    Returns:
        List of TaskSample with arithmetic prompts and gold answers.
    """
    return list(_BUILTIN_ARITHMETIC)


def load_symbolic_samples() -> list[TaskSample]:
    """Return a tiny built-in set of symbolic reasoning prompts.

    Returns:
        List of TaskSample with symbolic prompts and gold answers.
    """
    return list(_BUILTIN_SYMBOLIC)


def load_factual_mcq(path: str | Path | None = None) -> list[TaskSample]:
    """Load factual multiple-choice questions from a local JSONL file.

    Each JSONL line must contain ``id``, ``prompt``, and ``gold`` fields.
    If no path is given, returns a minimal built-in sample set.

    Args:
        path: Optional path to a JSONL file of MCQ samples.

    Returns:
        List of TaskSample with MCQ prompts and gold option letters.

    Raises:
        FileNotFoundError: If an explicit path is given but does not exist.
    """
    if path is None:
        return [
            TaskSample(
                id="mcq_001",
                prompt=(
                    "Who wrote 'Pride and Prejudice'? "
                    "A) Charles Dickens B) Jane Austen C) Mark Twain D) Emily Bronte"
                ),
                gold="B",
            ),
            TaskSample(
                id="mcq_002",
                prompt=(
                    "What is the capital of France? "
                    "A) Berlin B) Madrid C) Paris D) Rome"
                ),
                gold="C",
            ),
        ]

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"MCQ JSONL not found: {file_path}")

    samples: list[TaskSample] = []
    with file_path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            data = json.loads(stripped)
            samples.append(
                TaskSample(
                    id=data["id"],
                    prompt=data["prompt"],
                    gold=data["gold"],
                )
            )
    return samples
