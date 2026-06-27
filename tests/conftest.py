"""Shared test fixtures for hollow-chains metrics."""

from __future__ import annotations

import pytest

from hollow_chains.data.schema import GenerationRecord

TAGS = (
    "<|begin_of_thought|>",
    "<|end_of_thought|>",
    "<|begin_of_solution|>",
    "<|end_of_solution|>",
)


def _wrap(think: str, solution: str) -> str:
    return f"{TAGS[0]} {think} {TAGS[1]} " f"{TAGS[2]} {solution} {TAGS[3]}"


@pytest.fixture
def coherent_correct_record() -> GenerationRecord:
    """Well-formed trace with correct arithmetic answer."""
    think = (
        "Okay the user wants to know how many apples Tom has. "
        "Tom has 5 apples and buys 3 more. 5 + 3 = 8"
    )
    solution = "Tom has 8 apples in total."
    return GenerationRecord(
        id="coherent_correct",
        prompt="Tom has 5 apples. He buys 3 more. How many?",
        task_type="arithmetic",
        gold="8",
        generation=_wrap(think, solution),
    )


@pytest.fixture
def coherent_wrong_record() -> GenerationRecord:
    """Well-formed theater trace with wrong factual content."""
    think = (
        "Okay the user wants to know when AI was introduced. "
        "Let me start by recalling that AI was first introduced at MIT in 1965."
    )
    solution = "AI was first introduced at MIT in 1965."
    return GenerationRecord(
        id="coherent_wrong",
        prompt="When was AI first introduced?",
        task_type="factual_mcq",
        gold="B",
        generation=_wrap(think, solution),
    )


@pytest.fixture
def malformed_record() -> GenerationRecord:
    """Trace missing end_of_solution tag."""
    think = "Some reasoning without proper closure."
    generation = f"{TAGS[0]} {think} {TAGS[1]} " f"{TAGS[2]} The answer is 42."
    return GenerationRecord(
        id="malformed",
        prompt="What is 6 times 7?",
        task_type="arithmetic",
        gold="42",
        generation=generation,
    )


@pytest.fixture
def repetitive_record() -> GenerationRecord:
    """Degenerate repetitive trace."""
    think = "the the the the the the the the the the"
    solution = "the the the the the the the the"
    return GenerationRecord(
        id="repetitive",
        prompt="Say something unique.",
        task_type="symbolic",
        gold="unique",
        generation=_wrap(think, solution),
    )


@pytest.fixture
def arithmetic_wrong_step_record() -> GenerationRecord:
    """Arithmetic trace with one invalid intermediate equation."""
    think = (
        "Okay the user wants the total. " "First 5 + 3 = 9. Then we confirm 5 + 3 = 8."
    )
    solution = "The answer is 8."
    return GenerationRecord(
        id="arith_wrong_step",
        prompt="What is 5 + 3?",
        task_type="arithmetic",
        gold="8",
        generation=_wrap(think, solution),
    )


@pytest.fixture
def metrics_config() -> dict:
    """Minimal metrics config matching configs/metrics.yaml."""
    return {
        "tags": {
            "begin_thought": TAGS[0],
            "end_thought": TAGS[1],
            "begin_solution": TAGS[2],
            "end_solution": TAGS[3],
        },
        "structural_fidelity": {
            "weights": {
                "parse_rate": 0.15,
                "tag_validity": 0.20,
                "section_ratio_conformity": 0.15,
                "length_conformity": 0.15,
                "template_ngram_overlap": 0.15,
                "entropy_profile": 0.10,
                "repetition": 0.10,
            },
            "ngram_n": 3,
        },
        "semantic_correctness": {
            "weights": {
                "answer_accuracy": 0.70,
                "step_validity": 0.30,
            },
        },
        "gap": {
            "tau_high": 0.8,
            "tau_low": 0.2,
        },
        "teacher_opening_templates": [
            "okay the user wants",
            "let me start by recalling",
            "so the key points are",
        ],
        "reference": {
            "section_ratios": [0.6, 0.65, 0.7],
            "lengths": [20.0, 30.0, 40.0],
        },
    }
