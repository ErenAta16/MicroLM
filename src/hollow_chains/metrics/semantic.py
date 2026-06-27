"""Semantic Correctness metrics for reasoning traces."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from hollow_chains.data.schema import GenerationRecord
from hollow_chains.metrics.parse import ParsedTrace, parse_trace

# Default weights for composite semantic correctness.
DEFAULT_SC_WEIGHTS: dict[str, float] = {
    "answer_accuracy": 0.70,
    "step_validity": 0.30,
}


class SymbolicRuleChecker(Protocol):
    """Pluggable interface for symbolic step/rule validation."""

    def check(self, expression: str, gold: str) -> bool:
        """Return True if the expression is valid relative to gold."""
        ...


class TrivialSymbolicChecker:
    """Default symbolic checker: exact normalized match of extracted token."""

    def check(self, expression: str, gold: str) -> bool:
        """Check if expression matches gold after normalization.

        Args:
            expression: Extracted symbolic expression.
            gold: Ground-truth answer.

        Returns:
            True if normalized strings match.
        """
        return _normalize_answer(expression) == _normalize_answer(gold)


def _normalize_answer(text: str) -> str:
    """Normalize an answer string for comparison."""
    cleaned = text.strip().lower()
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned


_ARITH_EQUATION_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)"
)

_MCQ_OPTION_RE = re.compile(r"(?:^|\s)([A-Da-d])(?:\)|\.|:|\s|$)")

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_answer(trace: ParsedTrace, task_type: str) -> str:
    """Pull the final answer from the solution block.

    Robust to missing solution blocks — returns empty string on failure.

    Args:
        trace: Parsed trace with solution block.
        task_type: One of arithmetic, symbolic, factual_mcq.

    Returns:
        Extracted answer string (may be empty).
    """
    solution = trace.solution.strip()
    if not solution:
        return ""

    if task_type == "arithmetic":
        numbers = _NUMBER_RE.findall(solution)
        if numbers:
            return numbers[-1]
        return ""

    if task_type == "factual_mcq":
        # Prefer explicit "answer is X" patterns
        answer_is = re.search(
            r"(?:answer|option)\s*(?:is|:)\s*([A-Da-d])",
            solution,
            re.IGNORECASE,
        )
        if answer_is:
            return answer_is.group(1).upper()
        matches = _MCQ_OPTION_RE.findall(solution)
        if matches:
            return matches[-1].upper()
        return ""

    if task_type == "symbolic":
        # Take last alphanumeric token sequence as canonical answer
        tokens = re.findall(r"[a-zA-Z0-9]+", solution)
        if tokens:
            return tokens[-1]
        return ""

    return ""


@dataclass
class AnswerAccuracyResult:
    """Result of answer accuracy check."""

    score: float = 0.0
    extracted: str = ""
    gold: str = ""


def answer_accuracy(record: GenerationRecord) -> AnswerAccuracyResult:
    """Normalized exact match of extracted answer vs gold.

    Args:
        record: Generation record with gold answer.

    Returns:
        AnswerAccuracyResult with score in {0.0, 1.0}.
    """
    trace = parse_trace(record.generation)
    extracted = extract_answer(trace, record.task_type)
    gold_norm = _normalize_answer(record.gold)
    extracted_norm = _normalize_answer(extracted)
    score = 1.0 if extracted_norm == gold_norm and gold_norm else 0.0
    return AnswerAccuracyResult(score=score, extracted=extracted, gold=record.gold)


@dataclass
class StepValidityResult:
    """Result of step-level validity checking."""

    score: float | None = None
    valid_count: int = 0
    total_count: int = 0


def _eval_arithmetic(a: float, op: str, b: float) -> float:
    """Evaluate a simple binary arithmetic operation."""
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        return a / b if b != 0 else float("inf")
    return float("nan")


def step_validity(
    record: GenerationRecord,
    symbolic_checker: SymbolicRuleChecker | None = None,
) -> StepValidityResult:
    """Verify step-level validity in the think block.

    For arithmetic: parse ``a op b = c`` equations and verify numerically.
    For symbolic: pluggable rule checker (trivial default).
    For factual_mcq: returns None score (skipped).

    Args:
        record: Generation record.
        symbolic_checker: Optional custom symbolic checker.

    Returns:
        StepValidityResult with fraction valid and counts.
    """
    if record.task_type == "factual_mcq":
        return StepValidityResult(score=None, valid_count=0, total_count=0)

    trace = parse_trace(record.generation)
    think = trace.think

    if record.task_type == "arithmetic":
        equations = _ARITH_EQUATION_RE.findall(think)
        if not equations:
            return StepValidityResult(score=0.0, valid_count=0, total_count=0)

        valid = 0
        for a_str, op, b_str, c_str in equations:
            a, b, c = float(a_str), float(b_str), float(c_str)
            expected = _eval_arithmetic(a, op, b)
            if abs(expected - c) < 1e-6:
                valid += 1
        total = len(equations)
        score = valid / total if total > 0 else 0.0
        return StepValidityResult(score=score, valid_count=valid, total_count=total)

    if record.task_type == "symbolic":
        checker = symbolic_checker or TrivialSymbolicChecker()
        # Extract simple expressions from think block
        expressions = re.findall(r"[0-9x+\-*/()]+", think)
        if not expressions:
            return StepValidityResult(score=0.0, valid_count=0, total_count=0)
        valid = sum(1 for expr in expressions if checker.check(expr, record.gold))
        total = len(expressions)
        score = valid / total if total > 0 else 0.0
        return StepValidityResult(score=score, valid_count=valid, total_count=total)

    return StepValidityResult(score=None, valid_count=0, total_count=0)


@dataclass
class SCResult:
    """Composite Semantic Correctness result with all sub-components."""

    score: float = 0.0
    answer_accuracy_mean: float = 0.0
    step_validity_mean: float | None = None
    per_record_answer_accuracy: list[float] = field(default_factory=list)
    per_record_step_validity: list[float | None] = field(default_factory=list)
    weights_used: dict[str, float] = field(default_factory=dict)


def sample_semantic_correctness(
    record: GenerationRecord,
    symbolic_checker: SymbolicRuleChecker | None = None,
) -> float:
    """Per-record semantic correctness scalar.

    Combines answer accuracy and step validity (when applicable).

    Args:
        record: Single generation record.
        symbolic_checker: Optional symbolic checker.

    Returns:
        Sample-level SC scalar in [0, 1].
    """
    aa = answer_accuracy(record).score
    sv = step_validity(record, symbolic_checker)
    if sv.score is None:
        return aa
    return 0.7 * aa + 0.3 * sv.score


def semantic_correctness(
    records: list[GenerationRecord],
    *,
    weights: dict[str, float] | None = None,
    symbolic_checker: SymbolicRuleChecker | None = None,
) -> SCResult:
    """Compute composite Semantic Correctness and all sub-components.

    Composite is a weighted mean of answer_accuracy and step_validity.
    For factual_mcq, step_validity is excluded from the composite.
    Default weights are documented in ``DEFAULT_SC_WEIGHTS``.

    Args:
        records: Generation records to evaluate.
        weights: Optional weight overrides.
        symbolic_checker: Optional symbolic rule checker.

    Returns:
        SCResult with composite score and sub-components.
    """
    w = dict(DEFAULT_SC_WEIGHTS)
    if weights:
        w.update(weights)

    aa_scores = [answer_accuracy(r).score for r in records]
    aa_mean = sum(aa_scores) / len(aa_scores) if aa_scores else 0.0

    sv_results = [step_validity(r, symbolic_checker) for r in records]
    sv_values = [sv.score for sv in sv_results if sv.score is not None]
    sv_mean: float | None = sum(sv_values) / len(sv_values) if sv_values else None

    components: dict[str, float] = {"answer_accuracy": aa_mean}
    if sv_mean is not None:
        components["step_validity"] = sv_mean

    active_weights = {k: w[k] for k in components if k in w}
    weight_sum = sum(active_weights.values())
    if weight_sum == 0:
        score = 0.0
    else:
        score = (
            sum(components[k] * active_weights[k] for k in active_weights) / weight_sum
        )

    return SCResult(
        score=score,
        answer_accuracy_mean=aa_mean,
        step_validity_mean=sv_mean,
        per_record_answer_accuracy=aa_scores,
        per_record_step_validity=[sv.score for sv in sv_results],
        weights_used=active_weights,
    )
