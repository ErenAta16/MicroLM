"""Trace parsing for reasoning-tag structure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Default tag set — swap via module constant for alternate formats.
TAG_BEGIN_THOUGHT = "<|begin_of_thought|>"
TAG_END_THOUGHT = "<|end_of_thought|>"
TAG_BEGIN_SOLUTION = "<|begin_of_solution|>"
TAG_END_SOLUTION = "<|end_of_solution|>"

TAG_SET: dict[str, str] = {
    "begin_thought": TAG_BEGIN_THOUGHT,
    "end_thought": TAG_END_THOUGHT,
    "begin_solution": TAG_BEGIN_SOLUTION,
    "end_solution": TAG_END_SOLUTION,
}

EXPECTED_TAG_ORDER: tuple[str, ...] = (
    "begin_thought",
    "end_thought",
    "begin_solution",
    "end_solution",
)


@dataclass
class ParsedTrace:
    """Structured parse result for a raw generation string."""

    raw: str
    think: str = ""
    solution: str = ""
    well_formed: bool = False
    tag_present: dict[str, bool] = field(
        default_factory=lambda: {k: False for k in EXPECTED_TAG_ORDER}
    )
    tag_counts: dict[str, int] = field(
        default_factory=lambda: {k: 0 for k in EXPECTED_TAG_ORDER}
    )
    tag_order: list[str] = field(default_factory=list)
    tags_unique: bool = False
    tags_properly_closed: bool = False
    parse_errors: list[str] = field(default_factory=list)


def _find_tag_positions(text: str, tag: str) -> list[int]:
    """Return start indices of all occurrences of ``tag`` in ``text``."""
    positions: list[int] = []
    start = 0
    while True:
        idx = text.find(tag, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + len(tag)
    return positions


def parse_trace(
    generation: str,
    tags: dict[str, str] | None = None,
) -> ParsedTrace:
    """Parse a raw generation into structured trace components.

    Never raises on malformed input; returns structured failure info instead.

    Args:
        generation: Full raw model output including tags.
        tags: Optional tag mapping (keys: begin_thought, end_thought, etc.).

    Returns:
        ParsedTrace with extracted blocks, presence flags, and well_formed flag.
    """
    tag_map = tags or TAG_SET
    result = ParsedTrace(raw=generation)

    for key in EXPECTED_TAG_ORDER:
        tag_str = tag_map[key]
        positions = _find_tag_positions(generation, tag_str)
        result.tag_counts[key] = len(positions)
        result.tag_present[key] = len(positions) > 0
        for _pos in positions:
            result.tag_order.append(key)

    # Check uniqueness (each tag exactly once)
    result.tags_unique = all(c == 1 for c in result.tag_counts.values())

    # Check order
    order_correct = result.tag_order == list(EXPECTED_TAG_ORDER)

    # Extract think and solution blocks
    bt = tag_map["begin_thought"]
    et = tag_map["end_thought"]
    bs = tag_map["begin_solution"]
    es = tag_map["end_solution"]

    think_start = generation.find(bt)
    think_end = generation.find(et)
    sol_start = generation.find(bs)
    sol_end = generation.find(es)

    if think_start != -1 and think_end != -1 and think_end > think_start:
        result.think = generation[think_start + len(bt) : think_end].strip()
    elif think_start != -1:
        result.parse_errors.append("end_thought missing or before begin_thought")

    if sol_start != -1 and sol_end != -1 and sol_end > sol_start:
        result.solution = generation[sol_start + len(bs) : sol_end].strip()
    elif sol_start != -1:
        result.parse_errors.append("end_solution missing or before begin_solution")

    # Proper closure: each opening tag has a matching closing tag after it
    closure_ok = True
    if result.tag_present["begin_thought"] and not result.tag_present["end_thought"]:
        closure_ok = False
        result.parse_errors.append("begin_thought without end_thought")
    if result.tag_present["begin_solution"] and not result.tag_present["end_solution"]:
        closure_ok = False
        result.parse_errors.append("begin_solution without end_solution")
    if (
        result.tag_present["end_thought"]
        and think_start != -1
        and think_end != -1
        and think_end <= think_start
    ):
        closure_ok = False
    if (
        result.tag_present["end_solution"]
        and sol_start != -1
        and sol_end != -1
        and sol_end <= sol_start
    ):
        closure_ok = False

    result.tags_properly_closed = closure_ok

    all_present = all(result.tag_present[k] for k in EXPECTED_TAG_ORDER)
    result.well_formed = (
        all_present
        and result.tags_unique
        and order_correct
        and result.tags_properly_closed
    )

    if not all_present:
        missing = [k for k in EXPECTED_TAG_ORDER if not result.tag_present[k]]
        result.parse_errors.append(f"missing tags: {', '.join(missing)}")
    if not result.tags_unique:
        dupes = [k for k, c in result.tag_counts.items() if c > 1]
        if dupes:
            result.parse_errors.append(f"duplicate tags: {', '.join(dupes)}")
    if not order_correct and result.tag_order:
        result.parse_errors.append("incorrect tag order")

    return result


def tags_from_config(config: dict[str, Any]) -> dict[str, str]:
    """Build a tag mapping dict from a metrics config section.

    Args:
        config: Full metrics config dict (expects ``tags`` sub-dict).

    Returns:
        Tag mapping compatible with ``parse_trace``.
    """
    tag_section = config.get("tags", {})
    return {
        "begin_thought": tag_section.get("begin_thought", TAG_BEGIN_THOUGHT),
        "end_thought": tag_section.get("end_thought", TAG_END_THOUGHT),
        "begin_solution": tag_section.get("begin_solution", TAG_BEGIN_SOLUTION),
        "end_solution": tag_section.get("end_solution", TAG_END_SOLUTION),
    }
