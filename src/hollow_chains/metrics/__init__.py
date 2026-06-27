"""Metrics layer for hollow-chains."""

from hollow_chains.metrics.gap import (
    GapReport,
    four_way_classify,
    fsg,
    gap_report,
    theater_score,
)
from hollow_chains.metrics.parse import ParsedTrace, parse_trace
from hollow_chains.metrics.semantic import SCResult, semantic_correctness
from hollow_chains.metrics.structural import SFResult, structural_fidelity

__all__ = [
    "GapReport",
    "ParsedTrace",
    "SCResult",
    "SFResult",
    "fsg",
    "four_way_classify",
    "gap_report",
    "parse_trace",
    "semantic_correctness",
    "structural_fidelity",
    "theater_score",
]
