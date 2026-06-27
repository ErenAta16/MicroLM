"""Form-Substance Gap metrics: FSG, theater score, four-way classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from hollow_chains.data.schema import GenerationRecord
from hollow_chains.metrics.parse import tags_from_config
from hollow_chains.metrics.semantic import (
    SCResult,
    sample_semantic_correctness,
    semantic_correctness,
)
from hollow_chains.metrics.structural import (
    SFResult,
    reference_from_config,
    sample_structural_fidelity,
    structural_fidelity,
)

FourWayLabel = Literal[
    "coherent_correct",
    "coherent_wrong",
    "malformed_correct",
    "malformed_wrong",
]


def fsg(sf: float, sc: float) -> float:
    """Form-Substance Gap: structural fidelity minus semantic correctness.

    Args:
        sf: Structural fidelity scalar in [0, 1].
        sc: Semantic correctness scalar in [0, 1].

    Returns:
        FSG value (positive = theater tendency).
    """
    return sf - sc


def sample_sf(
    record: GenerationRecord,
    *,
    reference: dict[str, Any],
    tags: dict[str, str] | None = None,
    templates: list[str] | None = None,
    ngram_n: int = 3,
) -> float:
    """Per-record structural fidelity scalar.

    Args:
        record: Single generation record.
        reference: Reference distributions dict.
        tags: Optional custom tag mapping.
        templates: Teacher opening templates.
        ngram_n: N-gram size.

    Returns:
        Sample-level SF in [0, 1].
    """
    return sample_structural_fidelity(
        record,
        reference=reference,
        tags=tags,
        templates=templates,
        ngram_n=ngram_n,
    )


def sample_sc(
    record: GenerationRecord,
) -> float:
    """Per-record semantic correctness scalar.

    Args:
        record: Single generation record.

    Returns:
        Sample-level SC in [0, 1].
    """
    return sample_semantic_correctness(record)


def four_way_classify(
    record: GenerationRecord,
    *,
    tau_high: float = 0.8,
    tau_low: float = 0.2,
    sf: float | None = None,
    sc: float | None = None,
    reference: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    templates: list[str] | None = None,
) -> FourWayLabel:
    """Classify a record into one of four coherence/correctness quadrants.

    Args:
        record: Generation record.
        tau_high: High SF threshold for "coherent".
        tau_low: Low SC threshold for "wrong".
        sf: Precomputed SF (computed if None).
        sc: Precomputed SC (computed if None).
        reference: Reference dict for SF computation.
        tags: Optional tag mapping.
        templates: Teacher templates.

    Returns:
        One of coherent_correct, coherent_wrong, malformed_correct, malformed_wrong.
    """
    ref = reference or {}
    sf_val = (
        sf
        if sf is not None
        else sample_sf(record, reference=ref, tags=tags, templates=templates)
    )
    sc_val = sc if sc is not None else sample_sc(record)

    coherent = sf_val >= tau_high
    correct = sc_val > tau_low

    if coherent and correct:
        return "coherent_correct"
    if coherent and not correct:
        return "coherent_wrong"
    if not coherent and correct:
        return "malformed_correct"
    return "malformed_wrong"


def theater_score(
    records: list[GenerationRecord],
    *,
    tau_high: float = 0.8,
    tau_low: float = 0.2,
    reference: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    templates: list[str] | None = None,
) -> float:
    """Fraction of records with SF >= tau_high AND SC <= tau_low.

    Args:
        records: Generation records.
        tau_high: High SF threshold.
        tau_low: Low SC threshold.
        reference: Reference dict for per-record SF.
        tags: Optional tag mapping.
        templates: Teacher templates.

    Returns:
        Theater score in [0, 1].
    """
    if not records:
        return 0.0
    ref = reference or {}
    theater_count = 0
    for r in records:
        sf_val = sample_sf(r, reference=ref, tags=tags, templates=templates)
        sc_val = sample_sc(r)
        if sf_val >= tau_high and sc_val <= tau_low:
            theater_count += 1
    return theater_count / len(records)


@dataclass
class GapReport:
    """Aggregate gap metrics report with per-record scatter data."""

    fsg: float = 0.0
    theater_score: float = 0.0
    sf_aggregate: float = 0.0
    sc_aggregate: float = 0.0
    four_way_counts: dict[str, int] = field(
        default_factory=lambda: {
            "coherent_correct": 0,
            "coherent_wrong": 0,
            "malformed_correct": 0,
            "malformed_wrong": 0,
        }
    )
    per_record_sf: list[float] = field(default_factory=list)
    per_record_sc: list[float] = field(default_factory=list)
    sf_result: SFResult | None = None
    sc_result: SCResult | None = None
    tau_high: float = 0.8
    tau_low: float = 0.2

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to a JSON-compatible dict.

        Returns:
            Dictionary suitable for JSON export.
        """
        return {
            "fsg": self.fsg,
            "theater_score": self.theater_score,
            "sf_aggregate": self.sf_aggregate,
            "sc_aggregate": self.sc_aggregate,
            "four_way_counts": dict(self.four_way_counts),
            "per_record_sf": self.per_record_sf,
            "per_record_sc": self.per_record_sc,
            "tau_high": self.tau_high,
            "tau_low": self.tau_low,
            "sf_components": {
                "parse_rate": self.sf_result.parse_rate if self.sf_result else 0.0,
                "tag_validity_mean": (
                    self.sf_result.tag_validity_mean if self.sf_result else 0.0
                ),
                "section_ratio_conformity": (
                    self.sf_result.section_ratio_conformity if self.sf_result else 0.0
                ),
                "length_conformity": (
                    self.sf_result.length_conformity if self.sf_result else 0.0
                ),
                "template_ngram_overlap_mean": (
                    self.sf_result.template_ngram_overlap_mean
                    if self.sf_result
                    else 0.0
                ),
                "entropy_profile_mean": (
                    self.sf_result.entropy_profile_mean if self.sf_result else None
                ),
                "repetition_mean": (
                    self.sf_result.repetition_mean if self.sf_result else 0.0
                ),
            },
            "sc_components": {
                "answer_accuracy_mean": (
                    self.sc_result.answer_accuracy_mean if self.sc_result else 0.0
                ),
                "step_validity_mean": (
                    self.sc_result.step_validity_mean if self.sc_result else None
                ),
            },
        }


def gap_report(
    records: list[GenerationRecord],
    *,
    config: dict[str, Any] | None = None,
    tau_high: float | None = None,
    tau_low: float | None = None,
) -> GapReport:
    """Build a full gap metrics report for a set of records.

    Args:
        records: Generation records to evaluate.
        config: Optional metrics config dict (loads thresholds and weights).
        tau_high: Override high SF threshold.
        tau_low: Override low SC threshold.

    Returns:
        GapReport with aggregate metrics and per-record arrays.
    """
    cfg = config or {}
    gap_cfg = cfg.get("gap", {})
    tau_h = tau_high if tau_high is not None else gap_cfg.get("tau_high", 0.8)
    tau_l = tau_low if tau_low is not None else gap_cfg.get("tau_low", 0.2)

    tags = tags_from_config(cfg)
    templates = cfg.get("teacher_opening_templates", [])
    sf_weights = cfg.get("structural_fidelity", {}).get("weights", {})
    sc_weights = cfg.get("semantic_correctness", {}).get("weights", {})
    ngram_n = cfg.get("structural_fidelity", {}).get("ngram_n", 3)
    reference = reference_from_config(cfg)

    sf_result = structural_fidelity(
        records,
        reference=reference,
        weights=sf_weights,
        tags=tags,
        templates=templates,
        ngram_n=ngram_n,
    )
    sc_result = semantic_correctness(records, weights=sc_weights)

    per_sf = [
        sample_sf(
            r,
            reference=reference,
            tags=tags,
            templates=templates,
            ngram_n=ngram_n,
        )
        for r in records
    ]
    per_sc = [sample_sc(r) for r in records]

    sf_agg = sf_result.score
    sc_agg = sc_result.score
    fsg_val = fsg(sf_agg, sc_agg)

    theater = theater_score(
        records,
        tau_high=tau_h,
        tau_low=tau_l,
        reference=reference,
        tags=tags,
        templates=templates,
    )

    counts: dict[str, int] = {
        "coherent_correct": 0,
        "coherent_wrong": 0,
        "malformed_correct": 0,
        "malformed_wrong": 0,
    }
    for i, r in enumerate(records):
        label = four_way_classify(
            r,
            tau_high=tau_h,
            tau_low=tau_l,
            sf=per_sf[i],
            sc=per_sc[i],
        )
        counts[label] += 1

    return GapReport(
        fsg=fsg_val,
        theater_score=theater,
        sf_aggregate=sf_agg,
        sc_aggregate=sc_agg,
        four_way_counts=counts,
        per_record_sf=per_sf,
        per_record_sc=per_sc,
        sf_result=sf_result,
        sc_result=sc_result,
        tau_high=tau_h,
        tau_low=tau_l,
    )
