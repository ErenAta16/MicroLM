"""Structural Fidelity metrics for reasoning traces."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import wasserstein_distance

from hollow_chains.data.schema import GenerationRecord
from hollow_chains.metrics.parse import ParsedTrace, parse_trace

# Default weights for composite structural fidelity.
DEFAULT_SF_WEIGHTS: dict[str, float] = {
    "parse_rate": 0.15,
    "tag_validity": 0.20,
    "section_ratio_conformity": 0.15,
    "length_conformity": 0.15,
    "template_ngram_overlap": 0.15,
    "entropy_profile": 0.10,
    "repetition": 0.10,
}


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenization."""
    return text.split()


def parse_rate(
    records: list[GenerationRecord], tags: dict[str, str] | None = None
) -> float:
    """Fraction of records with well_formed == True.

    Args:
        records: Generation records to evaluate.
        tags: Optional custom tag mapping.

    Returns:
        Parse rate in [0, 1].
    """
    if not records:
        return 0.0
    well_count = sum(1 for r in records if parse_trace(r.generation, tags).well_formed)
    return well_count / len(records)


def tag_validity(trace: ParsedTrace) -> float:
    """Component score in [0,1] from presence, order, uniqueness, and closure.

    Each sub-criterion contributes equally (0.25 each).

    Args:
        trace: Parsed trace to score.

    Returns:
        Tag validity score in [0, 1].
    """
    presence = sum(trace.tag_present.values()) / 4.0
    uniqueness = 1.0 if trace.tags_unique else 0.0
    from hollow_chains.metrics.parse import EXPECTED_TAG_ORDER

    order_ok = trace.tag_order == list(EXPECTED_TAG_ORDER)
    order = 1.0 if order_ok else 0.0
    closure = 1.0 if trace.tags_properly_closed else 0.0
    return 0.25 * presence + 0.25 * uniqueness + 0.25 * order + 0.25 * closure


def section_ratio(trace: ParsedTrace) -> float:
    """Ratio of think tokens to total think+solution tokens.

    Args:
        trace: Parsed trace with think and solution blocks.

    Returns:
        Section ratio in [0, 1]. Returns 0.5 if both blocks are empty.
    """
    think_tokens = len(_tokenize(trace.think))
    sol_tokens = len(_tokenize(trace.solution))
    total = think_tokens + sol_tokens
    if total == 0:
        return 0.5
    return think_tokens / total


def _normalized_wasserstein(sample_a: list[float], sample_b: list[float]) -> float:
    """Normalized Wasserstein distance in [0, 1]."""
    if not sample_a or not sample_b:
        return 1.0
    dist = wasserstein_distance(sample_a, sample_b)
    max_val = max(max(sample_a), max(sample_b), 1.0)
    return min(dist / max_val, 1.0)


def section_ratio_conformity(
    records: list[GenerationRecord],
    reference_ratios: list[float],
    tags: dict[str, str] | None = None,
) -> float:
    """Conformity to a reference section-ratio distribution.

    Score = 1 - normalized Wasserstein distance.

    Args:
        records: Records to evaluate.
        reference_ratios: Reference distribution of section ratios.
        tags: Optional custom tag mapping.

    Returns:
        Conformity score in [0, 1].
    """
    if not records or not reference_ratios:
        return 0.0
    ratios = [section_ratio(parse_trace(r.generation, tags)) for r in records]
    return 1.0 - _normalized_wasserstein(ratios, reference_ratios)


def length_conformity(
    records: list[GenerationRecord],
    reference_lengths: list[float],
    tags: dict[str, str] | None = None,
) -> float:
    """Conformity to a reference total-length distribution.

    Total length = token count of think + solution blocks.
    Score = 1 - normalized Wasserstein distance.

    Args:
        records: Records to evaluate.
        reference_lengths: Reference length distribution.
        tags: Optional custom tag mapping.

    Returns:
        Conformity score in [0, 1].
    """
    if not records or not reference_lengths:
        return 0.0
    lengths = []
    for r in records:
        trace = parse_trace(r.generation, tags)
        lengths.append(
            float(len(_tokenize(trace.think)) + len(_tokenize(trace.solution)))
        )
    return 1.0 - _normalized_wasserstein(lengths, reference_lengths)


def _normalize_text(text: str) -> str:
    """Lowercase and strip punctuation for template matching."""
    return re.sub(r"[^\w\s]", "", text.lower())


def _leading_ngrams(text: str, n: int) -> set[str]:
    """Extract leading n-grams from normalized text."""
    tokens = _normalize_text(text).split()
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def template_ngram_overlap(
    trace: ParsedTrace,
    templates: list[str],
    n: int = 3,
) -> float:
    """Overlap of trace leading n-grams with teacher opening templates.

    Case and punctuation insensitive. Scores overlap against the union of
    template n-grams found in the think block opening.

    Args:
        trace: Parsed trace.
        templates: List of teacher opening template strings.
        n: N-gram size.

    Returns:
        Overlap score in [0, 1].
    """
    think_ngrams = _leading_ngrams(trace.think, n)
    if not think_ngrams:
        return 0.0

    template_ngrams: set[str] = set()
    for tmpl in templates:
        template_ngrams.update(_leading_ngrams(tmpl, n))

    if not template_ngrams:
        return 0.0

    overlap = len(think_ngrams & template_ngrams)
    return overlap / len(think_ngrams)


@dataclass
class EntropyProfileResult:
    """Mean entropy in think vs solution spans."""

    think_mean_entropy: float | None = None
    solution_mean_entropy: float | None = None
    score: float | None = None


def entropy_profile(record: GenerationRecord) -> EntropyProfileResult | None:
    """Compute mean token entropy in think vs solution spans.

    Requires ``token_entropies`` on the record and a parseable generation.
    Returns None gracefully when entropies are absent.

    Args:
        record: Generation record with optional per-token entropies.

    Returns:
        EntropyProfileResult or None if entropies unavailable.
    """
    if record.token_entropies is None:
        return None

    trace = parse_trace(record.generation)
    think_tokens = _tokenize(trace.think)
    sol_tokens = _tokenize(trace.solution)
    entropies = record.token_entropies

    if len(entropies) == 0:
        return None

    think_len = len(think_tokens)
    sol_len = len(sol_tokens)
    total_content = think_len + sol_len

    if total_content == 0:
        return EntropyProfileResult(score=0.0)

    # Map entropies proportionally to think/solution token counts
    n_ent = len(entropies)
    think_ent = entropies[: max(1, int(n_ent * think_len / total_content))]
    sol_start = len(think_ent)
    sol_ent = entropies[sol_start:]

    think_mean = float(np.mean(think_ent)) if think_ent else None
    sol_mean = float(np.mean(sol_ent)) if sol_ent else None

    # Score: higher entropy in think vs solution is expected (reasoning is harder)
    if think_mean is not None and sol_mean is not None:
        diff = think_mean - sol_mean
        score = min(max((diff + 1.0) / 2.0, 0.0), 1.0)
    elif think_mean is not None:
        score = min(think_mean, 1.0)
    else:
        score = 0.0

    return EntropyProfileResult(
        think_mean_entropy=think_mean,
        solution_mean_entropy=sol_mean,
        score=score,
    )


@dataclass
class RepetitionResult:
    """Distinct-n and repetition degeneration metrics."""

    distinct_1: float = 0.0
    distinct_2: float = 0.0
    rep_n_ratio: float = 0.0
    score: float = 0.0


def repetition(trace: ParsedTrace) -> RepetitionResult:
    """Compute distinct-1/distinct-2 and rep-n degeneration ratios.

    Higher distinct ratios and lower rep-n indicate less degeneration.
    The composite ``score`` combines these into [0, 1].

    Args:
        trace: Parsed trace.

    Returns:
        RepetitionResult with sub-metrics and composite score.
    """
    text = trace.think + " " + trace.solution
    tokens = _tokenize(text)
    if not tokens:
        return RepetitionResult(score=0.0)

    n_tokens = len(tokens)
    distinct_1 = len(set(tokens)) / n_tokens

    ngrams_2 = [tuple(tokens[i : i + 2]) for i in range(n_tokens - 1)]
    distinct_2 = len(set(ngrams_2)) / max(len(ngrams_2), 1)

    # rep-n: fraction of tokens that are exact repeats of a prior token
    seen: set[str] = set()
    repeats = 0
    for t in tokens:
        if t in seen:
            repeats += 1
        seen.add(t)
    rep_n_ratio = repeats / n_tokens

    score = 0.4 * distinct_1 + 0.4 * distinct_2 + 0.2 * (1.0 - rep_n_ratio)
    return RepetitionResult(
        distinct_1=distinct_1,
        distinct_2=distinct_2,
        rep_n_ratio=rep_n_ratio,
        score=min(max(score, 0.0), 1.0),
    )


@dataclass
class SFResult:
    """Composite Structural Fidelity result with all sub-components."""

    score: float = 0.0
    parse_rate: float = 0.0
    tag_validity_mean: float = 0.0
    section_ratio_conformity: float = 0.0
    length_conformity: float = 0.0
    template_ngram_overlap_mean: float = 0.0
    entropy_profile_mean: float | None = None
    repetition_mean: float = 0.0
    per_record_tag_validity: list[float] = field(default_factory=list)
    weights_used: dict[str, float] = field(default_factory=dict)


def sample_structural_fidelity(
    record: GenerationRecord,
    *,
    reference: dict[str, Any],
    tags: dict[str, str] | None = None,
    templates: list[str] | None = None,
    ngram_n: int = 3,
) -> float:
    """Per-record structural fidelity scalar (tag validity + template + repetition).

    Distribution-level metrics (parse_rate, conformities) are corpus-level only;
    this helper combines per-trace metrics for gap analysis.

    Args:
        record: Single generation record.
        reference: Reference distributions dict with optional keys.
        tags: Optional custom tag mapping.
        templates: Teacher opening templates.
        ngram_n: N-gram size for template overlap.

    Returns:
        Sample-level SF scalar in [0, 1].
    """
    trace = parse_trace(record.generation, tags)
    tmpl = templates or []
    tv = tag_validity(trace)
    tn = template_ngram_overlap(trace, tmpl, ngram_n)
    rep = repetition(trace).score
    ep = entropy_profile(record)
    ep_score = ep.score if ep is not None else 0.5

    # Per-trace SF: well-formed tag structure is the primary signal.
    return 0.70 * tv + 0.15 * tn + 0.10 * rep + 0.05 * ep_score


def structural_fidelity(
    records: list[GenerationRecord],
    *,
    reference: dict[str, Any],
    weights: dict[str, float] | None = None,
    tags: dict[str, str] | None = None,
    templates: list[str] | None = None,
    ngram_n: int = 3,
) -> SFResult:
    """Compute composite Structural Fidelity and all sub-components.

    Composite is a weighted mean of normalized sub-metrics in [0, 1].
    Default weights are documented in ``DEFAULT_SF_WEIGHTS``.

    Args:
        records: Generation records to evaluate.
        reference: Dict with optional ``section_ratios`` and ``lengths`` lists.
        weights: Optional weight overrides for sub-metrics.
        tags: Optional custom tag mapping.
        templates: Teacher opening template strings.
        ngram_n: N-gram size for template overlap.

    Returns:
        SFResult with composite score and all sub-components.
    """
    w = dict(DEFAULT_SF_WEIGHTS)
    if weights:
        w.update(weights)

    ref_ratios = reference.get("section_ratios", [])
    ref_lengths = reference.get("lengths", [])
    tmpl = templates or []

    pr = parse_rate(records, tags)
    tv_scores = [tag_validity(parse_trace(r.generation, tags)) for r in records]
    tv_mean = sum(tv_scores) / len(tv_scores) if tv_scores else 0.0

    src = section_ratio_conformity(records, ref_ratios, tags)
    lc = length_conformity(records, ref_lengths, tags)

    tn_scores = [
        template_ngram_overlap(parse_trace(r.generation, tags), tmpl, ngram_n)
        for r in records
    ]
    tn_mean = sum(tn_scores) / len(tn_scores) if tn_scores else 0.0

    ep_scores: list[float] = []
    for r in records:
        ep = entropy_profile(r)
        if ep is not None and ep.score is not None:
            ep_scores.append(ep.score)
    ep_mean: float | None = sum(ep_scores) / len(ep_scores) if ep_scores else None

    rep_scores = [repetition(parse_trace(r.generation, tags)).score for r in records]
    rep_mean = sum(rep_scores) / len(rep_scores) if rep_scores else 0.0

    components: dict[str, float] = {
        "parse_rate": pr,
        "tag_validity": tv_mean,
        "section_ratio_conformity": src,
        "length_conformity": lc,
        "template_ngram_overlap": tn_mean,
        "repetition": rep_mean,
    }
    if ep_mean is not None:
        components["entropy_profile"] = ep_mean

    active_weights = {k: w[k] for k in components if k in w}
    weight_sum = sum(active_weights.values())
    if weight_sum == 0:
        score = 0.0
    else:
        score = (
            sum(components[k] * active_weights[k] for k in active_weights) / weight_sum
        )

    return SFResult(
        score=score,
        parse_rate=pr,
        tag_validity_mean=tv_mean,
        section_ratio_conformity=src,
        length_conformity=lc,
        template_ngram_overlap_mean=tn_mean,
        entropy_profile_mean=ep_mean,
        repetition_mean=rep_mean,
        per_record_tag_validity=tv_scores,
        weights_used=active_weights,
    )


def reference_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract reference distributions from a metrics config.

    Args:
        config: Full metrics config dict.

    Returns:
        Reference dict for structural_fidelity.
    """
    ref = config.get("reference", {})
    return {
        "section_ratios": ref.get("section_ratios", [0.6, 0.65, 0.7, 0.55, 0.6]),
        "lengths": ref.get("lengths", [20.0, 30.0, 40.0, 50.0, 35.0]),
    }
