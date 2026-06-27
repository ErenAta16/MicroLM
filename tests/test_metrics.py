"""Unit tests for hollow-chains metrics layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from hollow_chains.config import load_config
from hollow_chains.data.schema import GenerationRecord, dump_jsonl, load_jsonl
from hollow_chains.metrics.gap import (
    four_way_classify,
    fsg,
    gap_report,
    sample_sc,
    sample_sf,
    theater_score,
)
from hollow_chains.metrics.parse import parse_trace
from hollow_chains.metrics.semantic import (
    TrivialSymbolicChecker,
    answer_accuracy,
    extract_answer,
    semantic_correctness,
    step_validity,
)
from hollow_chains.metrics.structural import (
    entropy_profile,
    length_conformity,
    parse_rate,
    repetition,
    section_ratio,
    section_ratio_conformity,
    structural_fidelity,
    tag_validity,
    template_ngram_overlap,
)


class TestParse:
    """Tests for trace parsing."""

    def test_well_formed_trace(self, coherent_correct_record: GenerationRecord) -> None:
        trace = parse_trace(coherent_correct_record.generation)
        assert trace.well_formed
        assert "5 + 3 = 8" in trace.think
        assert "8 apples" in trace.solution

    def test_malformed_missing_end_solution(
        self, malformed_record: GenerationRecord
    ) -> None:
        trace = parse_trace(malformed_record.generation)
        assert not trace.well_formed
        assert any("missing" in e or "end_solution" in e for e in trace.parse_errors)

    def test_parse_never_raises_on_empty(self) -> None:
        trace = parse_trace("")
        assert not trace.well_formed

    def test_duplicate_tags(self) -> None:
        gen = (
            "<|begin_of_thought|> hi <|end_of_thought|> "
            "<|begin_of_thought|> dup <|end_of_thought|> "
            "<|begin_of_solution|> ans <|end_of_solution|>"
        )
        trace = parse_trace(gen)
        assert not trace.well_formed
        assert not trace.tags_unique


class TestStructural:
    """Tests for Structural Fidelity metrics."""

    def test_parse_rate_with_mixed_records(
        self,
        coherent_correct_record: GenerationRecord,
        malformed_record: GenerationRecord,
    ) -> None:
        records = [coherent_correct_record, malformed_record]
        rate = parse_rate(records)
        assert rate == 0.5

    def test_tag_validity_well_formed(
        self, coherent_correct_record: GenerationRecord
    ) -> None:
        trace = parse_trace(coherent_correct_record.generation)
        assert tag_validity(trace) == 1.0

    def test_tag_validity_malformed(self, malformed_record: GenerationRecord) -> None:
        trace = parse_trace(malformed_record.generation)
        assert tag_validity(trace) < 1.0

    def test_section_ratio(self, coherent_correct_record: GenerationRecord) -> None:
        trace = parse_trace(coherent_correct_record.generation)
        ratio = section_ratio(trace)
        assert 0.0 < ratio < 1.0

    def test_template_overlap(
        self, coherent_correct_record: GenerationRecord, metrics_config: dict
    ) -> None:
        trace = parse_trace(coherent_correct_record.generation)
        templates = metrics_config["teacher_opening_templates"]
        overlap = template_ngram_overlap(trace, templates, n=3)
        assert overlap > 0.0

    def test_repetition_degenerate(self, repetitive_record: GenerationRecord) -> None:
        trace = parse_trace(repetitive_record.generation)
        rep = repetition(trace)
        assert rep.distinct_1 < 0.5
        assert rep.score < 0.7

    def test_structural_fidelity_returns_components(
        self,
        coherent_correct_record: GenerationRecord,
        metrics_config: dict,
    ) -> None:
        ref = metrics_config["reference"]
        result = structural_fidelity(
            [coherent_correct_record],
            reference=ref,
            templates=metrics_config["teacher_opening_templates"],
        )
        assert 0.0 <= result.score <= 1.0
        assert result.parse_rate == 1.0
        assert result.tag_validity_mean == 1.0

    def test_section_ratio_conformity(
        self,
        coherent_correct_record: GenerationRecord,
        metrics_config: dict,
    ) -> None:
        ref = metrics_config["reference"]["section_ratios"]
        score = section_ratio_conformity([coherent_correct_record], ref)
        assert 0.0 <= score <= 1.0

    def test_length_conformity(
        self,
        coherent_correct_record: GenerationRecord,
        metrics_config: dict,
    ) -> None:
        ref = metrics_config["reference"]["lengths"]
        score = length_conformity([coherent_correct_record], ref)
        assert 0.0 <= score <= 1.0

    def test_entropy_profile_with_entropies(
        self, coherent_correct_record: GenerationRecord
    ) -> None:
        record = coherent_correct_record.model_copy(
            update={"token_entropies": [0.8, 0.9, 0.7, 0.6, 0.5, 0.4]}
        )
        result = entropy_profile(record)
        assert result is not None
        assert result.score is not None

    def test_entropy_profile_missing(
        self, coherent_correct_record: GenerationRecord
    ) -> None:
        assert entropy_profile(coherent_correct_record) is None

    def test_empty_records_structural(self) -> None:
        assert parse_rate([]) == 0.0
        result = structural_fidelity(
            [], reference={"section_ratios": [], "lengths": []}
        )
        assert result.score == 0.0


class TestSemantic:
    """Tests for Semantic Correctness metrics."""

    def test_extract_answer_arithmetic(
        self, coherent_correct_record: GenerationRecord
    ) -> None:
        trace = parse_trace(coherent_correct_record.generation)
        answer = extract_answer(trace, "arithmetic")
        assert answer == "8"

    def test_answer_accuracy_correct(
        self, coherent_correct_record: GenerationRecord
    ) -> None:
        result = answer_accuracy(coherent_correct_record)
        assert result.score == 1.0

    def test_answer_accuracy_wrong(
        self, coherent_wrong_record: GenerationRecord
    ) -> None:
        result = answer_accuracy(coherent_wrong_record)
        assert result.score == 0.0

    def test_step_validity_partial(
        self, arithmetic_wrong_step_record: GenerationRecord
    ) -> None:
        result = step_validity(arithmetic_wrong_step_record)
        assert result.total_count == 2
        assert result.valid_count == 1
        assert result.score == 0.5

    def test_step_validity_mcq_skipped(
        self, coherent_wrong_record: GenerationRecord
    ) -> None:
        result = step_validity(coherent_wrong_record)
        assert result.score is None

    def test_extract_answer_mcq(self, coherent_wrong_record: GenerationRecord) -> None:
        trace = parse_trace(coherent_wrong_record.generation)
        assert extract_answer(trace, "factual_mcq") == ""

    def test_extract_answer_symbolic(self) -> None:
        from hollow_chains.metrics.parse import ParsedTrace

        trace = ParsedTrace(raw="", solution="The simplified form is 5x")
        assert extract_answer(trace, "symbolic") == "5x"

    def test_semantic_correctness_aggregate(
        self,
        coherent_correct_record: GenerationRecord,
        coherent_wrong_record: GenerationRecord,
    ) -> None:
        result = semantic_correctness([coherent_correct_record, coherent_wrong_record])
        assert 0.0 <= result.score <= 1.0
        assert result.answer_accuracy_mean == 0.5

    def test_symbolic_step_validity(self) -> None:
        think = "2x + 3x = 5x"
        gen = (
            "<|begin_of_thought|> " + think + " <|end_of_thought|> "
            "<|begin_of_solution|> 5x <|end_of_solution|>"
        )
        record = GenerationRecord(
            id="sym",
            prompt="simplify",
            task_type="symbolic",
            gold="5x",
            generation=gen,
        )
        result = step_validity(record, TrivialSymbolicChecker())
        assert result.total_count > 0


class TestGap:
    """Tests for gap metrics."""

    def test_fsg(self) -> None:
        assert fsg(0.9, 0.3) == pytest.approx(0.6)

    def test_coherent_correct_classification(
        self,
        coherent_correct_record: GenerationRecord,
        metrics_config: dict,
    ) -> None:
        ref = metrics_config["reference"]
        sf = sample_sf(
            coherent_correct_record,
            reference=ref,
            templates=metrics_config["teacher_opening_templates"],
        )
        sc = sample_sc(coherent_correct_record)
        label = four_way_classify(
            coherent_correct_record,
            sf=sf,
            sc=sc,
            tau_high=0.8,
            tau_low=0.2,
        )
        assert label == "coherent_correct"
        assert sf >= 0.8
        assert sc > 0.2

    def test_coherent_wrong_theater(
        self,
        coherent_wrong_record: GenerationRecord,
        metrics_config: dict,
    ) -> None:
        ref = metrics_config["reference"]
        sf = sample_sf(
            coherent_wrong_record,
            reference=ref,
            templates=metrics_config["teacher_opening_templates"],
        )
        sc = sample_sc(coherent_wrong_record)
        label = four_way_classify(
            coherent_wrong_record,
            sf=sf,
            sc=sc,
            tau_high=0.8,
            tau_low=0.2,
        )
        assert sf >= 0.8
        assert sc <= 0.2
        assert label == "coherent_wrong"

    def test_theater_score_counts_theater(
        self,
        coherent_correct_record: GenerationRecord,
        coherent_wrong_record: GenerationRecord,
        metrics_config: dict,
    ) -> None:
        records = [coherent_correct_record, coherent_wrong_record]
        score = theater_score(
            records,
            tau_high=0.8,
            tau_low=0.2,
            reference=metrics_config["reference"],
            templates=metrics_config["teacher_opening_templates"],
        )
        assert score == 0.5

    def test_malformed_classification(
        self,
        malformed_record: GenerationRecord,
        metrics_config: dict,
    ) -> None:
        ref = metrics_config["reference"]
        sf = sample_sf(
            malformed_record,
            reference=ref,
            templates=metrics_config["teacher_opening_templates"],
        )
        sc = sample_sc(malformed_record)
        label = four_way_classify(
            malformed_record,
            sf=sf,
            sc=sc,
            tau_high=0.8,
            tau_low=0.2,
        )
        assert sf < 0.8
        assert label in ("malformed_correct", "malformed_wrong")

    def test_gap_report_structure(
        self,
        coherent_correct_record: GenerationRecord,
        coherent_wrong_record: GenerationRecord,
        metrics_config: dict,
    ) -> None:
        report = gap_report(
            [coherent_correct_record, coherent_wrong_record],
            config=metrics_config,
        )
        d = report.to_dict()
        assert "fsg" in d
        assert "theater_score" in d
        assert len(d["per_record_sf"]) == 2
        assert len(d["per_record_sc"]) == 2
        assert sum(d["four_way_counts"].values()) == 2


class TestSchema:
    """Tests for data schema and JSONL I/O."""

    def test_jsonl_roundtrip(self, tmp_path: Path) -> None:
        record = GenerationRecord(
            id="test",
            prompt="p",
            task_type="arithmetic",
            gold="1",
            generation="g",
        )
        path = tmp_path / "data.jsonl"
        dump_jsonl([record], path)
        loaded = load_jsonl(path)
        assert len(loaded) == 1
        assert loaded[0].id == "test"

    def test_load_config(self) -> None:
        config_path = Path(__file__).parent.parent / "configs" / "metrics.yaml"
        cfg = load_config(config_path)
        assert "gap" in cfg
        assert cfg["gap"]["tau_high"] == 0.8
