"""Build reasoning SFT datasets with teacher traces and M1 tag formatting."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from hollow_chains.data.tasks import (
    TaskSample,
    load_arithmetic_samples,
    load_symbolic_samples,
)
from hollow_chains.metrics.parse import (
    TAG_BEGIN_SOLUTION,
    TAG_BEGIN_THOUGHT,
    TAG_END_SOLUTION,
    TAG_END_THOUGHT,
)
from hollow_chains.metrics.semantic import extract_answer

FormatType = Literal["full", "equation_only"]


@dataclass
class SFTExample:
    """One SFT training example."""

    prompt: str
    target_completion: str
    task_type: str
    gold: str
    teacher: str
    format: FormatType


def expand_seed_problems(count: int, seed: int = 42) -> list[tuple[TaskSample, str]]:
    """Expand arithmetic + symbolic seed problems to ``count`` items.

    Returns:
        List of (TaskSample, task_type) pairs.
    """
    rng = random.Random(seed)
    arith = load_arithmetic_samples()
    sym = load_symbolic_samples()
    pool: list[tuple[TaskSample, str]] = [(s, "arithmetic") for s in arith] + [
        (s, "symbolic") for s in sym
    ]
    out: list[tuple[TaskSample, str]] = []
    for i in range(count):
        sample, ttype = pool[i % len(pool)]
        out.append(
            (
                TaskSample(
                    id=f"{sample.id}_{i:04d}",
                    prompt=sample.prompt,
                    gold=sample.gold,
                ),
                ttype,
            )
        )
    rng.shuffle(out)
    return out


def _wrap_trace(think: str, solution: str) -> str:
    return (
        f"{TAG_BEGIN_THOUGHT} {think} {TAG_END_THOUGHT} "
        f"{TAG_BEGIN_SOLUTION} {solution} {TAG_END_SOLUTION}"
    )


def format_full_trace(think: str, solution: str) -> str:
    """Format think + solution with M1 tags."""
    return _wrap_trace(think, solution)


def format_equation_only(think: str, solution: str) -> str:
    """Collapse natural language into equation chain only."""
    equations = re.findall(
        r"\d+(?:\.\d+)?\s*[+\-*/]\s*\d+(?:\.\d+)?\s*=\s*\d+(?:\.\d+)?",
        think,
    )
    eq_block = "\n".join(equations) if equations else think.strip()
    return _wrap_trace(eq_block, solution)


def teacher_trace_heuristic(sample: TaskSample, task_type: str) -> str | None:
    """Deterministic teacher trace for smoke tests (no GPU teacher).

    Returns None if trace would fail correctness filter.
    """
    if task_type == "arithmetic":
        try:
            # Simple heuristic for built-in word problems.
            nums = [int(x) for x in re.findall(r"\d+", sample.prompt)]
            if len(nums) >= 2 and "+" in sample.prompt:
                a, b = nums[0], nums[1]
                think = f"Okay the user wants the answer. " f"{a} + {b} = {a + b}"
                solution = f"The answer is {sample.gold}."
                return format_full_trace(think, solution)
            if len(nums) >= 2 and ("sell" in sample.prompt or "away" in sample.prompt):
                a, b = nums[0], nums[1]
                think = f"Start with {a}, subtract {b}: {a} - {b} = {a - b}"
                solution = f"The answer is {sample.gold}."
                return format_full_trace(think, solution)
        except (ValueError, IndexError):
            return None
    if task_type == "symbolic":
        think = f"Simplify step by step to get {sample.gold}."
        solution = f"The answer is {sample.gold}."
        return format_full_trace(think, solution)
    return None


def passes_correctness_filter(completion: str, gold: str, task_type: str) -> bool:
    """Keep only teacher traces whose final answer matches gold."""
    from hollow_chains.metrics.parse import parse_trace

    parsed = parse_trace(completion)
    extracted = extract_answer(parsed, task_type)

    def norm(s: str) -> str:
        return re.sub(r"\s+", "", s.strip().lower())

    return norm(extracted) == norm(gold) and bool(norm(gold))


def build_sft_jsonl(
    config: dict[str, Any],
    *,
    teacher: str = "1p7b",
    samples: int = 500,
    fmt: FormatType = "full",
    smoke: bool = False,
) -> tuple[Path, float]:
    """Build SFT JSONL with correctness filtering.

    Args:
        config: sft.yaml dict.
        teacher: Teacher key (0p6b, 1p7b, 4b) or ``heuristic`` for smoke.
        samples: Number of examples to emit.
        fmt: ``full`` or ``equation_only``.
        smoke: Use heuristic traces (no HF teacher).

    Returns:
        Tuple of (output path, teacher pass rate).
    """
    seed = int(config.get("seed", 42))
    data_cfg = config.get("data", {})
    out_dir = Path(data_cfg.get("output_dir", "artifacts/sft_data"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sft_{teacher}_{samples}_{fmt}.jsonl"

    pre_supplied = data_cfg.get("teacher_traces_jsonl")
    if pre_supplied and Path(pre_supplied).is_file():
        return Path(pre_supplied), 1.0

    problems = expand_seed_problems(int(data_cfg.get("seed_problems", 1000)), seed=seed)
    kept: list[SFTExample] = []
    attempted = 0
    cache_dir = Path(data_cfg.get("teacher_cache_dir", "artifacts/teacher_cache"))
    teachers_cfg = config.get("teachers", {})
    teacher_gen = config.get("teacher_gen", {})

    for sample, task_type in problems:
        if len(kept) >= samples:
            break
        attempted += 1
        raw: str | None = None

        if smoke or teacher == "heuristic":
            raw = teacher_trace_heuristic(sample, task_type)
        else:
            from hollow_chains.data.teacher_hf import (
                generate_teacher_trace,
                load_teacher_cache,
                qwen_to_schema,
                save_teacher_cache,
                teacher_cache_path,
            )

            cache_path = teacher_cache_path(cache_dir, teacher, task_type, sample.id)
            completion = load_teacher_cache(cache_path)
            if completion is None:
                teacher_id = teachers_cfg.get(teacher, {}).get("model_id")
                if not teacher_id:
                    raise KeyError(f"No model_id for teacher '{teacher}' in sft.yaml")
                completion = generate_teacher_trace(
                    sample,
                    task_type,
                    teacher_id,
                    gen_cfg=teacher_gen,
                )
                if completion:
                    save_teacher_cache(cache_path, completion)
            raw = (
                qwen_to_schema(completion, sample.gold, task_type)
                if completion
                else None
            )
        if raw is None:
            continue
        if data_cfg.get("correctness_filter", True) and not passes_correctness_filter(
            raw, sample.gold, task_type
        ):
            continue
        if fmt == "equation_only":
            from hollow_chains.metrics.parse import parse_trace

            parsed = parse_trace(raw)
            raw = format_equation_only(parsed.think, parsed.solution)
        kept.append(
            SFTExample(
                prompt=sample.prompt,
                target_completion=raw,
                task_type=task_type,
                gold=sample.gold,
                teacher=teacher,
                format=fmt,
            )
        )

    with out_path.open("w", encoding="utf-8") as fh:
        for ex in kept[:samples]:
            fh.write(
                json.dumps(
                    {
                        "prompt": ex.prompt,
                        "target_completion": ex.target_completion,
                        "task_type": ex.task_type,
                        "gold": ex.gold,
                        "teacher": ex.teacher,
                        "format": ex.format,
                    }
                )
                + "\n"
            )

    pass_rate = len(kept) / max(attempted, 1)
    stats_path = out_dir / f"sft_{teacher}_{samples}_{fmt}_stats.json"
    stats_path.write_text(
        json.dumps(
            {"teacher_pass_rate": pass_rate, "kept": len(kept), "attempted": attempted},
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_path, pass_rate


def load_sft_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load SFT JSONL records."""
    records = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    return records
