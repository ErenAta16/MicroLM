"""Build reasoning SFT datasets with teacher traces and M1 tag formatting."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from hollow_chains.data.sft_format import (
    format_sft_prompt,
    format_sft_target,
)
from hollow_chains.data.tasks import (
    TaskSample,
    load_arithmetic_samples,
    load_symbolic_samples,
)
from hollow_chains.data.teacher_hf import is_correct, map_qwen_output

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


def format_equation_only(thought: str, solution: str) -> tuple[str, str]:
    """Collapse natural language thought into equation chain only."""
    equations = re.findall(
        r"\d+(?:\.\d+)?\s*[+\-*/]\s*\d+(?:\.\d+)?\s*=\s*\d+(?:\.\d+)?",
        thought,
    )
    eq_block = "\n".join(equations) if equations else thought.strip()
    return eq_block, solution


def _build_sft_pair(
    question: str,
    thought: str,
    answer: str,
    *,
    eos: str = "",
) -> tuple[str, str]:
    """Build masked-prompt + completion target matching eval format."""
    prompt = format_sft_prompt(question)
    target = format_sft_target(thought, answer, eos=eos)
    return prompt, target


def teacher_trace_heuristic(
    sample: TaskSample, task_type: str
) -> tuple[str, str] | None:
    """Deterministic teacher trace for smoke tests (no GPU teacher).

    Returns (prompt, target_completion) or None if trace would fail filter.
    """
    thought: str | None = None
    answer = f"The answer is {sample.gold}."

    if task_type == "arithmetic":
        try:
            nums = [int(x) for x in re.findall(r"\d+", sample.prompt)]
            if len(nums) >= 2 and "+" in sample.prompt:
                a, b = nums[0], nums[1]
                thought = f"Okay the user wants the answer. {a} + {b} = {a + b}"
            elif len(nums) >= 2 and (
                "sell" in sample.prompt or "away" in sample.prompt
            ):
                a, b = nums[0], nums[1]
                thought = f"Start with {a}, subtract {b}: {a} - {b} = {a - b}"
        except (ValueError, IndexError):
            return None
    elif task_type == "symbolic":
        thought = f"Simplify step by step to get {sample.gold}."
    else:
        return None

    if thought is None:
        return None

    if not is_correct(answer, sample.gold):
        return None
    return _build_sft_pair(sample.prompt, thought, answer)


def passes_correctness_filter(
    thought: str,
    answer: str,
    gold: str,
) -> bool:
    """Keep only teacher traces whose final answer matches gold."""
    return is_correct(answer, gold) and bool(gold.strip())


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
        prompt: str | None = None
        target: str | None = None

        if smoke or teacher == "heuristic":
            pair = teacher_trace_heuristic(sample, task_type)
            if pair:
                prompt, target = pair
        else:
            from hollow_chains.data.teacher_hf import (
                generate_teacher_trace,
                load_teacher_cache,
                save_teacher_cache,
                teacher_cache_path,
            )

            teacher_id = teachers_cfg.get(teacher, {}).get("model_id")
            if not teacher_id:
                raise KeyError(f"No model_id for teacher '{teacher}' in sft.yaml")

            cache_path = teacher_cache_path(cache_dir, teacher_id, sample.prompt)
            completion = load_teacher_cache(cache_path)
            if completion is None:
                completion = generate_teacher_trace(
                    sample,
                    task_type,
                    teacher_id,
                    gen_cfg=teacher_gen,
                )
                if completion:
                    save_teacher_cache(cache_path, completion)

            if completion:
                thought, answer = map_qwen_output(completion)
                if data_cfg.get(
                    "correctness_filter", True
                ) and not passes_correctness_filter(thought, answer, sample.gold):
                    continue
                if fmt == "equation_only":
                    thought, answer = format_equation_only(thought, answer)
                prompt, target = _build_sft_pair(sample.prompt, thought, answer)

        if prompt is None or target is None:
            continue

        kept.append(
            SFTExample(
                prompt=prompt,
                target_completion=target,
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
