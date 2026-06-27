"""HuggingFace teacher inference for reasoning SFT data (lazy torch import)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from hollow_chains.data.tasks import TaskSample
from hollow_chains.metrics.parse import (
    TAG_BEGIN_SOLUTION,
    TAG_BEGIN_THOUGHT,
    TAG_END_SOLUTION,
    TAG_END_THOUGHT,
    parse_trace,
)
from hollow_chains.metrics.semantic import extract_answer

# Qwen3 reasoning spans (native think tags; fixtures may use redacted_thinking names).
_THINK_PATTERNS = [
    re.compile(r"<\s*think\s*>(.*?)</\s*think\s*>", re.DOTALL | re.IGNORECASE),
    re.compile(
        r"<think>(.*?)</think>",
        re.DOTALL | re.IGNORECASE,
    ),
]

_ANSWER_LINE_RE = re.compile(
    r"(?:the answer is|final answer is|answer:)\s*.+",
    re.IGNORECASE,
)

_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


def _wrap_trace(think: str, solution: str) -> str:
    return (
        f"{TAG_BEGIN_THOUGHT} {think} {TAG_END_THOUGHT} "
        f"{TAG_BEGIN_SOLUTION} {solution} {TAG_END_SOLUTION}"
    )


def _load_teacher(teacher_id: str) -> tuple[Any, Any]:
    """Load and cache a teacher model and tokenizer keyed by HF id."""
    if teacher_id not in _MODEL_CACHE:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            teacher_id,
            torch_dtype="auto",
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(teacher_id)
        _MODEL_CACHE[teacher_id] = (model, tokenizer)
    return _MODEL_CACHE[teacher_id]


def _build_teacher_prompt(problem: str, tokenizer) -> str:
    """Format the user message with the teacher chat template (thinking mode)."""
    user_content = (
        problem + " Please reason step by step, and put your final answer on the "
        "last line as 'The answer is <X>.'"
    )
    messages = [{"role": "user", "content": user_content}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def generate_teacher_trace(
    sample: TaskSample,
    task_type: str,
    teacher_id: str,
    *,
    gen_cfg: dict[str, Any] | None = None,
) -> str | None:
    """Run the HF teacher and return the raw decoded completion.

    Uses sampling (not greedy) for Qwen3 thinking mode. Torch/transformers are
    imported only when this function is called.

    Args:
        sample: Problem prompt and metadata.
        task_type: Task label (reserved for cache callers).
        teacher_id: HuggingFace model id.
        gen_cfg: Optional overrides for temperature, top_p, top_k, max_new_tokens.

    Returns:
        Raw model output string, or None on failure.
    """
    del task_type
    import torch

    cfg = gen_cfg or {}
    temperature = float(cfg.get("temperature", 0.6))
    top_p = float(cfg.get("top_p", 0.95))
    top_k = int(cfg.get("top_k", 20))
    max_new_tokens = int(cfg.get("max_new_tokens", 512))

    model, tokenizer = _load_teacher(teacher_id)
    prompt_text = _build_teacher_prompt(sample.prompt, tokenizer)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
        if k in ("input_ids", "attention_mask")
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    completion = tokenizer.decode(
        outputs[0, inputs["input_ids"].shape[1] :],
        skip_special_tokens=False,
    )
    return completion.strip() or None


def _extract_qwen_thinking(raw: str) -> str:
    """Pull thinking text from Qwen-style tags or text before the answer line."""
    for pattern in _THINK_PATTERNS:
        match = pattern.search(raw)
        if match:
            return match.group(1).strip()

    answer_match = _ANSWER_LINE_RE.search(raw)
    if answer_match:
        return raw[: answer_match.start()].strip()
    return raw.strip()


def _extract_final_answer_line(raw: str) -> str:
    """Extract the final answer line for the solution block."""
    answer_match = _ANSWER_LINE_RE.search(raw)
    if answer_match:
        return answer_match.group(0).strip()
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return lines[-1] if lines else raw.strip()


def qwen_to_schema(raw_completion: str, gold: str, task_type: str) -> str | None:
    """Map Qwen teacher output into the M1 reasoning tag schema.

    Args:
        raw_completion: Raw teacher decode (may include thinking tags).
        gold: Ground-truth answer (unused for text fabrication).
        task_type: Task type for post-map validation.

    Returns:
        Formatted trace string, or None if mapping fails.
    """
    del gold
    if not raw_completion or not raw_completion.strip():
        return None

    think = _extract_qwen_thinking(raw_completion)
    solution = _extract_final_answer_line(raw_completion)
    if not think and not solution:
        return None

    trace = _wrap_trace(think, solution)
    parsed = parse_trace(trace)
    if not parsed.well_formed and not (think or solution):
        return None

    extracted = extract_answer(parsed, task_type)
    if not extracted and not solution:
        return None
    return trace


def teacher_cache_path(
    cache_dir: Path,
    teacher: str,
    task_type: str,
    problem_id: str,
) -> Path:
    """Path for a cached raw teacher completion."""
    safe_id = re.sub(r"[^\w\-.]", "_", problem_id)
    return cache_dir / teacher / task_type / f"{safe_id}.txt"


def load_teacher_cache(path: Path) -> str | None:
    """Read cached teacher output if the file exists."""
    if path.is_file():
        return path.read_text(encoding="utf-8").strip() or None
    return None


def save_teacher_cache(path: Path, completion: str) -> None:
    """Write raw teacher output to cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(completion, encoding="utf-8")
