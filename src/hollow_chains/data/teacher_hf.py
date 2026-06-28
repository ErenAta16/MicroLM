"""HuggingFace teacher inference for reasoning SFT data (lazy torch import)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from hollow_chains.data.tasks import TaskSample

# Qwen3 teacher output mapping (validated).
_QWEN_THINK_RE = re.compile(
    r"<think>(.*?)</think>(.*)",
    re.S,
)

_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


def norm_num(s: str) -> float | None:
    """Extract the last numeric literal from text (period-tolerant)."""
    m = re.findall(r"-?\d+\.?\d*", str(s))
    return float(m[-1].rstrip(".")) if m else None


def is_correct(text: str, gold: str) -> bool:
    """Numeric-robust correctness check for teacher trace filtering."""
    p, g = norm_num(text), norm_num(gold)
    return p is not None and g is not None and abs(p - g) < 1e-6


def map_qwen_output(raw: str) -> tuple[str, str]:
    """Map Qwen3 teacher decode to (thought, answer) text.

    If no think tags are present, the whole string is used for both fields.
    """
    cleaned = raw.replace("<|im_end|>", "").strip()
    if not cleaned:
        return "", ""

    match = _QWEN_THINK_RE.search(cleaned)
    if match:
        thought = match.group(1).strip()
        answer = match.group(2).strip()
    else:
        thought = cleaned
        answer = cleaned
    return thought, answer


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
    messages = [{"role": "user", "content": problem}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
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
    enc = tokenizer(prompt_text, return_tensors="pt")
    enc.pop("token_type_ids", None)
    device = next(model.parameters()).device
    inputs = {
        k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")
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


def qwen_to_schema(raw_completion: str, gold: str, task_type: str) -> str | None:
    """Map Qwen teacher output into a well-formed M1 schema trace string.

    Returns None when mapping yields no usable content. Correctness filtering
    is handled separately via ``is_correct``.
    """
    del gold, task_type
    from hollow_chains.data.sft_format import schema_trace_from_parts
    from hollow_chains.metrics.parse import parse_trace

    if not raw_completion or not raw_completion.strip():
        return None

    thought, answer = map_qwen_output(raw_completion)
    if not thought and not answer:
        return None

    trace = schema_trace_from_parts(thought, answer)
    parsed = parse_trace(trace)
    if not parsed.well_formed:
        return None
    return trace


def teacher_cache_path(
    cache_dir: Path,
    teacher_id: str,
    question: str,
) -> Path:
    """Path for cached raw teacher completion keyed by (teacher_id, question)."""
    safe_teacher = re.sub(r"[^\w\-.]", "_", teacher_id.replace("/", "__"))
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
    return cache_dir / safe_teacher / f"{digest}.txt"


def load_teacher_cache(path: Path) -> str | None:
    """Read cached teacher output if the file exists."""
    if path.is_file():
        return path.read_text(encoding="utf-8").strip() or None
    return None


def save_teacher_cache(path: Path, completion: str) -> None:
    """Write raw teacher output to cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(completion, encoding="utf-8")
