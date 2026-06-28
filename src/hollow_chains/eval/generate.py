"""Generation harness — produces M1 GenerationRecord JSONL."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from hollow_chains.data.schema import GenerationRecord, dump_jsonl
from hollow_chains.data.tasks import (
    load_arithmetic_samples,
    load_factual_mcq,
    load_symbolic_samples,
)
from hollow_chains.eval.recipes import (
    get_recipe,
    postprocess_generation,
)
from hollow_chains.metrics.parse import parse_trace

TaskType = Literal["arithmetic", "symbolic", "factual_mcq"]

logger = logging.getLogger(__name__)
_TRANSFORMERS_PATCHED = False


@dataclass
class EvalTask:
    """Task record for generation."""

    id: str
    prompt: str
    gold: str
    task_type: TaskType


def load_eval_tasks(task_sets: list[str]) -> list[EvalTask]:
    """Load evaluation tasks from M1 task loaders."""
    tasks: list[EvalTask] = []
    if "arithmetic" in task_sets:
        for s in load_arithmetic_samples():
            tasks.append(EvalTask(s.id, s.prompt, s.gold, "arithmetic"))
    if "symbolic" in task_sets:
        for s in load_symbolic_samples():
            tasks.append(EvalTask(s.id, s.prompt, s.gold, "symbolic"))
    if "factual_mcq" in task_sets:
        for s in load_factual_mcq():
            tasks.append(EvalTask(s.id, s.prompt, s.gold, "factual_mcq"))
    return tasks


def _apply_transformers_compat() -> None:
    """Patch TokenizersBackend for SupraLabs fast tokenizers on transformers 4.x."""
    global _TRANSFORMERS_PATCHED
    if _TRANSFORMERS_PATCHED:
        return
    import transformers

    if not hasattr(transformers, "TokenizersBackend"):
        transformers.TokenizersBackend = transformers.PreTrainedTokenizerFast
    _TRANSFORMERS_PATCHED = True


def _entropy_from_scores(scores) -> list[float]:
    """Compute per-step Shannon entropy from generation scores."""
    import torch
    import torch.nn.functional as F

    entropies: list[float] = []
    for score in scores:
        probs = F.softmax(score[0], dim=-1)
        log_probs = torch.log(probs + 1e-12)
        ent = -(probs * log_probs).sum().item()
        entropies.append(ent)
    return entropies


def _run_generate(
    model,
    tokenizer,
    prompt_text: str,
    decoding: dict[str, Any],
    *,
    capture_entropy: bool = True,
) -> tuple[str, list[float] | None]:
    """Tokenize, generate, and decode new tokens."""
    import torch

    enc = tokenizer(prompt_text, return_tensors="pt")
    enc.pop("token_type_ids", None)
    device = next(model.parameters()).device
    inputs = {
        k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")
    }

    gen_kwargs: dict[str, Any] = {
        **decoding,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "return_dict_in_generate": True,
        "output_scores": capture_entropy,
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    new_text = tokenizer.decode(
        outputs.sequences[0, inputs["input_ids"].shape[1] :],
        skip_special_tokens=False,
    )

    entropies: list[float] | None = None
    if capture_entropy and hasattr(outputs, "scores") and outputs.scores:
        entropies = _entropy_from_scores(outputs.scores)

    return new_text, entropies


def generate_with_recipe(
    model,
    tokenizer,
    task_records: list[EvalTask],
    recipe_name: str,
    *,
    model_id: str = "local",
    capture_entropies: bool = True,
) -> list[GenerationRecord]:
    """Generate using a validated recipe registry entry.

    Args:
        model: Causal LM.
        tokenizer: Matching tokenizer.
        task_records: Evaluation tasks.
        recipe_name: ``reasoning`` or ``instruct``.
        model_id: Stored in record meta.
        capture_entropies: Whether to record per-token entropies.

    Returns:
        List of GenerationRecord objects.
    """
    recipe = get_recipe(recipe_name)
    records: list[GenerationRecord] = []
    model.eval()

    for task in task_records:
        prompt_text = recipe.format_prompt(task.prompt)
        new_text, entropies = _run_generate(
            model,
            tokenizer,
            prompt_text,
            recipe.decoding,
            capture_entropy=capture_entropies,
        )
        generation = postprocess_generation(new_text, recipe)

        parsed = parse_trace(generation)
        records.append(
            GenerationRecord(
                id=task.id,
                prompt=task.prompt,
                task_type=task.task_type,
                gold=task.gold,
                generation=generation,
                think=parsed.think or None,
                solution=parsed.solution or None,
                token_entropies=entropies,
                meta={
                    "model_id": model_id,
                    "recipe": recipe_name,
                    **recipe.decoding,
                },
            )
        )

    return records


def generate_records(
    model,
    tokenizer,
    task_records: list[EvalTask],
    decoding_cfg: dict[str, Any],
    *,
    model_id: str = "local",
) -> list[GenerationRecord]:
    """Generate completions with a generic decoding config (local checkpoints)."""
    import torch

    records: list[GenerationRecord] = []
    do_sample = bool(decoding_cfg.get("do_sample", False))
    temperature = float(decoding_cfg.get("temperature", 1.0))
    max_new = int(decoding_cfg.get("max_new_tokens", 256))
    capture_entropy = bool(decoding_cfg.get("capture_token_entropies", True))

    model.eval()
    device = next(model.parameters()).device

    for task in task_records:
        enc = tokenizer(task.prompt, return_tensors="pt")
        enc.pop("token_type_ids", None)
        inputs = {
            k: v.to(device)
            for k, v in enc.items()
            if k in ("input_ids", "attention_mask")
        }
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": capture_entropy,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        full_generation = tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=False
        )

        entropies: list[float] | None = None
        if capture_entropy and hasattr(outputs, "scores") and outputs.scores:
            entropies = _entropy_from_scores(outputs.scores)

        parsed = parse_trace(full_generation)
        records.append(
            GenerationRecord(
                id=task.id,
                prompt=task.prompt,
                task_type=task.task_type,
                gold=task.gold,
                generation=full_generation,
                think=parsed.think or None,
                solution=parsed.solution or None,
                token_entropies=entropies,
                meta={
                    "model_id": model_id,
                    "do_sample": do_sample,
                    "temperature": temperature,
                    "max_new_tokens": max_new,
                },
            )
        )

    return records


def validate_records(records: list[GenerationRecord]) -> None:
    """Assert every record validates against the M1 schema."""
    for rec in records:
        GenerationRecord.model_validate(rec.model_dump())


def write_generation_jsonl(
    records: list[GenerationRecord],
    path: str | Path,
) -> Path:
    """Validate and write records via M1 dump_jsonl."""
    validate_records(records)
    out = Path(path)
    dump_jsonl(records, out)
    return out


def load_model_and_tokenizer(
    checkpoint: str | Path,
    *,
    external_hf_id: str | None = None,
    use_own_tokenizer: bool = False,
    own_tokenizer_path: str | Path | None = None,
):
    """Load model + tokenizer from checkpoint or external HF id."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if external_hf_id:
        _apply_transformers_compat()
        model = AutoModelForCausalLM.from_pretrained(external_hf_id)
        if use_own_tokenizer and own_tokenizer_path:
            tokenizer = AutoTokenizer.from_pretrained(str(own_tokenizer_path))
        elif use_own_tokenizer and not own_tokenizer_path:
            logger.warning(
                "use_own_tokenizer=True without own_tokenizer_path for %s; "
                "falling back to the model's tokenizer",
                external_hf_id,
            )
            tokenizer = AutoTokenizer.from_pretrained(external_hf_id)
        else:
            tokenizer = AutoTokenizer.from_pretrained(external_hf_id)
    else:
        ckpt = Path(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(str(ckpt))
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer
