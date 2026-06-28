#!/usr/bin/env python3
"""Generate R2 teacher SFT JSONL caches (local Qwen + Together API).

Writes ``r2_{teacher}.jsonl`` files compatible with the R2 Colab notebook.
Upload outputs to Drive ``MyDrive/MicroLM/sft_data/`` for cache hits in Colab.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# --- EXACT constants (must match R2 Colab) ---
SYS_REASON = (
    "Your role as an assistant involves thoroughly exploring questions through "
    "a systematic long thinking process before providing the final precise and "
    "accurate solutions."
)
THINK_OPEN = "<|begin_of_thought|>"

N_PROBLEMS_DEFAULT = 200
SEED_DEFAULT = 0

TEACHER_REGISTRY: dict[str, dict[str, str]] = {
    "qwen3-0p6b": {"kind": "local_qwen", "model": "Qwen/Qwen3-0.6B"},
    "qwen3-1p7b": {"kind": "local_qwen", "model": "Qwen/Qwen3-1.7B"},
    # Together serverless IDs (verified via GET /v1/models, 2026-06-28).
    # deepseek-v4: strongest DeepSeek V4 reasoning model on Together.
    # qwen3p5-397b: largest Qwen3.5 MoE (397B) chat endpoint on Together.
    "deepseek-v4": {
        "kind": "together",
        "model": "deepseek-ai/DeepSeek-V4-Pro",
    },
    "qwen3p5-397b": {
        "kind": "together",
        "model": "Qwen/Qwen3.5-397B-A17B",
    },
}

_TOGETHER_PARSE_RE = re.compile(r"(.*?)(the answer is.*)", re.S | re.I)


def make_word_problems(n: int, seed: int) -> list[tuple[str, str]]:
    """Generate 3-step shop inventory word problems."""
    rng = random.Random(seed)
    problems: list[tuple[str, str]] = []
    for _ in range(n):
        start = rng.randint(30, 90)
        sold = rng.randint(5, start // 2)
        restock = rng.randint(5, 30)
        extra_sold = rng.randint(2, 15)
        final = start - sold + restock - extra_sold
        q = (
            f"A shop had {start} items. It sold {sold}, then received "
            f"{restock} more, then sold {extra_sold} more. "
            "How many items does the shop have now?"
        )
        gold = str(final)
        problems.append((q, gold))
    return problems


def norm_num(s: str) -> float | None:
    m = re.findall(r"-?\d+\.?\d*", str(s))
    return float(m[-1].rstrip(".")) if m else None


def is_correct(text: str, gold: str) -> bool:
    p, g = norm_num(text), norm_num(gold)
    return p is not None and g is not None and abs(p - g) < 1e-6


def to_schema(thought: str, answer: str) -> str:
    return (
        f"{THINK_OPEN}\n{thought}\n"
        "<|end_of_thought|><|begin_of_solution|>\n"
        f"{answer}\n<|end_of_solution|>"
    )


def make_sft_row(question: str, thought: str, answer: str) -> dict[str, str]:
    prompt = (
        f"[SYSTEM]: {SYS_REASON}\n\n"
        f"[USER]: {question}\n\n"
        f"[ASSISTANT]: {THINK_OPEN}\n"
    )
    completion = to_schema(thought, answer)[len(THINK_OPEN) + 1 :]
    return {
        "question": question,
        "prompt": prompt,
        "completion": completion,
    }


def parse_local_qwen_output(text: str) -> tuple[str, str] | None:
    """Parse Qwen thinking output; None if truncated or empty."""
    if "</think>" not in text:
        return None
    idx = text.index("</think>")
    before = text[:idx]
    after = text[idx + len("</think>") :]
    thought = before.replace("<think>", "").strip()
    answer = after.strip()
    if not thought or not answer:
        return None
    return thought, answer


def parse_together_output(text: str) -> tuple[str, str]:
    stripped = text.strip()
    match = _TOGETHER_PARSE_RE.match(stripped)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return stripped, stripped


def _cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


def _load_local_qwen(model_id: str) -> tuple[Any, Any]:
    """Load Qwen3 for local teacher generation (fp16 or 4-bit)."""
    import os

    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    logger.info("Loading %s ...", model_id)
    use_fp16_small = "0.6B" in model_id or "0.6b" in model_id

    if use_fp16_small:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    elif _cuda_available():
        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
            )
        except ImportError:
            logger.warning(
                "bitsandbytes not installed; falling back to CPU fp32 (slow)."
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
            )
            model.to("cpu")
    else:
        logger.warning(
            "CUDA unavailable for %s; falling back to CPU fp32 (slow).",
            model_id,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        )
        model.to("cpu")

    return model, tokenizer


def generate_local_qwen(
    question: str,
    model_id: str,
    model: Any,
    tokenizer: Any,
) -> tuple[str, str] | None:
    import torch

    messages = [{"role": "user", "content": question}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    enc = tokenizer(prompt_text, return_tensors="pt")
    enc.pop("token_type_ids", None)
    device = next(model.parameters()).device
    inputs = {
        k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=768,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    new_tokens = outputs[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    return parse_local_qwen_output(text)


def _together_message_text(message: Any) -> str:
    """Extract usable text from Together chat completion message."""
    content = (message.content or "").strip()
    reasoning = (getattr(message, "reasoning", None) or "").strip()
    if content and reasoning:
        return f"{reasoning}\n{content}"
    if content:
        return content
    return reasoning


def generate_together(
    question: str,
    model_id: str,
    api_key: str,
) -> tuple[str, str] | None:
    from openai import OpenAI

    client = OpenAI(
        base_url="https://api.together.xyz/v1",
        api_key=api_key,
    )
    user_content = (
        question + " Reason step by step, then end with 'The answer is <number>.'"
    )
    create_kwargs: dict[str, Any] = {
        "model": model_id,
        "temperature": 0.3,
        "max_tokens": 512,
        "messages": [{"role": "user", "content": user_content}],
    }
    # Qwen3.5 on Together emits an empty content field when reasoning mode is on.
    if "Qwen3.5" in model_id or "397B" in model_id:
        create_kwargs["extra_body"] = {"reasoning": {"enabled": False}}

    response = client.chat.completions.create(**create_kwargs)
    text = _together_message_text(response.choices[0].message)
    if not text:
        return None
    thought, answer = parse_together_output(text)
    if not thought or not answer:
        return None
    return thought, answer


def run_teacher(
    name: str,
    problems: list[tuple[str, str]],
    out_dir: Path,
    together_key: str | None,
) -> float:
    """Generate and write ``r2_{name}.jsonl``; return keep rate."""
    if name not in TEACHER_REGISTRY:
        raise KeyError(f"Unknown teacher: {name}")

    spec = TEACHER_REGISTRY[name]
    kind = spec["kind"]
    model_id = spec["model"]

    model: Any = None
    tokenizer: Any = None
    if kind == "local_qwen":
        model, tokenizer = _load_local_qwen(model_id)
    if kind == "together" and not together_key:
        raise ValueError(
            f"Together API key required for teacher '{name}'. "
            "Pass --together-key or set TOGETHER_API_KEY."
        )

    kept: list[dict[str, str]] = []
    n = len(problems)

    for i, (question, gold) in enumerate(problems):
        logger.info("[%s] problem %d/%d", name, i + 1, n)
        parsed: tuple[str, str] | None = None
        if kind == "local_qwen":
            parsed = generate_local_qwen(question, model_id, model, tokenizer)
        else:
            parsed = generate_together(question, model_id, together_key or "")

        if parsed is None:
            continue
        thought, answer = parsed
        if not is_correct(answer, gold):
            continue
        kept.append(make_sft_row(question, thought, answer))

    out_path = out_dir / f"r2_{name}.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    keep_rate = len(kept) / n if n else 0.0
    print(f"{name}: kept {len(kept)}/{n} (keep-rate {keep_rate:.1%}) -> {out_path}")
    return keep_rate


def logic_check() -> bool:
    """Validate constants and row format without loading models."""
    problems = make_word_problems(4, SEED_DEFAULT)
    if len(problems) != 4:
        print("logic-check failed: expected 4 problems", file=sys.stderr)
        return False

    q, gold = problems[0]
    row = make_sft_row(q, "Start with 50, subtract 10: 40.", "The answer is 40.")
    expected_prompt_prefix = f"[SYSTEM]: {SYS_REASON}\n\n[USER]:"
    if not row["prompt"].startswith(expected_prompt_prefix):
        print("logic-check failed: prompt prefix mismatch", file=sys.stderr)
        return False
    if THINK_OPEN not in row["prompt"]:
        print("logic-check failed: missing THINK_OPEN in prompt", file=sys.stderr)
        return False
    if "<|end_of_thought|>" not in row["completion"]:
        print("logic-check failed: missing end_thought in completion", file=sys.stderr)
        return False
    if not is_correct("The answer is 5.", "5"):
        print("logic-check failed: is_correct regression", file=sys.stderr)
        return False
    if is_correct("The answer is 6.", "5"):
        print(
            "logic-check failed: is_correct should reject wrong answer", file=sys.stderr
        )
        return False

    parsed = parse_local_qwen_output("<think>2+3=5</think> The answer is 5.")
    if parsed is None or parsed[0] != "2+3=5":
        print("logic-check failed: parse_local_qwen_output", file=sys.stderr)
        return False

    thought, answer = parse_together_output("Step one.\nThe answer is 42.")
    if "answer" not in answer.lower():
        print("logic-check failed: parse_together_output", file=sys.stderr)
        return False

    print("logic-check: OK (constants, parsers, row format)")
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate R2 teacher SFT JSONL caches for Colab.",
    )
    parser.add_argument(
        "--teacher",
        action="append",
        dest="teachers",
        metavar="NAME",
        help="Teacher name (repeatable) or 'all'. "
        f"Choices: {', '.join(TEACHER_REGISTRY)}.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=N_PROBLEMS_DEFAULT,
        help=f"Number of word problems (default {N_PROBLEMS_DEFAULT}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED_DEFAULT,
        help=f"RNG seed for problem generation (default {SEED_DEFAULT}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./teacher_cache"),
        help="Output directory (default ./teacher_cache).",
    )
    parser.add_argument(
        "--together-key",
        default=None,
        help="Together API key (default: TOGETHER_API_KEY env).",
    )
    parser.add_argument(
        "--logic-check",
        action="store_true",
        help="Validate constants/parsers without downloading models.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if args.logic_check:
        return 0 if logic_check() else 1

    teachers = args.teachers or ["all"]
    if len(teachers) == 1 and teachers[0] == "all":
        teacher_names = list(TEACHER_REGISTRY.keys())
    else:
        teacher_names = teachers
        for name in teacher_names:
            if name not in TEACHER_REGISTRY:
                print(f"Unknown teacher: {name}", file=sys.stderr)
                return 1

    together_key = args.together_key
    if together_key is None:
        import os

        together_key = os.environ.get("TOGETHER_API_KEY")

    problems = make_word_problems(args.n, args.seed)
    out_dir = args.out_dir.resolve()

    for name in teacher_names:
        try:
            run_teacher(name, problems, out_dir, together_key)
        except (ValueError, KeyError) as exc:
            logger.error("%s: %s", name, exc)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
