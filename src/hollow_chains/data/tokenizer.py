"""ByteLevel BPE tokenizer with M1 reasoning-tag special tokens."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hollow_chains.metrics.parse import (
    TAG_BEGIN_SOLUTION,
    TAG_BEGIN_THOUGHT,
    TAG_END_SOLUTION,
    TAG_END_THOUGHT,
)

# M1 tag strings — single source of truth.
REASONING_TAGS = (
    TAG_BEGIN_THOUGHT,
    TAG_END_THOUGHT,
    TAG_BEGIN_SOLUTION,
    TAG_END_SOLUTION,
)

DEFAULT_SPECIAL = {
    "pad": "<|pad|>",
    "bos": "<|bos|>",
    "eos": "<|eos|>",
}


def reasoning_tag_tokens() -> dict[str, str]:
    """Return reasoning tag special-token mapping."""
    return {
        "begin_thought": TAG_BEGIN_THOUGHT,
        "end_thought": TAG_END_THOUGHT,
        "begin_solution": TAG_BEGIN_SOLUTION,
        "end_solution": TAG_END_SOLUTION,
    }


def all_special_tokens(config: dict[str, Any] | None = None) -> list[str]:
    """Collect all special token strings for training."""
    cfg = config or {}
    specials = cfg.get("special_tokens", DEFAULT_SPECIAL)
    tokens = [
        specials.get("pad", DEFAULT_SPECIAL["pad"]),
        specials.get("bos", DEFAULT_SPECIAL["bos"]),
        specials.get("eos", DEFAULT_SPECIAL["eos"]),
    ]
    tag_cfg = specials if "begin_thought" in specials else reasoning_tag_tokens()
    tokens.extend(
        [
            tag_cfg.get("begin_thought", TAG_BEGIN_THOUGHT),
            tag_cfg.get("end_thought", TAG_END_THOUGHT),
            tag_cfg.get("begin_solution", TAG_BEGIN_SOLUTION),
            tag_cfg.get("end_solution", TAG_END_SOLUTION),
        ]
    )
    return tokens


def train_tokenizer(
    texts: list[str],
    output_dir: str | Path,
    *,
    vocab_size: int = 16000,
    min_frequency: int = 2,
    config: dict[str, Any] | None = None,
) -> Path:
    """Train a ByteLevel BPE tokenizer and save to disk.

    Special tokens (including M1 reasoning tags) are never split.

    Args:
        texts: Corpus strings for BPE training.
        output_dir: Directory to write tokenizer files.
        vocab_size: Target vocabulary size.
        min_frequency: Minimum token frequency.
        config: Optional tokenizer.yaml dict.

    Returns:
        Path to the output directory.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    special = all_special_tokens(config)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special,
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)

    raw_path = out / "tokenizer.json"
    tokenizer.save(str(raw_path))

    cfg = config or {}
    specials = cfg.get("special_tokens", {})
    pad = specials.get("pad", DEFAULT_SPECIAL["pad"])
    bos = specials.get("bos", DEFAULT_SPECIAL["bos"])
    eos = specials.get("eos", DEFAULT_SPECIAL["eos"])

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=bos,
        eos_token=eos,
        pad_token=pad,
        additional_special_tokens=list(REASONING_TAGS),
    )
    hf_tok.save_pretrained(out)

    meta = {
        "vocab_size": vocab_size,
        "special_tokens": special,
        "reasoning_tags": list(REASONING_TAGS),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def load_tokenizer(path: str | Path):
    """Load a saved HuggingFace tokenizer.

    Args:
        path: Directory containing tokenizer files.

    Returns:
        PreTrainedTokenizerFast instance.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path), trust_remote_code=False)
