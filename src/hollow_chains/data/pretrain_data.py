"""Pretraining corpus streaming, packing, and shard materialization."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

SYNTHETIC_CORPUS: list[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "Reasoning traces help models explain their answers step by step.",
    "Arithmetic: five plus three equals eight.",
    "Symbolic algebra combines like terms to simplify expressions.",
    "FineWeb-Edu provides high quality educational web text for training.",
    "Small language models can learn structure before they learn facts.",
    "ByteLevel BPE tokenizers handle unicode and punctuation robustly.",
    "Causal language modeling predicts the next token in a sequence.",
    "RoPE positional embeddings scale to longer context windows.",
    "Tied embeddings share weights between input and output projections.",
] * 20


def synthetic_corpus() -> list[str]:
    """Return a deterministic synthetic corpus for CPU smoke tests."""
    return list(SYNTHETIC_CORPUS)


def stream_fineweb_edu(
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    subset: str = "sample-10BT",
    text_field: str = "text",
    *,
    max_texts: int | None = None,
) -> Iterator[str]:
    """Stream text from FineWeb-Edu (requires network).

    Args:
        dataset_name: HuggingFace dataset id.
        subset: Dataset subset/config name.
        text_field: Column containing raw text.
        max_texts: Optional cap on number of documents.

    Yields:
        Text strings from the dataset.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, subset, split="train", streaming=True)
    for i, row in enumerate(ds):
        if max_texts is not None and i >= max_texts:
            break
        yield row[text_field]


def get_corpus_texts(config: dict[str, Any], *, smoke: bool = False) -> list[str]:
    """Load corpus texts for tokenizer training or shard building.

    Args:
        config: Pretrain or tokenizer config dict.
        smoke: If True, return synthetic corpus (no network).

    Returns:
        List of text documents.
    """
    if smoke:
        return synthetic_corpus()

    ds_cfg = config.get("dataset", config)
    sample_count = int(
        config.get("training", {}).get("sample_texts")
        or config.get("sample_texts", 50000)
    )
    if smoke:
        sample_count = min(
            sample_count,
            int(config.get("training", {}).get("smoke_sample_texts", 200)),
        )

    texts = []
    for i, text in enumerate(
        stream_fineweb_edu(
            dataset_name=ds_cfg.get("name", "HuggingFaceFW/fineweb-edu"),
            subset=ds_cfg.get("subset", "sample-10BT"),
            text_field=ds_cfg.get("text_field", "text"),
        )
    ):
        texts.append(text)
        if i + 1 >= sample_count:
            break
    return texts


def pack_sequences(
    token_ids: list[int],
    context_len: int,
) -> np.ndarray:
    """Pack token ids into fixed-length sequences.

    Args:
        token_ids: Flat list of token ids.
        context_len: Sequence length.

    Returns:
        2-D int32 array of shape (num_sequences, context_len).
    """
    n_seq = len(token_ids) // context_len
    if n_seq == 0:
        padded = token_ids + [0] * (context_len - len(token_ids))
        return np.array([padded], dtype=np.int32)
    trimmed = token_ids[: n_seq * context_len]
    return np.array(trimmed, dtype=np.int32).reshape(n_seq, context_len)


def materialize_shard(
    texts: list[str],
    tokenizer,
    shard_dir: str | Path,
    *,
    context_len: int = 512,
) -> Path:
    """Tokenize, pack, and save a pretrain shard to disk.

    Args:
        texts: Raw corpus documents.
        tokenizer: HuggingFace tokenizer.
        shard_dir: Output directory for memmap shard.
        context_len: Packed sequence length.

    Returns:
        Path to shard directory.
    """
    out = Path(shard_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_ids: list[int] = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)

    packed = pack_sequences(all_ids, context_len)
    memmap_path = out / "input_ids.memmap"
    mmap = np.memmap(memmap_path, dtype=np.int32, mode="w+", shape=packed.shape)
    mmap[:] = packed
    mmap.flush()

    meta = {
        "num_sequences": int(packed.shape[0]),
        "context_len": context_len,
        "num_tokens": int(packed.size),
    }
    (out / "meta.json").write_text(
        __import__("json").dumps(meta, indent=2), encoding="utf-8"
    )
    return out


class PretrainShardDataset:
    """Minimal dataset over a materialized memmap shard."""

    def __init__(self, shard_dir: str | Path) -> None:
        import json

        self.shard_dir = Path(shard_dir)
        meta = json.loads((self.shard_dir / "meta.json").read_text(encoding="utf-8"))
        shape = (meta["num_sequences"], meta["context_len"])
        self.data = np.memmap(
            self.shard_dir / "input_ids.memmap",
            dtype=np.int32,
            mode="r",
            shape=shape,
        )

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        row = self.data[idx].tolist()
        return {"input_ids": row, "labels": row.copy()}
