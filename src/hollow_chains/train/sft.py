"""Supervised fine-tuning on reasoning traces with prompt-loss masking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hollow_chains.config import load_config
from hollow_chains.data.build_reasoning_sft import load_sft_jsonl
from hollow_chains.data.tokenizer import load_tokenizer
from hollow_chains.models.ladder import build_model
from hollow_chains.utils.seed import set_seed


class SFTDataset:
    """SFT dataset with prompt-loss masking (loss only on completion tokens)."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer,
        max_seq_len: int = 512,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        prompt = rec["prompt"]
        completion = rec["target_completion"]
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        full_text = prompt + completion
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=True)
        full_ids = full_ids[: self.max_seq_len]
        labels = full_ids.copy()
        prompt_len = min(len(prompt_ids) + 1, len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": [1] * len(full_ids),
        }


def _collate(batch: list[dict[str, Any]], pad_id: int = 0) -> dict[str, Any]:
    import torch

    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, attention_mask = [], [], []
    for item in batch:
        pad = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_id] * pad)
        labels.append(item["labels"] + [-100] * pad)
        attention_mask.append(item["attention_mask"] + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def run_sft(
    config: dict[str, Any],
    *,
    rung: str = "tiny_1m",
    teacher: str = "heuristic",
    samples: int = 2,
    epochs: int = 1,
    fmt: str = "full",
    pretrain_checkpoint: Path | None = None,
    sft_jsonl: Path | None = None,
    smoke: bool = False,
) -> Path:
    """Run SFT on reasoning traces for one sweep cell.

    Args:
        config: Parsed sft.yaml.
        rung: Model ladder rung name.
        teacher: Teacher key.
        samples: SFT sample count.
        epochs: Training epochs.
        fmt: ``full`` or ``equation_only``.
        pretrain_checkpoint: Base model checkpoint path.
        sft_jsonl: Pre-built SFT JSONL path.
        smoke: Use smoke_max_steps.

    Returns:
        Path to SFT checkpoint directory.
    """
    import torch
    from torch.utils.data import Dataset
    from transformers import Trainer, TrainingArguments

    seed = int(config.get("seed", 42))
    set_seed(seed)

    ladder_cfg = load_config(
        config.get("model_ladder", {}).get("config", "configs/model_ladder.yaml")
    )
    tok_path = Path(config.get("tokenizer", {}).get("path", "artifacts/tokenizer"))
    tokenizer = load_tokenizer(pretrain_checkpoint if pretrain_checkpoint else tok_path)

    if sft_jsonl is None:
        from hollow_chains.data.build_reasoning_sft import build_sft_jsonl

        sft_jsonl, _ = build_sft_jsonl(
            config, teacher=teacher, samples=samples, fmt=fmt, smoke=smoke  # type: ignore[arg-type]
        )

    records = load_sft_jsonl(sft_jsonl)
    train_cfg = config.get("training", {})
    max_seq = int(train_cfg.get("max_seq_len", 512))
    dataset: Dataset = SFTDataset(records, tokenizer, max_seq_len=max_seq)  # type: ignore[assignment]

    if pretrain_checkpoint and (pretrain_checkpoint / "config.json").exists():
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(str(pretrain_checkpoint))
    else:
        model = build_model(rung_name=rung, ladder_config=ladder_cfg)
        model.resize_token_embeddings(len(tokenizer))

    cell_id = f"{rung}_{teacher}_{samples}_{epochs}_{fmt}"
    root = Path(config.get("paths", {}).get("sft_checkpoint_root", "artifacts/sft"))
    runs_dir = Path(config.get("paths", {}).get("runs_dir", "runs"))
    out_dir = root / runs_dir / cell_id
    out_dir.mkdir(parents=True, exist_ok=True)

    max_steps = int(train_cfg.get("smoke_max_steps", 1)) if smoke else None
    if max_steps is None:
        batch = int(train_cfg.get("per_device_train_batch_size", 4))
        steps_per_epoch = max(1, len(dataset) // batch)
        max_steps = steps_per_epoch * int(epochs)

    args = TrainingArguments(
        output_dir=str(out_dir),
        max_steps=int(max_steps),
        per_device_train_batch_size=int(
            train_cfg.get("per_device_train_batch_size", 4)
        ),
        gradient_accumulation_steps=int(
            train_cfg.get("gradient_accumulation_steps", 4)
        ),
        learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        logging_steps=1,
        save_steps=max(int(max_steps), 1),
        report_to=[],
        use_cpu=not torch.cuda.is_available(),
        fp16=bool(train_cfg.get("fp16", False)) and torch.cuda.is_available(),
        seed=seed,
    )

    pad_id = tokenizer.pad_token_id or 0
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=lambda b: _collate(b, pad_id=pad_id),
    )
    trainer.train()
    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    return final_dir
