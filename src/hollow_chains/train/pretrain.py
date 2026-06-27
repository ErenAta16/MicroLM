"""Causal language model pretraining."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hollow_chains.config import load_config
from hollow_chains.data.pretrain_data import (
    PretrainShardDataset,
    get_corpus_texts,
    materialize_shard,
    synthetic_corpus,
)
from hollow_chains.data.tokenizer import load_tokenizer, train_tokenizer
from hollow_chains.models.ladder import build_model, count_params
from hollow_chains.utils.seed import set_seed


def _compute_max_steps(config: dict[str, Any], num_params: int) -> int:
    """Derive training steps from token budget config."""
    tb = config.get("token_budget", {})
    mode = tb.get("mode", "tokens_per_param")
    train_cfg = config.get("training", {})
    batch = int(train_cfg.get("per_device_train_batch_size", 4))
    accum = int(train_cfg.get("gradient_accumulation_steps", 8))
    ctx = int(config.get("data", {}).get("context_len", 512))

    if mode == "absolute":
        total_tokens = int(tb.get("max_tokens", 1_500_000_000))
    else:
        tpp = float(tb.get("tokens_per_param", 20))
        total_tokens = min(
            int(num_params * tpp), int(tb.get("max_tokens", 1_500_000_000))
        )

    tokens_per_step = batch * accum * ctx
    return max(1, total_tokens // tokens_per_step)


def run_pretrain(
    config: dict[str, Any],
    *,
    smoke: bool = False,
) -> Path:
    """Run causal LM pretraining (or smoke pass).

    Args:
        config: Parsed pretrain.yaml (or smoke overrides).
        smoke: If True, use synthetic corpus and smoke_max_steps.

    Returns:
        Path to final checkpoint directory.
    """
    import torch
    from torch.utils.data import Dataset
    from transformers import Trainer, TrainingArguments

    seed = int(config.get("seed", 42))
    set_seed(seed)

    tok_cfg_path = config.get("tokenizer", {}).get("config", "configs/tokenizer.yaml")
    tok_path = Path(config.get("tokenizer", {}).get("path", "artifacts/tokenizer"))
    ladder_cfg = load_config(
        config.get("model_ladder", {}).get("config", "configs/model_ladder.yaml")
    )

    data_cfg = config.get("data", {})
    ctx_len = int(data_cfg.get("context_len", 512))
    shard_dir = Path(data_cfg.get("shard_dir", "artifacts/pretrain_shards"))

    if (
        not (tok_path / "tokenizer.json").exists()
        and not (tok_path / "tokenizer_config.json").exists()
    ):
        tok_yaml = load_config(tok_cfg_path) if Path(tok_cfg_path).is_file() else {}
        texts = synthetic_corpus() if smoke else get_corpus_texts(tok_yaml, smoke=smoke)
        train_tokenizer(
            texts,
            tok_path,
            vocab_size=int(tok_yaml.get("vocab_size", 16000)),
            config=tok_yaml,
        )

    tokenizer = load_tokenizer(tok_path)
    if smoke or data_cfg.get("smoke_use_synthetic", False):
        texts = synthetic_corpus()
    else:
        texts = get_corpus_texts(config, smoke=False)

    materialize_shard(texts, tokenizer, shard_dir, context_len=ctx_len)
    dataset: Dataset = PretrainShardDataset(shard_dir)  # type: ignore[assignment]

    rung = config.get("training", {}).get("rung", "tiny_1m")
    model = build_model(rung_name=rung, ladder_config=ladder_cfg)
    model.resize_token_embeddings(len(tokenizer))

    n_params = count_params(model.config)
    train_cfg = config.get("training", {})
    max_steps = (
        int(train_cfg.get("smoke_max_steps", 2))
        if smoke
        else train_cfg.get("max_steps")
    )
    if max_steps is None:
        max_steps = _compute_max_steps(config, n_params)
    max_steps = int(max_steps)

    out_dir = Path(config.get("paths", {}).get("checkpoint_dir", "artifacts/pretrain"))
    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out_dir),
        max_steps=max_steps,
        per_device_train_batch_size=int(
            train_cfg.get("per_device_train_batch_size", 4)
        ),
        gradient_accumulation_steps=int(
            train_cfg.get("gradient_accumulation_steps", 8)
        ),
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.1)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.05)),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        save_steps=max(max_steps, 1),
        report_to=[],
        use_cpu=not torch.cuda.is_available(),
        fp16=bool(train_cfg.get("fp16", False)) and torch.cuda.is_available(),
        seed=seed,
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.train()
    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    return final_dir
