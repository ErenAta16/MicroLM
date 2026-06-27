#!/usr/bin/env python3
"""Train the frozen ByteLevel BPE tokenizer."""

from __future__ import annotations

from pathlib import Path

import typer

from hollow_chains.config import load_config
from hollow_chains.data.pretrain_data import get_corpus_texts, synthetic_corpus
from hollow_chains.data.tokenizer import train_tokenizer
from hollow_chains.utils.seed import set_seed

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Path = typer.Option(
        Path("configs/tokenizer.yaml"),
        "--config",
        help="Tokenizer config path.",
    ),
    smoke: bool = typer.Option(False, "--smoke", help="Use synthetic corpus only."),
) -> None:
    """Train tokenizer and save to artifacts/tokenizer/."""
    cfg = load_config(config)
    set_seed(int(cfg.get("seed", 42)) if "seed" in cfg else 42)
    texts = synthetic_corpus() if smoke else get_corpus_texts(cfg, smoke=smoke)
    train_cfg = cfg.get("training", cfg)
    out = train_tokenizer(
        texts,
        cfg.get("output_dir", "artifacts/tokenizer"),
        vocab_size=int(cfg.get("vocab_size", 16000)),
        min_frequency=int(train_cfg.get("min_frequency", 2)),
        config=cfg,
    )
    typer.echo(f"Tokenizer saved to {out}")


if __name__ == "__main__":
    app()
