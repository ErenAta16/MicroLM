"""Create a minimal Colab notebook skeleton."""

from __future__ import annotations

import json
from pathlib import Path


def _nbformat(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
        },
        "cells": cells,
    }


def _md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.splitlines(True),
        "outputs": [],
        "execution_count": None,
    }


NOTEBOOKS: dict[str, list[dict]] = {
    "00_setup_colab.ipynb": [
        _md(
            "# 00 — Colab Setup\n\nMount Drive, install hollow-chains, train tokenizer, materialize pretrain shard."
        ),
        _code(
            "# CONFIG\n"
            "DRIVE = '/content/drive/MyDrive/hollow-chains'\n"
            "REPO = '/content/hollow-chains'\n"
            "SEED = 42"
        ),
        _code(
            "from google.colab import drive\n"
            "drive.mount('/content/drive')\n"
            "!git clone https://github.com/ErenAta16/MicroLM.git $REPO || true\n"
            "%cd $REPO\n"
            "!pip install -q -e '.[gpu]'"
        ),
        _code(
            "from hollow_chains.config import load_config\n"
            "from hollow_chains.data.tokenizer import train_tokenizer\n"
            "from hollow_chains.data.pretrain_data import get_corpus_texts, materialize_shard\n"
            "from hollow_chains.data.tokenizer import load_tokenizer\n"
            "from hollow_chains.utils.seed import set_seed\n"
            "\n"
            "set_seed(SEED)\n"
            "tok_cfg = load_config('configs/tokenizer.yaml')\n"
            "pre_cfg = load_config('configs/pretrain.yaml')\n"
            "texts = get_corpus_texts(tok_cfg)\n"
            "train_tokenizer(texts, f'{DRIVE}/artifacts/tokenizer', vocab_size=tok_cfg['vocab_size'], config=tok_cfg)\n"
            "tok = load_tokenizer(f'{DRIVE}/artifacts/tokenizer')\n"
            "materialize_shard(texts[:50000], tok, f'{DRIVE}/artifacts/pretrain_shards', context_len=512)\n"
            "print('Artifacts:', f'{DRIVE}/artifacts/tokenizer', f'{DRIVE}/artifacts/pretrain_shards')"
        ),
    ],
    "01_pretrain_ladder.ipynb": [
        _md(
            "# 01 — Pretrain Ladder\n\nPretrain each rung from `configs/model_ladder.yaml`."
        ),
        _code(
            "DRIVE = '/content/drive/MyDrive/hollow-chains'\nREPO = '/content/hollow-chains'\n%cd $REPO"
        ),
        _code(
            "from hollow_chains.config import load_config\n"
            "from hollow_chains.train.pretrain import run_pretrain\n"
            "from hollow_chains.utils.seed import set_seed\n"
            "\n"
            "cfg = load_config('configs/pretrain.yaml')\n"
            "cfg['paths']['checkpoint_dir'] = f'{DRIVE}/checkpoints/pretrain'\n"
            "cfg['tokenizer']['path'] = f'{DRIVE}/artifacts/tokenizer'\n"
            "cfg['data']['shard_dir'] = f'{DRIVE}/artifacts/pretrain_shards'\n"
            "set_seed(cfg['seed'])\n"
            "for rung in ['tiny_1m','small_8m','mid_50m','large_150m']:\n"
            "    cfg['training']['rung'] = rung\n"
            "    ckpt = run_pretrain(cfg)\n"
            "    print(rung, '->', ckpt)"
        ),
    ],
    "02_sft_sweeps.ipynb": [
        _md(
            "# 02 — SFT Sweeps\n\nBuild SFT data and run one-axis sweep cells from `configs/sft.yaml`."
        ),
        _code(
            "DRIVE = '/content/drive/MyDrive/hollow-chains'\n%cd /content/hollow-chains"
        ),
        _code(
            "from hollow_chains.config import load_config\n"
            "from hollow_chains.data.build_reasoning_sft import build_sft_jsonl\n"
            "from hollow_chains.train.sft import run_sft\n"
            "\n"
            "cfg = load_config('configs/sft.yaml')\n"
            "cfg['paths']['pretrain_checkpoint_root'] = f'{DRIVE}/checkpoints/pretrain'\n"
            "cfg['paths']['sft_checkpoint_root'] = f'{DRIVE}/checkpoints/sft'\n"
            "for cell in cfg['run_cells']:\n"
            "    sft_path, rate = build_sft_jsonl(cfg, teacher=cell['teacher'], samples=cell['samples'], fmt=cell['format'])\n"
            "    pretrain_ckpt = f\"{DRIVE}/checkpoints/pretrain/{cell['rung']}/final\"\n"
            "    run_sft(cfg, rung=cell['rung'], teacher=cell['teacher'], samples=cell['samples'], epochs=cell['epochs'], fmt=cell['format'], pretrain_checkpoint=pretrain_ckpt, sft_jsonl=sft_path)"
        ),
    ],
    "03_generate_emergence.ipynb": [
        _md(
            "# 03 — Generate & Metrics\n\nRun emergence driver and M1 compute-metrics."
        ),
        _code(
            "DRIVE = '/content/drive/MyDrive/hollow-chains'\n%cd /content/hollow-chains"
        ),
        _code(
            "from hollow_chains.eval.run_emergence import run_emergence\n"
            "\n"
            "manifest = run_emergence('configs/sft.yaml', 'configs/generate.yaml', run_metrics=True)\n"
            "print('Manifest:', manifest)\n"
            "print('JSONL + metrics written under generate.yaml output_dir')"
        ),
    ],
}


def write_notebooks(base: Path) -> None:
    base.mkdir(parents=True, exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        (base / name).write_text(
            json.dumps(_nbformat(cells), indent=1), encoding="utf-8"
        )


if __name__ == "__main__":
    write_notebooks(Path(__file__).resolve().parent.parent / "notebooks")
