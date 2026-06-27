"""End-to-end CPU smoke test for M2 train → generate → metrics loop."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from hollow_chains.config import load_config
from hollow_chains.data.build_reasoning_sft import build_sft_jsonl
from hollow_chains.data.pretrain_data import materialize_shard, synthetic_corpus
from hollow_chains.data.schema import GenerationRecord, load_jsonl
from hollow_chains.data.tokenizer import load_tokenizer, train_tokenizer
from hollow_chains.eval.generate import (
    EvalTask,
    generate_records,
    write_generation_jsonl,
)
from hollow_chains.metrics.gap import gap_report
from hollow_chains.metrics.semantic import semantic_correctness
from hollow_chains.metrics.structural import structural_fidelity
from hollow_chains.train.pretrain import run_pretrain
from hollow_chains.train.sft import run_sft


@pytest.fixture
def smoke_root(tmp_path: Path) -> Path:
    return tmp_path / "smoke"


def test_pipeline_smoke_end_to_end(smoke_root: Path) -> None:
    """Train tiny model, SFT, generate, validate M1 round-trip (CPU, no network)."""
    repo = Path(__file__).parent.parent
    smoke_cfg = load_config(repo / "configs" / "smoke.yaml")
    pretrain_cfg = load_config(repo / "configs" / "pretrain.yaml")
    sft_cfg = load_config(repo / "configs" / "sft.yaml")
    metrics_cfg = load_config(repo / "configs" / "metrics.yaml")

    tok_dir = smoke_root / "tokenizer"
    shard_dir = smoke_root / "shards"
    ckpt_root = smoke_root / "checkpoints"
    gen_path = smoke_root / "generations.jsonl"

    # 1. Micro tokenizer on synthetic corpus
    texts = synthetic_corpus()
    vocab_size = int(smoke_cfg.get("tokenizer", {}).get("vocab_size", 512))
    train_tokenizer(texts, tok_dir, vocab_size=vocab_size)
    tokenizer = load_tokenizer(tok_dir)

    # 2. Materialize tiny shard
    ctx = int(smoke_cfg.get("pretrain", {}).get("context_len", 64))
    materialize_shard(texts, tokenizer, shard_dir, context_len=ctx)

    # 3. Pretrain 2 steps
    pretrain_cfg = {
        **pretrain_cfg,
        "seed": int(smoke_cfg.get("seed", 0)),
        "tokenizer": {"path": str(tok_dir)},
        "data": {
            "context_len": ctx,
            "shard_dir": str(shard_dir),
            "smoke_use_synthetic": True,
        },
        "training": {
            **pretrain_cfg.get("training", {}),
            "rung": "tiny_1m",
            "smoke_max_steps": 2,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
        },
        "paths": {"checkpoint_dir": str(ckpt_root / "pretrain")},
        "model_ladder": {"config": str(repo / "configs" / "model_ladder.yaml")},
    }
    pretrain_ckpt = run_pretrain(pretrain_cfg, smoke=True)

    # 4. SFT data + 1 step
    sft_cfg_run = {
        **sft_cfg,
        "seed": int(smoke_cfg.get("seed", 0)),
        "tokenizer": {"path": str(tok_dir)},
        "data": {"output_dir": str(smoke_root / "sft_data"), "seed_problems": 10},
        "training": {
            **sft_cfg.get("training", {}),
            "smoke_max_steps": 1,
            "max_seq_len": int(smoke_cfg.get("sft", {}).get("max_seq_len", 64)),
            "per_device_train_batch_size": 1,
        },
        "paths": {
            "sft_checkpoint_root": str(ckpt_root),
            "runs_dir": "runs",
        },
        "model_ladder": {"config": str(repo / "configs" / "model_ladder.yaml")},
    }
    sft_jsonl, pass_rate = build_sft_jsonl(
        sft_cfg_run, teacher="heuristic", samples=2, fmt="full", smoke=True
    )
    assert pass_rate > 0.0

    sft_ckpt = run_sft(
        sft_cfg_run,
        rung="tiny_1m",
        teacher="heuristic",
        samples=2,
        pretrain_checkpoint=pretrain_ckpt,
        sft_jsonl=sft_jsonl,
        smoke=True,
    )

    # 5. Generate one record
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(sft_ckpt))
    tasks = [
        EvalTask(
            id="smoke_001",
            prompt="Tom has 5 apples. He buys 3 more. How many?",
            gold="8",
            task_type="arithmetic",
        )
    ]
    decoding = {
        "do_sample": False,
        "max_new_tokens": int(smoke_cfg.get("generate", {}).get("max_new_tokens", 64)),
        "capture_token_entropies": True,
    }
    records = generate_records(model, tokenizer, tasks, decoding, model_id="smoke")
    write_generation_jsonl(records, gen_path)

    # 6. Schema round-trip
    loaded = load_jsonl(gen_path)
    assert len(loaded) == 1
    assert isinstance(loaded[0], GenerationRecord)

    # 7. M1 metrics (unchanged behavior)
    ref = metrics_cfg.get(
        "reference",
        {
            "section_ratios": [0.6, 0.65, 0.7],
            "lengths": [20.0, 30.0, 40.0],
        },
    )
    sf = structural_fidelity(
        loaded,
        reference=ref,
        templates=metrics_cfg.get("teacher_opening_templates", []),
        weights=metrics_cfg.get("structural_fidelity", {}).get("weights"),
    )
    sc = semantic_correctness(
        loaded,
        weights=metrics_cfg.get("semantic_correctness", {}).get("weights"),
    )
    report = gap_report(loaded, config=metrics_cfg)
    assert 0.0 <= sf.score <= 1.0
    assert 0.0 <= sc.score <= 1.0
    assert 0.0 <= report.sf_aggregate <= 1.0
    assert 0.0 <= report.sc_aggregate <= 1.0
    assert report.fsg == pytest.approx(
        report.sf_aggregate - report.sc_aggregate, abs=1e-6
    )
    assert "four_way_counts" in report.to_dict()
