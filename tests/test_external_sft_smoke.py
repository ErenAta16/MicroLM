"""CPU smoke test for external-base SFT on a tiny HF checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from hollow_chains.config import load_config
from hollow_chains.data.sft_format import format_sft_prompt, format_sft_target
from hollow_chains.train.sft import run_sft


def _write_canned_sft_jsonl(path: Path) -> None:
    examples = [
        {
            "prompt": format_sft_prompt("What is 2 + 3?"),
            "target_completion": format_sft_target("2 + 3 = 5", "The answer is 5."),
            "task_type": "arithmetic",
            "gold": "5",
            "teacher": "canned",
            "format": "full",
        },
        {
            "prompt": format_sft_prompt("Tom has 5 apples. He buys 3 more. How many?"),
            "target_completion": format_sft_target("5 + 3 = 8", "The answer is 8."),
            "task_type": "arithmetic",
            "gold": "8",
            "teacher": "canned",
            "format": "full",
        },
    ]
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex) + "\n")


def test_external_base_sft_smoke(tmp_path: Path) -> None:
    """One training step on EleutherAI/pythia-14m with canned SFT examples."""
    repo = Path(__file__).parent.parent
    scale_cfg = load_config(repo / "configs" / "scale_ladder.yaml")
    sft_jsonl = tmp_path / "canned_sft.jsonl"
    _write_canned_sft_jsonl(sft_jsonl)

    cfg = {
        **scale_cfg,
        "seed": 0,
        "training": {
            **scale_cfg.get("training", {}),
            "smoke_max_steps": 1,
            "max_seq_len": 256,
            "per_device_train_batch_size": 1,
        },
        "paths": {
            "sft_checkpoint_root": str(tmp_path / "sft"),
            "runs_dir": "runs",
        },
    }

    ckpt = run_sft(
        cfg,
        teacher="canned",
        samples=2,
        smoke=True,
        sft_jsonl=sft_jsonl,
        base_hf_id="EleutherAI/pythia-14m",
    )

    assert (ckpt / "config.json").is_file()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(str(ckpt))
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    prompt = format_sft_prompt("What is 1 + 1?")
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    assert "<|begin_of_thought|>" in text
