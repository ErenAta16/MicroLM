"""Emergence-axis evaluation driver over SFT sweep cells."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from hollow_chains.config import load_config
from hollow_chains.eval.generate import (
    generate_records,
    load_eval_tasks,
    load_model_and_tokenizer,
    write_generation_jsonl,
)
from hollow_chains.utils.seed import set_seed


def _cell_checkpoint_dir(config: dict[str, Any], cell: dict[str, Any]) -> Path:
    """Resolve SFT checkpoint path for a run cell."""
    paths = config.get("paths", {})
    root = Path(paths.get("sft_checkpoint_root", "artifacts/sft"))
    runs_dir = Path(paths.get("runs_dir", "runs"))
    rung = cell["rung"]
    teacher = cell["teacher"]
    samples = cell["samples"]
    epochs = cell["epochs"]
    fmt = cell["format"]
    cell_id = f"{rung}_{teacher}_{samples}_{epochs}_{fmt}"
    return root / runs_dir / cell_id / "final"


def run_emergence(
    sft_config_path: str | Path,
    generate_config_path: str | Path,
    *,
    run_metrics: bool = True,
    smoke: bool = False,
    cells: list[str] | None = None,
) -> Path:
    """Run generation for each SFT cell and optionally compute M1 metrics.

    Args:
        sft_config_path: Path to sft.yaml.
        generate_config_path: Path to generate.yaml.
        run_metrics: Shell out to compute-metrics per cell.
        smoke: Limit to first cell with minimal tasks.
        cells: Optional list of cell ids to run (default: all).

    Returns:
        Path to manifest CSV.
    """
    sft_cfg = load_config(sft_config_path)
    gen_cfg = load_config(generate_config_path)
    seed = int(gen_cfg.get("seed", sft_cfg.get("seed", 42)))
    set_seed(seed)

    run_cells = sft_cfg.get("run_cells", [])
    if cells:
        run_cells = [c for c in run_cells if c.get("id") in cells]
    if smoke:
        run_cells = run_cells[:1]

    out_root = Path(gen_cfg.get("paths", {}).get("output_dir", "artifacts/generations"))
    out_root.mkdir(parents=True, exist_ok=True)
    metrics_cfg = gen_cfg.get("paths", {}).get("metrics_config", "configs/metrics.yaml")
    task_sets = gen_cfg.get("task_sets", ["arithmetic"])
    if smoke:
        task_sets = ["arithmetic"]
    tasks = load_eval_tasks(task_sets)
    if smoke:
        tasks = tasks[:1]

    decoding = gen_cfg.get("decoding", {})
    tok_path = gen_cfg.get("tokenizer", {}).get("path", "artifacts/tokenizer")
    manifest_path = out_root / "manifest.csv"
    rows: list[dict[str, str]] = []

    for cell in run_cells:
        cell_id = cell.get("id", "unknown")
        ckpt = _cell_checkpoint_dir(sft_cfg, cell)
        if not ckpt.is_dir() and not smoke:
            rows.append(
                {
                    "cell_id": cell_id,
                    "jsonl": "",
                    "metrics_json": "",
                    "status": f"missing_checkpoint:{ckpt}",
                }
            )
            continue

        if smoke and not ckpt.is_dir():
            # Smoke test provides checkpoint externally
            pass

        model, tokenizer = load_model_and_tokenizer(
            ckpt,
            own_tokenizer_path=tok_path,
        )
        records = generate_records(
            model,
            tokenizer,
            tasks,
            decoding,
            model_id=cell_id,
        )
        jsonl_path = out_root / f"{cell_id}.jsonl"
        write_generation_jsonl(records, jsonl_path)

        metrics_path = out_root / f"{cell_id}_metrics.json"
        status = "ok"
        if run_metrics and jsonl_path.is_file():
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "hollow_chains.cli.compute_metrics",
                        "--records",
                        str(jsonl_path),
                        "--config",
                        str(metrics_cfg),
                        "--out",
                        str(metrics_path),
                    ],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as exc:
                status = f"metrics_failed:{exc.returncode}"

        rows.append(
            {
                "cell_id": cell_id,
                "jsonl": str(jsonl_path),
                "metrics_json": str(metrics_path) if metrics_path.is_file() else "",
                "status": status,
            }
        )

    with manifest_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["cell_id", "jsonl", "metrics_json", "status"]
        )
        writer.writeheader()
        writer.writerows(rows)

    (out_root / "manifest.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    return manifest_path
