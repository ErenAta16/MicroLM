#!/usr/bin/env python3
"""Build paper figures from cached experiment JSON under data/results/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "data" / "results"
FIGURES_DIR = REPO_ROOT / "paper" / "figures"

FIGURE_MAP = {
    "fig_control.pdf": ("control_arith.json", "control_mcq.json"),
    "fig_teacher.pdf": ("r2clean.json",),
    "fig_entropy.pdf": ("entropy.json",),
    "fig_scale.pdf": (),  # R1 scale table; optional scale JSON when present
}


def _load_json(name: str) -> dict:
    path = RESULTS_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing cached result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_figures(out_dir: Path) -> list[Path]:
    """Render PDF figures; returns paths written."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Figure: control harness (arith + MCQ SF/SC bars)
    arith = _load_json("control_arith.json")
    mcq = _load_json("control_mcq.json")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, payload, title in (
        (axes[0], arith, "Arithmetic control"),
        (axes[1], mcq, "MCQ control"),
    ):
        sf = payload.get("sf_aggregate", payload.get("sf", 0))
        sc = payload.get("sc_aggregate", payload.get("sc", 0))
        ax.bar(["SF", "SC"], [sf, sc], color=["#4c72b0", "#55a868"])
        ax.set_ylim(0, 1)
        ax.set_title(title)
    fig.tight_layout()
    p = out_dir / "fig_control.pdf"
    fig.savefig(p)
    plt.close(fig)
    written.append(p)

    # Figure: teacher axis
    teacher = _load_json("r2clean.json")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    labels = teacher.get("teachers", list(teacher.get("by_teacher", {}).keys()))
    sf_vals = teacher.get("sf", [])
    sc_vals = teacher.get("sc", [])
    if labels and sf_vals and sc_vals:
        x = range(len(labels))
        w = 0.35
        ax.bar([i - w / 2 for i in x], sf_vals, width=w, label="SF")
        ax.bar([i + w / 2 for i in x], sc_vals, width=w, label="SC")
        ax.set_xticks(list(x), labels, rotation=20, ha="right")
        ax.legend()
    ax.set_ylim(0, 1)
    ax.set_title("Teacher axis (R2 clean)")
    fig.tight_layout()
    p = out_dir / "fig_teacher.pdf"
    fig.savefig(p)
    plt.close(fig)
    written.append(p)

    # Figure: entropy
    ent = _load_json("entropy.json")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    think = ent.get("think_entropy_mean")
    sol = ent.get("solution_entropy_mean")
    if think is not None and sol is not None:
        ax.bar(["think", "solution"], [think, sol], color=["#8172b2", "#ccb974"])
    ax.set_title("Mean token entropy by section")
    fig.tight_layout()
    p = out_dir / "fig_entropy.pdf"
    fig.savefig(p)
    plt.close(fig)
    written.append(p)

    # Scale figure placeholder when scale JSON exists
    scale_path = RESULTS_DIR / "scale.json"
    if scale_path.is_file():
        scale = json.loads(scale_path.read_text(encoding="utf-8"))
        fig, ax = plt.subplots(figsize=(5, 3.5))
        rungs = scale.get("rungs", [])
        ax.plot(rungs, scale.get("sf", []), marker="o", label="SF")
        ax.plot(rungs, scale.get("sc", []), marker="o", label="SC")
        ax.legend()
        ax.set_title("Scale axis")
        fig.tight_layout()
        p = out_dir / "fig_scale.pdf"
        fig.savefig(p)
        plt.close(fig)
        written.append(p)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper figures from cached JSON.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=FIGURES_DIR,
        help="Output directory for PDF figures.",
    )
    args = parser.parse_args()
    paths = build_figures(args.out_dir.resolve())
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
