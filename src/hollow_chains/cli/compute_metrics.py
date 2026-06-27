"""Typer CLI for computing hollow-chains metrics."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from hollow_chains.config import load_config
from hollow_chains.data.schema import load_jsonl
from hollow_chains.metrics.gap import gap_report
from hollow_chains.metrics.parse import tags_from_config
from hollow_chains.metrics.structural import parse_rate

app = typer.Typer(
    name="compute-metrics",
    help="Compute Structural Fidelity, Semantic Correctness, and gap metrics.",
    add_completion=False,
)


def _print_summary_table(report_dict: dict) -> None:
    """Print a compact summary table to stdout."""
    counts = report_dict["four_way_counts"]
    typer.echo("=" * 52)
    typer.echo(f"  parse_rate      {report_dict.get('parse_rate', 0.0):.4f}")
    typer.echo(f"  SF (aggregate)  {report_dict['sf_aggregate']:.4f}")
    typer.echo(f"  SC (aggregate)  {report_dict['sc_aggregate']:.4f}")
    typer.echo(f"  FSG             {report_dict['fsg']:.4f}")
    typer.echo(f"  theater_score   {report_dict['theater_score']:.4f}")
    typer.echo("  four-way counts:")
    for label, count in counts.items():
        typer.echo(f"    {label}: {count}")
    typer.echo("=" * 52)


@app.command()
def main(
    records: Path = typer.Option(
        ...,
        "--records",
        help="Path to input JSONL of GenerationRecords.",
    ),
    config: Path = typer.Option(
        ...,
        "--config",
        help="Path to metrics YAML config.",
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        help="Path to output JSON report.",
    ),
) -> None:
    """Load records, compute SF/SC/gap metrics, write JSON report."""
    cfg = load_config(config)
    data = load_jsonl(records)

    report = gap_report(data, config=cfg)
    report_dict = report.to_dict()

    tags = tags_from_config(cfg)
    report_dict["parse_rate"] = parse_rate(data, tags)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(report_dict, fh, indent=2)

    _print_summary_table(report_dict)
    typer.echo(f"Report written to {out}")


if __name__ == "__main__":
    app()
