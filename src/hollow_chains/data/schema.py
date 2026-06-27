"""Pydantic schema and JSONL I/O for generation records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class GenerationRecord(BaseModel):
    """A single model generation sample with optional parsed fields."""

    id: str
    prompt: str
    task_type: Literal["arithmetic", "symbolic", "factual_mcq"]
    gold: str
    generation: str
    think: str | None = None
    solution: str | None = None
    token_entropies: list[float] | None = None
    meta: dict = Field(default_factory=dict)


def load_jsonl(path: str | Path) -> list[GenerationRecord]:
    """Load and validate generation records from a JSONL file.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of validated GenerationRecord objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        pydantic.ValidationError: If any line fails schema validation.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")
    records: list[GenerationRecord] = []
    with file_path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
                records.append(GenerationRecord.model_validate(data))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
    return records


def dump_jsonl(records: list[GenerationRecord], path: str | Path) -> None:
    """Write generation records to a JSONL file.

    Args:
        records: Records to serialize.
        path: Output file path.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.model_dump_json() + "\n")
