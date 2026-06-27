"""Configuration loading for hollow-chains."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DRIVE_ROOT_ENV = "HOLLOW_CHAINS_DRIVE_ROOT"
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/MicroLM"
DRIVE_ROOT_PLACEHOLDER = "${DRIVE_ROOT}"


def _drive_root() -> str:
    """Return the resolved Google Drive project root."""
    return os.environ.get(DRIVE_ROOT_ENV, DEFAULT_DRIVE_ROOT)


def _resolve_placeholders(value: Any) -> Any:
    """Replace ``${DRIVE_ROOT}`` in config string values recursively."""
    root = _drive_root()
    if isinstance(value, str):
        return value.replace(DRIVE_ROOT_PLACEHOLDER, root)
    if isinstance(value, dict):
        return {k: _resolve_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_placeholders(v) for v in value]
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and return its contents as a dict.

    String values may contain ``${DRIVE_ROOT}``, resolved from
    ``HOLLOW_CHAINS_DRIVE_ROOT`` (default ``/content/drive/MyDrive/MicroLM``).

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary with placeholders expanded.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    return _resolve_placeholders(data)
