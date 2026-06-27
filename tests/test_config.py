"""Tests for config placeholder resolution."""

from hollow_chains.config import load_config


def test_drive_root_placeholder_resolved(tmp_path, monkeypatch) -> None:
    """${DRIVE_ROOT} in YAML is expanded from HOLLOW_CHAINS_DRIVE_ROOT."""
    monkeypatch.setenv("HOLLOW_CHAINS_DRIVE_ROOT", "/tmp/test_micro_lm")
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text(
        "paths:\n  output_dir: ${DRIVE_ROOT}/generations\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg["paths"]["output_dir"] == "/tmp/test_micro_lm/generations"
