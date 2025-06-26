"""Tests for configuration utilities."""

from pathlib import Path

import pytest

from eurosat.config import load_config


def test_load_config_success(tmp_path: Path) -> None:
    """Happy path for loading config."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("a: 1")
    result = load_config(cfg)
    assert isinstance(result, dict)
    assert result["a"] == 1


def test_load_config_missing(tmp_path: Path) -> None:
    """Edge case: missing file."""
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")
