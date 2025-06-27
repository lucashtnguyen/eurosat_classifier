"""Tests for configuration utilities."""

from pathlib import Path

import pytest

from eurosat.config import load_config


def test_load_config_success(tmp_path: Path) -> None:
    """Happy path for loading config."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
model: {}
training: {}
data: {}
output: {}
"""
    )
    result = load_config(cfg)
    assert isinstance(result, dict)


def test_load_config_missing(tmp_path: Path) -> None:
    """Edge case: missing file."""
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_load_config_missing_section(tmp_path: Path) -> None:
    """Validation fails when a required section is absent."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("model: {}\ntraining: {}")
    with pytest.raises(KeyError):
        load_config(cfg)


def test_load_config_wrong_type(tmp_path: Path) -> None:
    """Validation fails when section is wrong type."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
model: []
training: {}
data: {}
output: {}
"""
    )
    with pytest.raises(TypeError):
        load_config(cfg)
