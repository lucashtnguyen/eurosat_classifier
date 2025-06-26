"""Tests for model building."""

import pytest
import torch

from eurosat.model import build_model


def test_build_model_default() -> None:
    """Happy path: build default model."""
    model = build_model({})
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    assert out.shape[0] == 1
    assert out.shape[1] == 10


def test_build_model_unknown() -> None:
    """Edge case: unknown model name."""
    with pytest.raises(ValueError):
        build_model({"name": "unknown"})
