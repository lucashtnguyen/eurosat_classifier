"""Tests for training loop."""

import pytest
import torch

from eurosat.model import build_model
from eurosat.train_loop import train_epoch


def test_train_epoch_runs() -> None:
    """Happy path: single epoch runs."""
    model = build_model({})
    data = [
        (torch.randn(1, 3, 64, 64), torch.tensor([0])),
        (torch.randn(1, 3, 64, 64), torch.tensor([1])),
    ]
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    loss = train_epoch(model, data, optim)
    assert loss >= 0.0


def test_train_epoch_empty() -> None:
    """Edge case: empty dataset."""
    model = build_model({})
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(ZeroDivisionError):
        train_epoch(model, [], optim)
