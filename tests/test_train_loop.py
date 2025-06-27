"""Tests for training loop."""

import pytest
import torch

from eurosat.model import build_model
from eurosat.train_loop import train_epoch


@pytest.mark.slow
def test_train_epoch_runs(eurosat_subset) -> None:
    """Happy path: single epoch runs on real data subset."""
    model = build_model({})
    loader = torch.utils.data.DataLoader(eurosat_subset, batch_size=4, shuffle=False)
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    loss = train_epoch(model, loader, optim)
    assert loss >= 0.0


def test_train_epoch_empty() -> None:
    """Edge case: empty dataset."""
    model = build_model({})
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(ZeroDivisionError):
        train_epoch(model, [], optim)
