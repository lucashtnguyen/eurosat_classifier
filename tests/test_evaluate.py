"""Tests for evaluation utilities."""

import pytest
import torch
from torch.utils.data import DataLoader

from eurosat.evaluate import evaluate


class DummyModel(torch.nn.Module):
    def __init__(self, preds):
        super().__init__()
        self.preds = preds

    def forward(self, x):
        return self.preds.pop(0)


def test_evaluate_metrics() -> None:
    preds = [torch.tensor([[0.6, 0.4]]), torch.tensor([[0.3, 0.7]])]
    model = DummyModel(preds)
    data = DataLoader(
        [
            (torch.zeros(1, 2), torch.tensor(0)),
            (torch.zeros(1, 2), torch.tensor(1)),
        ],
        batch_size=1,
    )
    acc, f1, cm = evaluate(model, data)
    assert acc == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0)
    assert cm.sum() == 2
