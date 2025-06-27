"""Tests for train entrypoint."""

from pathlib import Path

import pytest

import torch

import train as train_module


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.zeros(len(x), 2)


def fake_build_model(cfg):
    return DummyModel()


def fake_dataloaders(root, split, batch_size, seed):
    data = [(torch.zeros(1, 3, 64, 64), torch.tensor(0))]
    loader = torch.utils.data.DataLoader(data, batch_size=1)
    return loader, loader


def fake_train_epoch(model, data, optim):
    return 0.0


def fake_evaluate(model, data):
    return 0.5, 0.5, torch.zeros((2, 2))


@pytest.fixture()
def config_file(tmp_path: Path) -> Path:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"""
model: {{}}
training:
  batch_size: 1
  learning_rate: 0.001
  epochs: 1
  seed: 0
data:
  root_dir: ./data
  train_split: 0.5
output:
  base_dir: {tmp_path}
"""
    )
    return cfg


def test_train_saves_artifacts(monkeypatch, config_file: Path) -> None:
    monkeypatch.setattr(train_module, "build_model", fake_build_model)
    monkeypatch.setattr(train_module, "get_dataloaders", fake_dataloaders)
    monkeypatch.setattr(train_module, "train_epoch", fake_train_epoch)
    monkeypatch.setattr(train_module, "evaluate", fake_evaluate)
    out = train_module.train(config_file)
    assert (out / "model.pt").exists()
    assert (out / "metrics.json").exists()
    assert (out / "confusion_matrix.png").exists()
    assert (out / "config.yaml").exists()
