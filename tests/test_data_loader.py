"""Tests for data loading."""

from pathlib import Path

import pytest

from eurosat.data_loader import get_dataloaders


def test_get_dataloaders_invalid_path(tmp_path: Path) -> None:
    """Edge case: dataset path does not exist."""
    with pytest.raises(Exception):
        get_dataloaders(tmp_path / "missing", 0.8, 1, 0)


def test_get_dataloaders_placeholder(monkeypatch, tmp_path: Path) -> None:
    """Happy path placeholder using fake dataset."""
    from torch.utils.data import Dataset, DataLoader

    class DummySet(Dataset):
        def __init__(self) -> None:
            self.targets = [0, 0, 1, 1]

        def __len__(self) -> int:
            return 4

        def __getitem__(self, idx: int):
            return (0, self.targets[idx])

    def fake_imagefolder(root: Path, transform=None):
        return DummySet()

    monkeypatch.setattr("eurosat.data_loader.datasets.ImageFolder", fake_imagefolder)
    train_dl, val_dl = get_dataloaders(tmp_path, 0.5, 1, 0)
    assert isinstance(train_dl, DataLoader)
    assert isinstance(val_dl, DataLoader)


def test_get_dataloaders_stratified(monkeypatch, tmp_path: Path) -> None:
    """Stratified split is deterministic and balanced."""
    import torch
    from torch.utils.data import Dataset

    class Dummy(Dataset):
        def __init__(self) -> None:
            self.targets = [0, 0, 1, 1]

        def __len__(self) -> int:
            return 4

        def __getitem__(self, idx: int):
            return torch.zeros(1), self.targets[idx]

    def fake_imagefolder(root: Path, transform=None):
        return Dummy()

    monkeypatch.setattr("eurosat.data_loader.datasets.ImageFolder", fake_imagefolder)
    dl1_a, dl2_a = get_dataloaders(tmp_path, 0.5, 1, 42)
    dl1_b, dl2_b = get_dataloaders(tmp_path, 0.5, 1, 42)
    assert dl1_a.dataset.indices == dl1_b.dataset.indices
    labels1 = [dl1_a.dataset.dataset.targets[i] for i in dl1_a.dataset.indices]
    labels2 = [dl2_a.dataset.dataset.targets[i] for i in dl2_a.dataset.indices]
    assert sorted(labels1) == [0, 1]
    assert sorted(labels2) == [0, 1]
