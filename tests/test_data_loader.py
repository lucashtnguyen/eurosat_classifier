"""Tests for data loading."""

from pathlib import Path

import pytest

from eurosat.data_loader import get_dataloaders


def test_get_dataloaders_invalid_path(tmp_path: Path) -> None:
    """Edge case: dataset path does not exist."""
    with pytest.raises(Exception):
        get_dataloaders(tmp_path / "missing", 0.8, 1)


def test_get_dataloaders_placeholder(monkeypatch, tmp_path: Path) -> None:
    """Happy path placeholder using fake dataset."""
    from torch.utils.data import Dataset, DataLoader

    class DummySet(Dataset):
        def __len__(self) -> int:
            return 2

        def __getitem__(self, idx: int):
            return (0, 0)

    def fake_imagefolder(root: Path, transform=None):
        return DummySet()

    monkeypatch.setattr("eurosat.data_loader.datasets.ImageFolder", fake_imagefolder)
    train_dl, val_dl = get_dataloaders(tmp_path, 0.5, 1)
    assert isinstance(train_dl, DataLoader)
    assert isinstance(val_dl, DataLoader)
