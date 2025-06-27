"""Tests for data loading."""

from pathlib import Path

import pytest

from eurosat.data_loader import get_dataloaders


def test_get_dataloaders_invalid_path(tmp_path: Path) -> None:
    """Edge case: dataset path does not exist."""
    with pytest.raises(Exception):
        get_dataloaders(tmp_path / "missing", 0.8, 1, 0)


@pytest.mark.slow
def test_get_dataloaders_real_dataset(eurosat_root: Path) -> None:
    """Happy path: dataloaders created from real EuroSAT data."""
    train_dl, val_dl = get_dataloaders(eurosat_root, 0.1, 2, 0)
    assert len(train_dl.dataset) > 0
    assert len(val_dl.dataset) > 0


@pytest.mark.slow
def test_get_dataloaders_stratified(eurosat_root: Path) -> None:
    """Stratified split is deterministic and balanced."""
    dl1_a, dl2_a = get_dataloaders(eurosat_root, 0.5, 2, 42)
    dl1_b, dl2_b = get_dataloaders(eurosat_root, 0.5, 2, 42)
    assert dl1_a.dataset.indices == dl1_b.dataset.indices
    train_labels = [dl1_a.dataset.dataset.targets[i] for i in dl1_a.dataset.indices]
    val_labels = [dl2_a.dataset.dataset.targets[i] for i in dl2_a.dataset.indices]
    from collections import Counter

    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)
    assert len(set(train_counts.values())) == 1
    assert len(set(val_counts.values())) == 1
