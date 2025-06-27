"""Data loading utilities for EuroSAT dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import random

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DEBUG = True
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(levelname)s: %(message)s",
)
log_buffer: list[str] = []
logger = logging.getLogger(__name__)


TRANSFORM = transforms.Compose([transforms.ToTensor()])


def get_dataloaders(
    root: Path, train_split: float, batch_size: int, seed: int
) -> Tuple[DataLoader, DataLoader]:
    """Return train and validation dataloaders.

    Parameters
    ----------
    root : Path
        Dataset root directory.
    train_split : float
        Fraction of data used for training.
    batch_size : int
        Batch size for loaders.
    seed : int
        Random seed for deterministic split.

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        Training and validation dataloaders.
    """
    logger.debug("Preparing dataloaders from %s", root)
    log_buffer.append(f"DEBUG: Preparing dataloaders from {root}")
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    dataset = datasets.ImageFolder(root=root, transform=TRANSFORM)
    indices = list(range(len(dataset)))
    if hasattr(dataset, "targets"):
        labels = list(dataset.targets)
    else:
        labels = [s[1] for s in dataset.samples]
    train_idx, val_idx = train_test_split(
        indices, train_size=train_split, stratify=labels, random_state=seed
    )
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )
