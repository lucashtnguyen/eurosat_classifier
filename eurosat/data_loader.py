"""Data loading utilities for EuroSAT dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

DEBUG = True
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(levelname)s: %(message)s",
)
log_buffer: list[str] = []
logger = logging.getLogger(__name__)


TRANSFORM = transforms.Compose([transforms.ToTensor()])


def get_dataloaders(root: Path, train_split: float, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Return train and validation dataloaders.

    Parameters
    ----------
    root : Path
        Dataset root directory.
    train_split : float
        Fraction of data used for training.
    batch_size : int
        Batch size for loaders.

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        Training and validation dataloaders.
    """
    logger.debug("Preparing dataloaders from %s", root)
    log_buffer.append(f"DEBUG: Preparing dataloaders from {root}")
    dataset = datasets.ImageFolder(root=root, transform=TRANSFORM)
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )
