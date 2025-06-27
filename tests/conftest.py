"""Pytest configuration for import paths."""

from pathlib import Path
import logging
import random
import sys
from typing import Generator

import numpy as np
import pytest
import torch
from torchvision.datasets import EuroSAT
from torchvision import transforms
from torch.utils.data import Subset

sys.path.append(str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def requires_eurosat_data() -> EuroSAT:
    """Download EuroSAT dataset if needed and return the full dataset."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    root = Path(".data/eurosat")
    transform = transforms.ToTensor()
    try:
        ds = EuroSAT(root=str(root), transform=transform, download=True)
    except Exception as exc:  # pragma: no cover - download failure
        logger.warning("EuroSAT dataset unavailable: %s", exc)
        pytest.skip("EuroSAT dataset not available for tests")
    return ds


@pytest.fixture(scope="session")
def eurosat_root(requires_eurosat_data: EuroSAT) -> Path:
    """Return path to the directory used by ImageFolder."""
    return Path(requires_eurosat_data.root) / "eurosat" / "2750"


@pytest.fixture(scope="session")
def eurosat_subset(requires_eurosat_data: EuroSAT) -> Subset:
    """Return a deterministic small subset with ≤50 samples per class."""
    max_per_class = 50
    indices: list[int] = []
    counts = {cls: 0 for cls in range(len(requires_eurosat_data.classes))}
    for idx, label in enumerate(requires_eurosat_data.targets):
        if counts[label] < max_per_class:
            indices.append(idx)
            counts[label] += 1
        if all(v >= max_per_class for v in counts.values()):
            break
    return Subset(requires_eurosat_data, indices)
