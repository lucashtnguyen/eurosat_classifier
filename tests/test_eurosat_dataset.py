import pytest
import torch
from torch.utils.data import DataLoader


@pytest.mark.slow
def test_dataset_loads(requires_eurosat_data) -> None:
    """Dataset can be indexed and returns tensors."""
    img, label = requires_eurosat_data[0]
    assert isinstance(label, int)
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 64, 64)
    assert img.dtype == torch.float32


@pytest.mark.slow
def test_subset_dataloader_shapes(eurosat_subset) -> None:
    """DataLoader from subset yields correctly shaped batches."""
    loader = DataLoader(eurosat_subset, batch_size=4, shuffle=False)
    images, labels = next(iter(loader))
    assert images.shape == (4, 3, 64, 64)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64
