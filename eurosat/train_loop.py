"""Training loop implementation."""

from __future__ import annotations

import logging
from typing import Iterable

import torch

DEBUG = True
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(levelname)s: %(message)s",
)
log_buffer: list[str] = []
logger = logging.getLogger(__name__)


def train_epoch(
    model: torch.nn.Module, data: Iterable, optimizer: torch.optim.Optimizer
) -> float:
    """Run a single training epoch.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    data : Iterable
        Training data loader.
    optimizer : torch.optim.Optimizer
        Optimizer instance.

    Returns
    -------
    float
        Mean loss for the epoch.
    """
    logger.debug("Starting training epoch")
    log_buffer.append("DEBUG: Starting training epoch")
    # TODO: implement proper training loop
    model.train()
    total_loss = 0.0
    for batch in data:
        optimizer.zero_grad()
        outputs = model(batch[0])
        loss = torch.nn.functional.cross_entropy(outputs, batch[1])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data)
