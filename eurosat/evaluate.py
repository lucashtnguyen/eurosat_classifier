"""Evaluation utilities."""

from __future__ import annotations

import logging
from typing import Iterable, Tuple

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

DEBUG = True
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(levelname)s: %(message)s",
)
log_buffer: list[str] = []
logger = logging.getLogger(__name__)


def evaluate(
    model: torch.nn.Module, data: Iterable
) -> Tuple[float, float, torch.Tensor]:
    """Evaluate model accuracy and return confusion matrix.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    data : Iterable
        Validation dataloader.

    Returns
    -------
    Tuple[float, float, torch.Tensor]
        Accuracy, F1 score, and confusion matrix.
    """
    logger.debug("Evaluating model")
    log_buffer.append("DEBUG: Evaluating model")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data:
            outputs = model(batch[0])
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch[1].cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)
    return acc, f1, torch.tensor(cm)
