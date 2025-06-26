"""Model architecture definitions."""

from __future__ import annotations

import logging
from typing import Any

import torch.nn as nn

DEBUG = True
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(levelname)s: %(message)s",
)
log_buffer: list[str] = []
logger = logging.getLogger(__name__)


class BaselineCNN(nn.Module):
    """Simple convolutional neural network."""

    def __init__(self, hidden_units: int, dropout: float) -> None:
        super().__init__()
        logger.debug("Initializing BaselineCNN")
        log_buffer.append("DEBUG: Initializing BaselineCNN")
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, 10),
        )

    def forward(self, x: Any) -> Any:
        """Forward pass."""
        return self.classifier(self.features(x))


def build_model(config: dict) -> nn.Module:
    """Factory for models from config."""
    logger.debug("Building model with config: %s", config)
    log_buffer.append(f"DEBUG: Building model with {config}")
    name = config.get("name", "baseline_cnn")
    if name == "baseline_cnn":
        return BaselineCNN(config.get("hidden_units", 128), config.get("dropout", 0.3))
    # TODO: support more architectures
    raise ValueError(f"Unknown model name: {name}")
