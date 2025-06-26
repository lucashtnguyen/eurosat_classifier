"""Top-level training entrypoint."""

from __future__ import annotations

import logging
from pathlib import Path

from eurosat.config import load_config
from eurosat.data_loader import get_dataloaders
from eurosat.model import build_model
from eurosat.train_loop import train_epoch
from eurosat.evaluate import evaluate

DEBUG = True
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(levelname)s: %(message)s",
)
log_buffer: list[str] = []
logger = logging.getLogger(__name__)


def train(config_path: Path) -> Path:
    """Run full training pipeline.

    Parameters
    ----------
    config_path : Path
        Path to YAML configuration.

    Returns
    -------
    Path
        Output directory with run artifacts.
    """
    logger.debug("Starting training with config at %s", config_path)
    log_buffer.append(f"DEBUG: Starting training with config at {config_path}")
    config = load_config(config_path)
    root = Path(config["data"]["root_dir"])
    loaders = get_dataloaders(root, config["data"]["train_split"], config["training"]["batch_size"])
    model = build_model(config["model"])
    optim = __import__("torch").optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    for _ in range(config["training"]["epochs"]):
        train_epoch(model, loaders[0], optim)
    acc, cm = evaluate(model, loaders[1])
    logger.info("Accuracy: %.4f", acc)
    log_buffer.append(f"INFO: Accuracy: {acc:.4f}")
    # TODO: save artifacts to timestamped directory
    return Path(config["output"]["base_dir"]).resolve()
