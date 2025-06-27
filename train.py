"""Top-level training entrypoint."""

from __future__ import annotations

import logging
import datetime
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
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
    seed = config["training"].get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    root = Path(config["data"]["root_dir"])
    loaders = get_dataloaders(
        root, config["data"]["train_split"], config["training"]["batch_size"], seed
    )
    model = build_model(config["model"])
    optim = __import__("torch").optim.Adam(
        model.parameters(), lr=config["training"]["learning_rate"]
    )
    for _ in range(config["training"]["epochs"]):
        train_epoch(model, loaders[0], optim)
    acc, f1, cm = evaluate(model, loaders[1])
    logger.info("Accuracy: %.4f F1: %.4f", acc, f1)
    log_buffer.append(f"INFO: Accuracy: {acc:.4f} F1: {f1:.4f}")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(config["output"]["base_dir"]) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump({"accuracy": acc, "f1": f1}, fh)
    plt.figure()
    plt.imshow(cm)
    plt.savefig(out_dir / "confusion_matrix.png")
    plt.close()
    with (out_dir / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh)
    return out_dir.resolve()
