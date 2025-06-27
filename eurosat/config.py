"""Configuration utilities for EuroSAT classifier."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

DEBUG = True
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(levelname)s: %(message)s",
)
log_buffer: list[str] = []
logger = logging.getLogger(__name__)


REQUIRED_SECTIONS = {"model", "training", "data", "output"}


def _validate(cfg: Dict[str, Any]) -> None:
    """Validate configuration dictionary.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Parsed configuration dictionary.
    """
    for key in REQUIRED_SECTIONS:
        if key not in cfg:
            raise KeyError(f"Missing required section: {key}")
        if not isinstance(cfg[key], dict):
            raise TypeError(f"Section '{key}' must be a mapping")


def load_config(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file.

    Parameters
    ----------
    path : Path
        Path to the YAML configuration file.

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary.

    """
    logger.debug("Loading config from %s", path)
    log_buffer.append(f"DEBUG: Loading config from {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    _validate(cfg)
    return cfg
