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
    # TODO: validate schema
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
