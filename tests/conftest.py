"""Pytest configuration for import paths."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
