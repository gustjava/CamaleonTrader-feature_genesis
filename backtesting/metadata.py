"""Metadata loading helpers for the backtesting module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .config import BacktestMetadata


def load_backtest_metadata(source: Any) -> BacktestMetadata:
    """Load metadata from a JSON file or mapping.

    Args:
        source: Path to a JSON file, path-like object, or already-loaded mapping.

    Returns:
        BacktestMetadata instance.
    """
    if isinstance(source, BacktestMetadata):
        return source

    if isinstance(source, Mapping):
        return BacktestMetadata.from_mapping(source)

    metadata_path = Path(source)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    return BacktestMetadata.from_mapping(payload)
