"""Backtesting package for Feature Genesis models.

This package exposes utilities to replay the feature pipeline over backtest
parquet files, score CatBoost models, and stream trading signals to a front-end
via WebSocket.
"""

from .config import BacktestRunConfig, SignalThresholds
from .metadata import load_backtest_metadata
from .model_runner import BacktestRunner
from .service import create_app

__all__ = [
    "BacktestRunConfig",
    "SignalThresholds",
    "load_backtest_metadata",
    "BacktestRunner",
    "create_app",
]
