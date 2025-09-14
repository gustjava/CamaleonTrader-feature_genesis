"""
Features module for Dynamic Stage 0 pipeline.

This module contains all GPU-accelerated feature engineering implementations,
including stationarization techniques, statistical tests, and signal processing.
"""

from .base_engine import BaseFeatureEngine
from .stationarization import StationarizationEngine
from .feature_engineering import FeatureEngineeringEngine
from .statistical_tests import StatisticalTests
from .garch_models import GARCHModels
# Import selection utilities lazily to avoid heavy optional deps at import time
try:
    from .selection import create_target_variable, select_features_with_catboost
except Exception:
    # Provide placeholders that raise informative errors when called
    def create_target_variable(*args, **kwargs):
        raise RuntimeError("features.selection.create_target_variable is unavailable due to missing optional dependencies.")
    def select_features_with_catboost(*args, **kwargs):
        raise RuntimeError("features.selection.select_features_with_catboost is unavailable due to missing optional dependencies (catboost). Install 'catboost' to use it.")
from .signal_processing import apply_emd_to_series

__all__ = [
    'BaseFeatureEngine',
    'StationarizationEngine',
    'FeatureEngineeringEngine',
    'StatisticalTests', 
    'GARCHModels',
    'create_target_variable',
    'select_features_with_catboost',
    'apply_emd_to_series',
]
