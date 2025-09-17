"""
Statistical Tests Module Package

This package contains modular components for statistical testing and feature selection.
"""

from .controller import StatisticalTests
# ADF tests removed from project
from .distance_correlation import DistanceCorrelation
from .feature_selection import FeatureSelection
from .statistical_analysis import StatisticalAnalysis
from .utils import (
    _free_gpu_memory_worker,
    _adaptive_tile,
    _hermitian_pinv_gpu,
    _tail_k,
    _tail_k_to_pandas
)

__all__ = [
    'StatisticalTests',
    'DistanceCorrelation',
    'FeatureSelection',
    'StatisticalAnalysis',
    '_free_gpu_memory_worker',
    '_adaptive_tile',
    '_hermitian_pinv_gpu',
    '_tail_k',
    '_tail_k_to_pandas'
]
