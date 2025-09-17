"""
Statistical Tests Module for Feature Engineering Pipeline

This module now serves as a compatibility layer that imports the modular
statistical tests components. The main functionality has been refactored
into separate modules for better organization and maintainability.

For new code, prefer importing directly from the specific modules:
- ADF tests removed from project
- statistical_tests.distance_correlation for distance correlation analysis
- statistical_tests.feature_selection for feature selection operations
- statistical_tests.statistical_analysis for general statistical analysis
- statistical_tests.controller for the main orchestrating class
"""

# Import the main controller class for backward compatibility
from .statistical_tests.controller import StatisticalTests

# Re-export for backward compatibility
__all__ = ['StatisticalTests']