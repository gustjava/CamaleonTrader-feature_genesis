"""
ADF (Augmented Dickey-Fuller) Tests Module

This module contains functionality for performing ADF tests for stationarity testing
on time series data, including batch processing and rolling window implementations.
"""

import logging
import numpy as np
import cupy as cp
import cudf
from typing import List, Dict, Any, Tuple
from .utils import _adf_tstat_window_host, _adf_rolling_partition

logger = logging.getLogger(__name__)


class ADFTests:
    """Class for performing ADF (Augmented Dickey-Fuller) stationarity tests."""
    
    def __init__(self, logger_instance=None):
        """Initialize ADF tests with optional logger."""
        self.logger = logger_instance or logger
    
    def _log_info(self, message: str, **kwargs):
        """Log info message with optional context."""
        if self.logger:
            self.logger.info(f"ADF: {message}", extra=kwargs)
    
    def _log_warn(self, message: str, **kwargs):
        """Log warning message with optional context."""
        if self.logger:
            self.logger.warning(f"ADF: {message}", extra=kwargs)
    
    def _log_error(self, message: str, **kwargs):
        """Log error message with optional context."""
        if self.logger:
            self.logger.error(f"ADF: {message}", extra=kwargs)
    
    def _critical_error(self, message: str, **kwargs):
        """Log critical error and raise exception."""
        self._log_error(message, **kwargs)
        raise RuntimeError(f"ADF Critical Error: {message}")
    
    def _adf_tstat_window(self, vals):
        """Compute ADF t-statistic for a window (GPU version)."""
        try:
            if len(vals) < 3:
                return cp.nan
            
            # Convert to CuPy array
            vals_cp = cp.asarray(vals, dtype=cp.float64)
            
            # Remove NaN values
            valid_mask = ~cp.isnan(vals_cp)
            if cp.sum(valid_mask) < 3:
                return cp.nan
            
            vals_clean = vals_cp[valid_mask]
            n = len(vals_clean)
            
            # Calculate first differences
            diff_vals = cp.diff(vals_clean)
            lagged_vals = vals_clean[:-1]
            
            # Simple AR(1) regression: Δy_t = α + β*y_{t-1} + ε_t
            # We want to test H0: β = 0 (unit root)
            
            # Calculate regression coefficients
            n_diff = len(diff_vals)
            if n_diff < 2:
                return cp.nan
            
            # Mean center the data
            mean_lagged = cp.mean(lagged_vals)
            mean_diff = cp.mean(diff_vals)
            
            # Calculate sums for regression
            sum_xx = cp.sum((lagged_vals - mean_lagged) ** 2)
            sum_xy = cp.sum((lagged_vals - mean_lagged) * (diff_vals - mean_diff))
            
            if sum_xx <= 0:
                return cp.nan
            
            # Regression coefficient
            beta = sum_xy / sum_xx
            
            # Calculate residuals and standard error
            residuals = diff_vals - mean_diff - beta * (lagged_vals - mean_lagged)
            mse = cp.sum(residuals ** 2) / (n_diff - 2)
            
            # Standard error of beta
            se_beta = cp.sqrt(mse / sum_xx)
            
            if se_beta <= 0:
                return cp.nan
            
            # ADF t-statistic
            adf_stat = beta / se_beta
            
            return float(adf_stat)
            
        except Exception as e:
            self._log_error(f"Error in ADF t-statistic calculation: {e}")
            return cp.nan
    
    def _apply_adf_rolling(self, s: cudf.Series, window: int = 252, min_periods: int = 200) -> cudf.Series:
        """Apply ADF test to rolling windows of a time series."""
        try:
            self._log_info(f"Applying ADF rolling test", window=window, min_periods=min_periods)
            
            # Use the utility function for rolling ADF
            return _adf_rolling_partition(s, window, min_periods)
            
        except Exception as e:
            self._log_error(f"Error in rolling ADF: {e}")
            return cudf.Series(cp.full(len(s), cp.nan))
    
    def _compute_adf_vectorized(self, data: cp.ndarray, max_lag: int = None) -> Dict[str, float]:
        """
        Compute ADF test for a single time series using vectorized operations.
        
        Args:
            data: Time series data as CuPy array
            max_lag: Maximum lag for ADF test
            
        Returns:
            Dictionary with ADF test results
        """
        try:
            # Remove NaN values
            valid_mask = ~cp.isnan(data)
            if cp.sum(valid_mask) < 10:
                return {
                    'adf_stat': float('nan'),
                    'p_value': float('nan'),
                    'critical_values': {'1%': float('nan'), '5%': float('nan'), '10%': float('nan')},
                    'is_stationary': False
                }
            
            data_clean = data[valid_mask]
            n = len(data_clean)
            
            if max_lag is None:
                max_lag = int(12 * (n / 100) ** (1/4))  # Schwert criterion
            
            # Calculate first differences
            diff_data = cp.diff(data_clean)
            lagged_data = data_clean[:-1]
            
            # Create lagged differences matrix
            lagged_diffs = cp.zeros((len(diff_data), max_lag))
            for lag in range(1, max_lag + 1):
                lagged_diffs[:, lag-1] = cp.roll(diff_data, lag)
                lagged_diffs[:lag, lag-1] = 0  # Set invalid lags to 0
            
            # Create regression matrix: [lagged_series, lagged_diffs, trend, constant]
            trend = cp.arange(len(lagged_data), dtype=cp.float32)
            constant = cp.ones(len(lagged_data), dtype=cp.float32)
            
            # Stack all regressors
            X = cp.column_stack([lagged_data, lagged_diffs, trend, constant])
            y = diff_data
            
            # Remove rows with NaN values
            valid_rows = ~cp.any(cp.isnan(X), axis=1) & ~cp.isnan(y)
            if cp.sum(valid_rows) < max_lag + 5:
                return {
                    'adf_stat': float('nan'),
                    'p_value': float('nan'),
                    'critical_values': {'1%': float('nan'), '5%': float('nan'), '10%': float('nan')},
                    'is_stationary': False
                }
            
            X_clean = X[valid_rows]
            y_clean = y[valid_rows]
            
            # Solve least squares using CuPy
            try:
                # Use QR decomposition for numerical stability
                Q, R = cp.linalg.qr(X_clean)
                beta = cp.linalg.solve(R, Q.T @ y_clean)
                
                # Calculate residuals and standard error
                residuals = y_clean - X_clean @ beta
                mse = cp.sum(residuals**2) / (len(y_clean) - len(beta))
                
                # Standard error of the coefficient on lagged_series (first coefficient)
                X_inv = cp.linalg.inv(X_clean.T @ X_clean)
                se_beta = cp.sqrt(mse * X_inv[0, 0])
                
                # ADF statistic
                adf_stat = beta[0] / se_beta
                
                # Calculate p-value
                p_value = self._calculate_adf_pvalue_vectorized(adf_stat, len(y_clean))
                
                # Critical values (approximate)
                critical_values = {
                    '1%': -3.43,
                    '5%': -2.86,
                    '10%': -2.57
                }
                
                # Determine stationarity
                is_stationary = adf_stat < critical_values['5%']
                
                return {
                    'adf_stat': float(adf_stat),
                    'p_value': float(p_value),
                    'critical_values': critical_values,
                    'is_stationary': bool(is_stationary)
                }
                
            except Exception as e:
                self._log_error(f"Error in ADF regression: {e}")
                return {
                    'adf_stat': float('nan'),
                    'p_value': float('nan'),
                    'critical_values': {'1%': float('nan'), '5%': float('nan'), '10%': float('nan')},
                    'is_stationary': False
                }
                
        except Exception as e:
            self._log_error(f"Error in vectorized ADF computation: {e}")
            return {
                'adf_stat': float('nan'),
                'p_value': float('nan'),
                'critical_values': {'1%': float('nan'), '5%': float('nan'), '10%': float('nan')},
                'is_stationary': False
            }
    
    def _create_adf_regression_matrix_vectorized(self, data: cp.ndarray, lag: int) -> Tuple[cp.ndarray, cp.ndarray]:
        """Create regression matrix for ADF test in vectorized form."""
        try:
            n = len(data)
            if n < lag + 2:
                return cp.array([]), cp.array([])
            
            # Create lagged differences
            diff_data = cp.diff(data)
            lagged_diffs = cp.zeros((len(diff_data), lag))
            
            for i in range(lag):
                lagged_diffs[:, i] = cp.roll(diff_data, i + 1)
                lagged_diffs[:i+1, i] = 0  # Set invalid lags to 0
            
            # Create dependent variable (first differences)
            y = diff_data[lag:]
            
            # Create independent variables: [lagged_series, lagged_diffs, trend, constant]
            lagged_series = data[lag:-1]  # y_{t-1}
            trend = cp.arange(len(lagged_series), dtype=cp.float32)
            constant = cp.ones(len(lagged_series), dtype=cp.float32)
            
            # Stack all regressors
            X = cp.column_stack([lagged_series, lagged_diffs[lag:], trend, constant])
            
            return X, y
            
        except Exception as e:
            self._log_error(f"Error creating ADF regression matrix: {e}")
            return cp.array([]), cp.array([])
    
    def _calculate_adf_pvalue_vectorized(self, adf_stat: float, n_obs: int) -> float:
        """Calculate approximate p-value for ADF statistic."""
        try:
            # This is a simplified approach - for production, use proper ADF p-value tables
            if adf_stat < -3.43:
                return 0.01
            elif adf_stat < -2.86:
                return 0.05
            elif adf_stat < -2.57:
                return 0.10
            else:
                return 1.0
                
        except Exception:
            return float('nan')
    
    def _compute_adf_batch_vectorized(self, data_matrix: cp.ndarray, max_lag: int = None) -> Dict[str, cp.ndarray]:
        """
        Batched ADF test implementation using vectorized operations.
        Processes multiple time series simultaneously for better performance.
        
        Args:
            data_matrix: 3D array of shape (n_series, n_observations, 1)
            max_lag: Maximum lag for ADF test
            
        Returns:
            Dictionary with ADF statistics for all series
        """
        try:
            n_series, n_obs, _ = data_matrix.shape
            
            if max_lag is None:
                max_lag = int(12 * (n_obs / 100) ** (1/4))  # Schwert criterion
            
            # Initialize results arrays
            adf_stats = cp.zeros(n_series, dtype=cp.float32)
            p_values = cp.zeros(n_series, dtype=cp.float32)
            critical_values = cp.zeros((n_series, 3), dtype=cp.float32)  # 1%, 5%, 10%
            
            # Process each series
            for i in range(n_series):
                series = data_matrix[i, :, 0]
                
                # Remove NaN values
                valid_mask = ~cp.isnan(series)
                if cp.sum(valid_mask) < max_lag + 10:
                    adf_stats[i] = cp.nan
                    p_values[i] = cp.nan
                    critical_values[i, :] = cp.nan
                    continue
                
                series_clean = series[valid_mask]
                
                # Calculate differences
                diff_series = cp.diff(series_clean)
                
                # Create lagged differences matrix
                lagged_diffs = cp.zeros((len(diff_series), max_lag))
                for lag in range(1, max_lag + 1):
                    lagged_diffs[:, lag-1] = cp.roll(diff_series, lag)
                    lagged_diffs[:lag, lag-1] = 0  # Set invalid lags to 0
                
                # Create regression matrix: [lagged_series, lagged_diffs, trend, constant]
                lagged_series = series_clean[:-1]  # y_{t-1}
                trend = cp.arange(len(lagged_series), dtype=cp.float32)
                constant = cp.ones(len(lagged_series), dtype=cp.float32)
                
                # Stack all regressors
                X = cp.column_stack([lagged_series, lagged_diffs, trend, constant])
                y = diff_series
                
                # Remove rows with NaN values
                valid_rows = ~cp.any(cp.isnan(X), axis=1) & ~cp.isnan(y)
                if cp.sum(valid_rows) < max_lag + 5:
                    adf_stats[i] = cp.nan
                    p_values[i] = cp.nan
                    critical_values[i, :] = cp.nan
                    continue
                
                X_clean = X[valid_rows]
                y_clean = y[valid_rows]
                
                # Solve least squares using CuPy
                try:
                    # Use QR decomposition for numerical stability
                    Q, R = cp.linalg.qr(X_clean)
                    beta = cp.linalg.solve(R, Q.T @ y_clean)
                    
                    # Calculate residuals and standard error
                    residuals = y_clean - X_clean @ beta
                    mse = cp.sum(residuals**2) / (len(y_clean) - len(beta))
                    
                    # Standard error of the coefficient on lagged_series (first coefficient)
                    X_inv = cp.linalg.inv(X_clean.T @ X_clean)
                    se_beta = cp.sqrt(mse * X_inv[0, 0])
                    
                    # ADF statistic
                    adf_stat = beta[0] / se_beta
                    adf_stats[i] = float(adf_stat)
                    
                    # Approximate p-value using critical values
                    p_value = self._calculate_adf_pvalue_vectorized(adf_stat, len(y_clean))
                    p_values[i] = float(p_value)
                    
                    # Critical values (approximate)
                    critical_values[i, 0] = -3.43  # 1%
                    critical_values[i, 1] = -2.86  # 5%
                    critical_values[i, 2] = -2.57  # 10%
                    
                except Exception as e:
                    self._log_error(f"Error in ADF regression for series {i}: {e}")
                    adf_stats[i] = cp.nan
                    p_values[i] = cp.nan
                    critical_values[i, :] = cp.nan
            
            return {
                'adf_stat': adf_stats,
                'p_value': p_values,
                'critical_values': critical_values
            }
            
        except Exception as e:
            self._log_error(f"Error in batch ADF computation: {e}")
            return {
                'adf_stat': cp.array([]),
                'p_value': cp.array([]),
                'critical_values': cp.array([])
            }
    
    def _apply_comprehensive_adf_tests(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """
        Apply comprehensive ADF tests to all frac_diff_* series.
        
        Generates:
        - adf_stat_* for each series
        - adf_pvalue_* for each series
        - adf_lag_order_* for each series
        - is_stationary_adf_* flag for each series
        """
        try:
            self._log_info("Applying comprehensive ADF tests...")
            
            # Find all frac_diff columns
            frac_diff_cols = [col for col in df.columns if "frac_diff" in col]
            self._log_info(f"Found {len(frac_diff_cols)} frac_diff columns for ADF testing")
            
            if not frac_diff_cols:
                self._log_warn("No frac_diff columns found for ADF testing")
                return df
            
            # Prepare data for batch ADF testing
            data_series = []
            series_names = []
            
            for col in frac_diff_cols:
                if col in df.columns:
                    data = df[col].to_cupy()
                    # Remove NaN values
                    valid_mask = ~cp.isnan(data)
                    if cp.sum(valid_mask) > 50:  # Minimum sample size
                        clean_data = data[valid_mask]
                        data_series.append(clean_data)
                        series_names.append(col)
            
            if not data_series:
                self._log_warn("No valid data series found for ADF testing")
                return df
            
            # Run batch ADF tests
            self._log_info(f"Running batch ADF tests for {len(data_series)} series")
            adf_results = self.compute_adf_batch(data_series, max_lag=12)
            
            # Add results to DataFrame
            for i, col in enumerate(series_names):
                if i < len(adf_results['adf_stat']):
                    # Extract base name (remove 'frac_diff_' prefix)
                    base_name = col.replace('frac_diff_', '')
                    
                    # Add ADF statistics
                    df[f'adf_stat_{base_name}'] = float(adf_results['adf_stat'][i])
                    df[f'adf_pvalue_{base_name}'] = float(adf_results['p_value'][i])
                    
                    # Add stationarity flag
                    is_stationary = adf_results['p_value'][i] < 0.05
                    df[f'is_stationary_adf_{base_name}'] = bool(is_stationary)
                    
                    # Add lag order (simplified)
                    df[f'adf_lag_order_{base_name}'] = 12
            
            self._log_info("Comprehensive ADF tests completed")
            return df
            
        except Exception as e:
            self._log_error(f"Error in comprehensive ADF tests: {e}")
            return df
    
    def compute_adf_batch(self, data_series: List[cp.ndarray], max_lag: int = None) -> Dict[str, np.ndarray]:
        """
        Compute ADF test for multiple time series in batch.
        
        Args:
            data_series: List of time series arrays
            max_lag: Maximum lag for ADF test
            
        Returns:
            Dictionary with ADF results for all series
        """
        try:
            if not data_series:
                return {'adf_stat': [], 'p_value': [], 'critical_values': []}
            
            # Find maximum length for padding
            max_length = max(len(series) for series in data_series)
            n_series = len(data_series)
            
            # Create 3D matrix with padding
            data_matrix = cp.full((n_series, max_length, 1), cp.nan, dtype=cp.float32)
            
            for i, series in enumerate(data_series):
                if len(series) > 0:
                    data_matrix[i, :len(series), 0] = series
            
            # Compute ADF for all series
            results = self._compute_adf_batch_gpu(data_matrix, max_lag)
            
            return results
            
        except Exception as e:
            self._log_error(f"Error in batch ADF computation: {e}")
            return {'adf_stat': [], 'p_value': [], 'critical_values': []}
    
    def _compute_adf_batch_gpu(self, data_matrix: cp.ndarray, max_lag: int = None) -> Dict[str, cp.ndarray]:
        """
        Batched ADF test implementation using GPU operations.
        Processes multiple time series simultaneously for better performance.
        
        Args:
            data_matrix: 3D array of shape (n_series, n_observations, 1)
            max_lag: Maximum lag for ADF test
            
        Returns:
            Dictionary with ADF statistics for all series
        """
        try:
            n_series, n_obs, _ = data_matrix.shape
            
            if max_lag is None:
                max_lag = int(12 * (n_obs / 100) ** (1/4))  # Schwert criterion
            
            # Initialize results arrays
            adf_stats = cp.zeros(n_series, dtype=cp.float32)
            p_values = cp.zeros(n_series, dtype=cp.float32)
            critical_values = cp.zeros((n_series, 3), dtype=cp.float32)  # 1%, 5%, 10%
            
            # Process each series
            for i in range(n_series):
                series = data_matrix[i, :, 0]
                
                # Remove NaN values
                valid_mask = ~cp.isnan(series)
                if cp.sum(valid_mask) < max_lag + 10:
                    adf_stats[i] = cp.nan
                    p_values[i] = cp.nan
                    critical_values[i, :] = cp.nan
                    continue
                
                series_clean = series[valid_mask]
                
                # Calculate differences
                diff_series = cp.diff(series_clean)
                
                # Create lagged differences matrix
                lagged_diffs = cp.zeros((len(diff_series), max_lag))
                for lag in range(1, max_lag + 1):
                    lagged_diffs[:, lag-1] = cp.roll(diff_series, lag)
                    lagged_diffs[:lag, lag-1] = 0  # Set invalid lags to 0
                
                # Create regression matrix: [lagged_series, lagged_diffs, trend, constant]
                lagged_series = series_clean[:-1]  # y_{t-1}
                trend = cp.arange(len(lagged_series), dtype=cp.float32)
                constant = cp.ones(len(lagged_series), dtype=cp.float32)
                
                # Stack all regressors
                X = cp.column_stack([lagged_series, lagged_diffs, trend, constant])
                y = diff_series
                
                # Remove rows with NaN values
                valid_rows = ~cp.any(cp.isnan(X), axis=1) & ~cp.isnan(y)
                if cp.sum(valid_rows) < max_lag + 5:
                    adf_stats[i] = cp.nan
                    p_values[i] = cp.nan
                    critical_values[i, :] = cp.nan
                    continue
                
                X_clean = X[valid_rows]
                y_clean = y[valid_rows]
                
                # Solve least squares using CuPy
                try:
                    # Use QR decomposition for numerical stability
                    Q, R = cp.linalg.qr(X_clean)
                    beta = cp.linalg.solve(R, Q.T @ y_clean)
                    
                    # Calculate residuals and standard error
                    residuals = y_clean - X_clean @ beta
                    mse = cp.sum(residuals**2) / (len(y_clean) - len(beta))
                    
                    # Standard error of the coefficient on lagged_series (first coefficient)
                    X_inv = cp.linalg.inv(X_clean.T @ X_clean)
                    se_beta = cp.sqrt(mse * X_inv[0, 0])
                    
                    # ADF statistic
                    adf_stat = beta[0] / se_beta
                    adf_stats[i] = float(adf_stat)
                    
                    # Approximate p-value using critical values
                    p_value = self._calculate_adf_pvalue_vectorized(adf_stat, len(y_clean))
                    p_values[i] = float(p_value)
                    
                    # Critical values (approximate)
                    critical_values[i, 0] = -3.43  # 1%
                    critical_values[i, 1] = -2.86  # 5%
                    critical_values[i, 2] = -2.57  # 10%
                    
                except Exception as e:
                    self._log_error(f"Error in ADF regression for series {i}: {e}")
                    adf_stats[i] = cp.nan
                    p_values[i] = cp.nan
                    critical_values[i, :] = cp.nan
            
            return {
                'adf_stat': adf_stats,
                'p_value': p_values,
                'critical_values': critical_values
            }
            
        except Exception as e:
            self._log_error(f"Error in batch ADF computation: {e}")
            return {
                'adf_stat': cp.array([]),
                'p_value': cp.array([]),
                'critical_values': cp.array([])
            }
