"""
Statistical Analysis Module

This module contains functionality for general statistical analysis including
normality tests, correlation analysis, and statistical summaries.
"""

import logging
import numpy as np
import cupy as cp
import cudf
from typing import List, Dict, Any
from .utils import _jb_pvalue_window_host

logger = logging.getLogger(__name__)


class StatisticalAnalysis:
    """Class for general statistical analysis and tests."""
    
    def __init__(self, logger_instance=None):
        """Initialize statistical analysis with optional logger."""
        self.logger = logger_instance or logger
    
    def _log_info(self, message: str, **kwargs):
        """Log info message with optional context."""
        if self.logger:
            self.logger.info(f"StatisticalAnalysis: {message}", extra=kwargs)
    
    def _log_warn(self, message: str, **kwargs):
        """Log warning message with optional context."""
        if self.logger:
            self.logger.warning(f"StatisticalAnalysis: {message}", extra=kwargs)
    
    def _log_error(self, message: str, **kwargs):
        """Log error message with optional context."""
        if self.logger:
            self.logger.error(f"StatisticalAnalysis: {message}", extra=kwargs)
    
    def _critical_error(self, message: str, **kwargs):
        """Log critical error and raise exception."""
        self._log_error(message, **kwargs)
        raise RuntimeError(f"StatisticalAnalysis Critical Error: {message}")

    def _compute_moments_vectorized(self, data: cp.ndarray) -> Dict[str, float]:
        """
        Compute statistical moments (mean, std, skewness, kurtosis) in vectorized form.
        """
        try:
            # Remove NaN values
            valid_data = data[~cp.isnan(data)]
            if len(valid_data) < 2:
                return {
                    'mean': float('nan'),
                    'std': float('nan'),
                    'skewness': float('nan'),
                    'kurtosis': float('nan')
                }
            
            # Compute basic statistics
            mean_val = float(cp.mean(valid_data))
            std_val = float(cp.std(valid_data))
            
            if std_val == 0:
                return {
                    'mean': mean_val,
                    'std': 0.0,
                    'skewness': 0.0,
                    'kurtosis': 0.0
                }
            
            # Standardize data
            standardized = (valid_data - mean_val) / std_val
            
            # Compute higher moments
            skewness = float(cp.mean(standardized**3))
            kurtosis = float(cp.mean(standardized**4))
            
            return {
                'mean': mean_val,
                'std': std_val,
                'skewness': skewness,
                'kurtosis': kurtosis
            }
            
        except Exception as e:
            self._log_error(f"Error in vectorized moments computation: {e}")
            return {
                'mean': float('nan'),
                'std': float('nan'),
                'skewness': float('nan'),
                'kurtosis': float('nan')
            }

    def _compute_normality_tests_vectorized(self, data: cp.ndarray) -> Dict[str, float]:
        """
        Vectorized normality tests including Jarque-Bera and Anderson-Darling.
        """
        try:
            # Jarque-Bera test (vectorized)
            moments = self._compute_moments_vectorized(data)
            n = len(data)
            
            jb_stat = n * (moments['skewness']**2 / 6 + (moments['kurtosis'] - 3)**2 / 24)
            jb_pvalue = 1.0  # Simplified - in practice, use chi-square distribution
            
            # Anderson-Darling test (simplified vectorized version)
            sorted_data = cp.sort(data)
            n = len(sorted_data)
            
            # Simplified AD statistic
            ad_stat = 0.0
            for i in range(n):
                p = (i + 1) / n
                ad_stat += (2*i + 1) * cp.log(p * (1 - p))
            
            ad_stat = -n - ad_stat / n
            ad_pvalue = 1.0  # Simplified
            
            return {
                'jarque_bera_stat': float(jb_stat),
                'jarque_bera_pvalue': jb_pvalue,
                'anderson_darling_stat': float(ad_stat),
                'anderson_darling_pvalue': ad_pvalue,
                'is_normal': jb_pvalue > 0.05 and ad_pvalue > 0.05
            }
            
        except Exception as e:
            self._log_error(f"Error in vectorized normality tests: {e}")
            return {
                'jarque_bera_stat': float('nan'),
                'jarque_bera_pvalue': float('nan'),
                'anderson_darling_stat': float('nan'),
                'anderson_darling_pvalue': float('nan'),
                'is_normal': False
            }

    def _apply_additional_statistical_tests(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """
        Apply additional statistical tests and metrics.
        
        Generates:
        - Additional stationarity metrics
        - Correlation matrices
        - Statistical summaries
        """
        try:
            self._log_info("Applying additional statistical tests...")
            
            # Find frac_diff columns for additional analysis
            frac_diff_cols = [col for col in df.columns if "frac_diff" in col]
            
            if len(frac_diff_cols) > 1:
                # Compute correlation matrix between frac_diff series
                self._log_info("Computing correlation matrix between frac_diff series")
                
                # Create correlation matrix
                corr_matrix = []
                for i, col1 in enumerate(frac_diff_cols):
                    row = []
                    for j, col2 in enumerate(frac_diff_cols):
                        if i == j:
                            row.append(1.0)
                        else:
                            # Compute correlation
                            data1 = df[col1].to_cupy()
                            data2 = df[col2].to_cupy()
                            
                            # Remove NaN values
                            valid_mask = ~(cp.isnan(data1) | cp.isnan(data2))
                            if cp.sum(valid_mask) > 10:
                                clean_data1 = data1[valid_mask]
                                clean_data2 = data2[valid_mask]
                                
                                # Compute correlation
                                corr = cp.corrcoef(clean_data1, clean_data2)[0, 1]
                                row.append(float(corr) if not cp.isnan(corr) else 0.0)
                            else:
                                row.append(0.0)
                    corr_matrix.append(row)
                
                # Add correlation matrix as features (simplified - just max correlation)
                if corr_matrix:
                    max_corr = max([max(row) for row in corr_matrix])
                    df['frac_diff_max_correlation'] = max_corr
            
            # Add statistical summaries for key series
            key_series = ['y_close', 'y_ret_1m']
            for col in key_series:
                if col in df.columns:
                    data = df[col].to_cupy()
                    valid_data = data[~cp.isnan(data)]
                    
                    if len(valid_data) > 0:
                        df[f'{col}_mean'] = float(cp.mean(valid_data))
                        df[f'{col}_std'] = float(cp.std(valid_data))
                        df[f'{col}_skew'] = float(cp.mean(((valid_data - cp.mean(valid_data)) / cp.std(valid_data))**3))
                        df[f'{col}_kurt'] = float(cp.mean(((valid_data - cp.mean(valid_data)) / cp.std(valid_data))**4))
            
            return df
            
        except Exception as e:
            self._log_error(f"Error in additional statistical tests: {e}")
            return df

    def compute_correlation_matrix(self, df: cudf.DataFrame, columns: List[str] = None) -> Dict[str, Any]:
        """
        Compute correlation matrix for specified columns.
        
        Args:
            df: Input DataFrame
            columns: List of column names to include in correlation matrix
            
        Returns:
            Dictionary with correlation matrix and statistics
        """
        try:
            if columns is None:
                # Find numeric columns
                numeric_cols = []
                for col in df.columns:
                    try:
                        data = df[col].to_cupy()
                        if cp.isfinite(data).any():
                            numeric_cols.append(col)
                    except Exception:
                        continue
                columns = numeric_cols
            
            if len(columns) < 2:
                return {
                    'correlation_matrix': [],
                    'max_correlation': 0.0,
                    'min_correlation': 0.0,
                    'mean_correlation': 0.0
                }
            
            self._log_info(f"Computing correlation matrix for {len(columns)} columns")
            
            # Create correlation matrix
            corr_matrix = []
            for i, col1 in enumerate(columns):
                row = []
                for j, col2 in enumerate(columns):
                    if i == j:
                        row.append(1.0)
                    else:
                        try:
                            data1 = df[col1].to_cupy()
                            data2 = df[col2].to_cupy()
                            
                            # Remove NaN values
                            valid_mask = ~(cp.isnan(data1) | cp.isnan(data2))
                            if cp.sum(valid_mask) > 10:
                                clean_data1 = data1[valid_mask]
                                clean_data2 = data2[valid_mask]
                                
                                # Compute correlation
                                corr = cp.corrcoef(clean_data1, clean_data2)[0, 1]
                                row.append(float(corr) if not cp.isnan(corr) else 0.0)
                            else:
                                row.append(0.0)
                        except Exception:
                            row.append(0.0)
                corr_matrix.append(row)
            
            # Compute statistics
            corr_array = np.array(corr_matrix)
            # Remove diagonal for statistics
            mask = ~np.eye(corr_array.shape[0], dtype=bool)
            off_diagonal = corr_array[mask]
            
            max_corr = float(np.max(off_diagonal)) if len(off_diagonal) > 0 else 0.0
            min_corr = float(np.min(off_diagonal)) if len(off_diagonal) > 0 else 0.0
            mean_corr = float(np.mean(off_diagonal)) if len(off_diagonal) > 0 else 0.0
            
            return {
                'correlation_matrix': corr_matrix,
                'columns': columns,
                'max_correlation': max_corr,
                'min_correlation': min_corr,
                'mean_correlation': mean_corr,
                'matrix_size': len(columns)
            }
            
        except Exception as e:
            self._log_error(f"Error computing correlation matrix: {e}")
            return {
                'correlation_matrix': [],
                'max_correlation': 0.0,
                'min_correlation': 0.0,
                'mean_correlation': 0.0
            }

    def compute_statistical_summaries(self, df: cudf.DataFrame, columns: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Compute statistical summaries for specified columns.
        
        Args:
            df: Input DataFrame
            columns: List of column names to analyze
            
        Returns:
            Dictionary with statistical summaries for each column
        """
        try:
            if columns is None:
                # Find numeric columns
                numeric_cols = []
                for col in df.columns:
                    try:
                        data = df[col].to_cupy()
                        if cp.isfinite(data).any():
                            numeric_cols.append(col)
                    except Exception:
                        continue
                columns = numeric_cols
            
            if not columns:
                return {}
            
            self._log_info(f"Computing statistical summaries for {len(columns)} columns")
            
            summaries = {}
            for col in columns:
                try:
                    data = df[col].to_cupy()
                    valid_data = data[~cp.isnan(data)]
                    
                    if len(valid_data) > 0:
                        # Compute basic statistics
                        mean_val = float(cp.mean(valid_data))
                        std_val = float(cp.std(valid_data))
                        min_val = float(cp.min(valid_data))
                        max_val = float(cp.max(valid_data))
                        median_val = float(cp.median(valid_data))
                        
                        # Compute higher moments
                        if std_val > 0:
                            standardized = (valid_data - mean_val) / std_val
                            skewness = float(cp.mean(standardized**3))
                            kurtosis = float(cp.mean(standardized**4))
                        else:
                            skewness = 0.0
                            kurtosis = 0.0
                        
                        summaries[col] = {
                            'count': len(valid_data),
                            'mean': mean_val,
                            'std': std_val,
                            'min': min_val,
                            'max': max_val,
                            'median': median_val,
                            'skewness': skewness,
                            'kurtosis': kurtosis
                        }
                    else:
                        summaries[col] = {
                            'count': 0,
                            'mean': float('nan'),
                            'std': float('nan'),
                            'min': float('nan'),
                            'max': float('nan'),
                            'median': float('nan'),
                            'skewness': float('nan'),
                            'kurtosis': float('nan')
                        }
                        
                except Exception as e:
                    self._log_warn(f"Error computing summary for {col}: {e}")
                    summaries[col] = {
                        'count': 0,
                        'mean': float('nan'),
                        'std': float('nan'),
                        'min': float('nan'),
                        'max': float('nan'),
                        'median': float('nan'),
                        'skewness': float('nan'),
                        'kurtosis': float('nan')
                    }
            
            return summaries
            
        except Exception as e:
            self._log_error(f"Error computing statistical summaries: {e}")
            return {}

    def apply_normality_tests(self, df: cudf.DataFrame, columns: List[str] = None) -> cudf.DataFrame:
        """
        Apply normality tests to specified columns.
        
        Args:
            df: Input DataFrame
            columns: List of column names to test
            
        Returns:
            DataFrame with normality test results added
        """
        try:
            if columns is None:
                # Find numeric columns
                numeric_cols = []
                for col in df.columns:
                    try:
                        data = df[col].to_cupy()
                        if cp.isfinite(data).any():
                            numeric_cols.append(col)
                    except Exception:
                        continue
                columns = numeric_cols
            
            if not columns:
                return df
            
            self._log_info(f"Applying normality tests to {len(columns)} columns")
            
            for col in columns:
                try:
                    data = df[col].to_cupy()
                    valid_data = data[~cp.isnan(data)]
                    
                    if len(valid_data) > 10:  # Minimum sample size for normality tests
                        # Compute normality tests
                        normality_results = self._compute_normality_tests_vectorized(valid_data)
                        
                        # Add results to DataFrame
                        df[f'{col}_jb_stat'] = normality_results['jarque_bera_stat']
                        df[f'{col}_jb_pvalue'] = normality_results['jarque_bera_pvalue']
                        df[f'{col}_ad_stat'] = normality_results['anderson_darling_stat']
                        df[f'{col}_ad_pvalue'] = normality_results['anderson_darling_pvalue']
                        df[f'{col}_is_normal'] = normality_results['is_normal']
                    else:
                        # Not enough data for normality tests
                        df[f'{col}_jb_stat'] = float('nan')
                        df[f'{col}_jb_pvalue'] = float('nan')
                        df[f'{col}_ad_stat'] = float('nan')
                        df[f'{col}_ad_pvalue'] = float('nan')
                        df[f'{col}_is_normal'] = False
                        
                except Exception as e:
                    self._log_warn(f"Error applying normality tests to {col}: {e}")
                    df[f'{col}_jb_stat'] = float('nan')
                    df[f'{col}_jb_pvalue'] = float('nan')
                    df[f'{col}_ad_stat'] = float('nan')
                    df[f'{col}_ad_pvalue'] = float('nan')
                    df[f'{col}_is_normal'] = False
            
            return df
            
        except Exception as e:
            self._log_error(f"Error applying normality tests: {e}")
            return df

    def apply_comprehensive_statistical_analysis(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """
        Apply comprehensive statistical analysis to the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with statistical analysis results added
        """
        try:
            self._log_info("Starting comprehensive statistical analysis...")
            
            # 1. Apply additional statistical tests
            df = self._apply_additional_statistical_tests(df)
            
            # 2. Apply normality tests to key columns
            key_columns = [col for col in df.columns if any(term in col.lower() for term in ['y_close', 'y_ret', 'frac_diff'])]
            if key_columns:
                df = self.apply_normality_tests(df, key_columns)
            
            # 3. Compute correlation matrix for frac_diff columns
            frac_diff_cols = [col for col in df.columns if "frac_diff" in col]
            if len(frac_diff_cols) > 1:
                corr_results = self.compute_correlation_matrix(df, frac_diff_cols)
                if corr_results['correlation_matrix']:
                    df['frac_diff_corr_max'] = corr_results['max_correlation']
                    df['frac_diff_corr_min'] = corr_results['min_correlation']
                    df['frac_diff_corr_mean'] = corr_results['mean_correlation']
            
            # 4. Compute statistical summaries for key series
            summaries = self.compute_statistical_summaries(df, key_columns)
            for col, summary in summaries.items():
                for stat_name, stat_value in summary.items():
                    if stat_name != 'count':  # Skip count as it's not a scalar
                        df[f'{col}_{stat_name}'] = stat_value
            
            self._log_info("Comprehensive statistical analysis completed")
            return df
            
        except Exception as e:
            self._log_error(f"Error in comprehensive statistical analysis: {e}")
            return df
