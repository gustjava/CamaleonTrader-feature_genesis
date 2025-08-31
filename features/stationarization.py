"""
Stationarization Module for Dynamic Stage 0 Pipeline

This module implements GPU-accelerated stationarization techniques including:
- Fractional differentiation for memory-preserving stationarization
- Rolling window stationarization
- Variance stabilization
- Rolling correlations for dynamic relationship analysis
"""

import logging
import numpy as np
import cupy as cp
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import dask_cudf
import cudf
from functools import lru_cache

from config.settings import Settings
from dask.distributed import Client
from .base_engine import BaseFeatureEngine

logger = logging.getLogger(__name__)


@dataclass
class FracDiffConfig:
    """Configuration for fractional differentiation parameters."""
    d_values: List[float] = None  # Fractional differentiation orders
    threshold: float = 1e-5       # Threshold for stationarity
    max_lag: int = 1000          # Maximum lag for computation
    min_periods: int = 50        # Minimum periods required


class StationarizationEngine(BaseFeatureEngine):
    """
    GPU-accelerated stationarization engine for financial time series.
    
    Implements fractional differentiation and other techniques to achieve
    stationarity while preserving memory properties of the original series.
    """
    
    def __init__(self, settings: Settings, client: Client):
        """Initialize the stationarization engine with configuration."""
        super().__init__(settings, client)
        self.config = FracDiffConfig(
            d_values=self.settings.features.frac_diff_values,
            threshold=self.settings.features.frac_diff_threshold,
            max_lag=self.settings.features.frac_diff_max_lag
        )
    
    # ---------- pesos da fracdiff (corretos, truncados e cacheados) ----------
    @staticmethod
    @lru_cache(maxsize=64)
    def _fracdiff_weights_cpu(d: float, max_lag: int, tol: float = 0.0) -> np.ndarray:
        """
        Pesos de (1 - L)^d. Trunca por 'tol' (fração do peso acumulado) ou por 'max_lag'.
        Retorna em float32 (CPU) para depois enviar à GPU.
        """
        m = max_lag
        w = np.empty(m + 1, dtype=np.float32)
        w[0] = 1.0
        for k in range(1, m + 1):
            w[k] = -w[k-1] * (d - k + 1) / k  # <<< sinal negativo correto

        if tol and tol > 0:
            # trunca quando o peso residual acumulado fica < tol
            aw = np.abs(w)
            cum = np.cumsum(aw)
            total = cum[-1]
            # ponto a partir do qual o residual é pequeno
            keep = np.searchsorted(cum, (1.0 - tol) * total, side='left') + 1
            w = w[:keep]

        return w
    
    def _compute_fractional_diff(self, data: cp.ndarray, d: float) -> cp.ndarray:
        """
        Fracdiff de ordem d usando convolução em GPU.
        Define NaN nas primeiras (len(w)-1) posições por borda.
        """
        n = int(data.size)
        w_cpu = self._fracdiff_weights_cpu(d, max_lag=min(self.config.max_lag, n-1),
                                           tol=getattr(self.config, "threshold", 0.0))
        w = cp.asarray(w_cpu)  # envia pesos p/ GPU

        # kernel curto: conv direta costuma ser mais rápida; longo: FFT
        if w.size <= 129:
            y = cp.convolve(data, w, mode="full")[:n]
        else:
            try:
                from cusignal import fftconvolve
                y = fftconvolve(data, w, mode="full")[:n]
            except ImportError:
                # Fallback to scipy if cusignal is not available
                from scipy.signal import fftconvolve as scipy_fftconvolve
                y = cp.asarray(scipy_fftconvolve(cp.asnumpy(data), cp.asnumpy(w), mode="full")[:n])

        # borda inválida
        k = w.size - 1
        if k > 0:
            y[:k] = cp.nan
        return y.astype(cp.float32, copy=False)
    
    def _test_stationarity(self, data: cp.ndarray) -> Dict[str, Any]:
        """
        Test stationarity using variance ratio test.
        """
        try:
            # Remove NaN values
            valid_data = data[~cp.isnan(data)]
            if len(valid_data) < 100:
                return {'is_stationary': False, 'variance_ratio': None}
            
            # Test periods: 2, 4, 8, 16
            periods = [2, 4, 8, 16]
            ratios = []
            
            for period in periods:
                if len(valid_data) < period * 2:
                    continue
                
                # Calculate variance ratio
                returns = cp.diff(valid_data)
                var_1 = cp.var(returns)
                
                # Aggregate returns
                agg_returns = cp.array([cp.sum(returns[i:i+period]) for i in range(0, len(returns)-period+1, period)])
                var_period = cp.var(agg_returns)
                
                if var_1 > 0:
                    ratio = var_period / (period * var_1)
                    ratios.append(float(ratio))
                else:
                    ratios.append(1.0)
            
            if not ratios:
                return {'is_stationary': False, 'variance_ratio': None}
            
            # Consider stationary if variance ratio is close to 1
            mean_ratio = np.mean(ratios)
            is_stationary = abs(mean_ratio - 1.0) < 0.1  # More realistic threshold
            
            return {
                'is_stationary': bool(is_stationary),
                'variance_ratio': float(mean_ratio)
            }
            
        except Exception as e:
            self._log_error(f"Error in stationarity test: {e}")
            return {'is_stationary': False, 'variance_ratio': None}
    
    def find_optimal_d(self, data: cp.ndarray, column_name: str = "series"):
        """
        Find optimal fractional differentiation order.
        """
        try:
            d_grid = self.config.d_values
            if not d_grid:
                self._log_error("No d_values configured for optimal d search")
                return None

            best = None
            for d in d_grid:
                y = self._compute_fractional_diff(data, d)
                stats = self._test_stationarity(y)
                if best is None:
                    best = (d, y, stats)
                else:
                    # escolha pelo |VR-1| menor; preferindo os estacionários
                    curr = abs((stats['variance_ratio'] or np.inf) - 1.0)
                    prev = abs((best[2]['variance_ratio'] or np.inf) - 1.0)
                    take = (stats['is_stationary'] and not best[2]['is_stationary']) or (curr < prev)
                    if take:
                        best = (d, y, stats)

            if best is None:
                return None

            d_opt, y_opt, st = best
            # estatísticas adicionais (protege contra std=0)
            mu_o, sd_o = float(cp.mean(data)), float(cp.std(data))
            mu_y, sd_y = float(cp.mean(y_opt)), float(cp.std(y_opt))
            def _mom(x, mu, sd, p):
                if sd == 0: return float("nan")
                z = (x - mu) / sd
                return float(cp.mean(z**p))

            return {
                'differentiated_series': y_opt,
                'd_order': d_opt,
                'optimal_d': d_opt,
                'is_stationary': bool(st['is_stationary']),
                'variance_ratio': st['variance_ratio'],
                'original_stats': {'mean': mu_o, 'std': sd_o,
                                   'skewness': _mom(data, mu_o, sd_o, 3),
                                   'kurtosis': _mom(data, mu_o, sd_o, 4)},
                'differentiated_stats': {'mean': mu_y, 'std': sd_y,
                                         'skewness': _mom(y_opt, mu_y, sd_y, 3),
                                         'kurtosis': _mom(y_opt, mu_y, sd_y, 4)},
                'sample_size': int(data.size),
                'all_d_values': d_grid,
            }
        except Exception as e:
            self._log_error(f"Error finding optimal d for {column_name}: {e}")
            return None
    
    def rolling_stationarization(self, data: cp.ndarray, window: int = 252, column_name: str = "series"):
        """
        Rolling stationarization using cudf.rolling (fast on GPU).
        """
        try:
            if data.size < window:
                self._log_warn(f"Insufficient data for rolling stationarization: {data.size} < {window}")
                return None

            s = cudf.Series(data.astype(np.float32, copy=False))
            mean = s.rolling(window).mean()
            std  = s.rolling(window).std()
            z = (s - mean) / std
            z = z.astype(np.float32)
            return z.to_cupy()
        except Exception as e:
            self._log_error(f"Error in rolling stationarization for {column_name}: {e}")
            return None
    
    def variance_stabilization(self, data: cp.ndarray, method: str = "log", column_name: str = "series"):
        """
        Variance stabilization with robust epsilon handling.
        """
        try:
            x = data.astype(cp.float32, copy=False)
            if method == "log":
                eps = cp.asarray(1e-8, dtype=x.dtype)
                shift = cp.minimum(x.min(), 0)
                x = x - shift + eps
                return cp.log(x)
            elif method == "sqrt":
                eps = cp.asarray(0.0, dtype=x.dtype)
                shift = cp.minimum(x.min(), 0)
                x = x - shift + eps
                return cp.sqrt(x)
            elif method == "boxcox":
                # aproximação λ=0 → log, com mesmo shift do caso log
                eps = cp.asarray(1e-8, dtype=x.dtype)
                shift = cp.minimum(x.min(), 0)
                x = x - shift + eps
                return cp.log(x)
            else:
                self._log_error(f"Unknown variance stabilization method: {method}")
                return None
        except Exception as e:
            self._log_error(f"Error in variance stabilization for {column_name}: {e}")
            return None
    
    def process(self, df: dask_cudf.DataFrame) -> dask_cudf.DataFrame:
        """
        Main processing method for dask_cudf DataFrame.
        
        Applies stationarization techniques including:
        - Fractional differentiation
        - Rolling stationarization
        - Variance stabilization
        - Rolling correlations
        
        Args:
            df: Input dask_cudf DataFrame
            
        Returns:
            DataFrame with additional stationarized features
        """
        try:
            self._log_info("Starting stationarization pipeline...")
            
            # Apply fractional differentiation
            df = self._apply_fractional_differentiation(df)
            
            # Apply rolling correlations
            df = self._compute_rolling_correlations(df)
            
            # Apply rolling stationarization
            df = self._apply_rolling_stationarization(df)
            
            # Apply variance stabilization
            df = self._apply_variance_stabilization(df)
            
            self._log_info("Stationarization pipeline completed successfully")
            return df
            
        except Exception as e:
            self._log_error(f"Error in stationarization pipeline: {e}")
            return df
    
    def _compute_rolling_correlations(self, df: dask_cudf.DataFrame) -> dask_cudf.DataFrame:
        """
        Calcula correlações rolantes para pares de features pré-definidos.
        
        Args:
            df: Input dask_cudf DataFrame
            
        Returns:
            DataFrame with rolling correlation features added
        """
        self._log_progress("Applying rolling correlations...")

        # Pares de features para analisar
        feature_pairs = [
            ('returns', 'tick_volume'),
            ('returns', 'ofi'),
            ('spread_rel', 'realized_vol'),
            ('close', 'volume'),
            ('returns', 'spread_rel'),
            ('tick_volume', 'spread_rel')
        ]

        # Janelas definidas no config.yaml
        windows = self.settings.features.rolling_windows
        min_periods = self.settings.features.rolling_min_periods

        for col1, col2 in feature_pairs:
            if col1 in df.columns and col2 in df.columns:
                for window in windows:
                    self._log_progress(f"  -> Corr({col1}, {col2}) | Window={window}")

                    # Create a function for this specific correlation computation
                    def compute_corr_for_pair(pdf, col1=col1, col2=col2, window=window, min_periods=min_periods):
                        return self._compute_rolling_corr_partition(pdf, col1, col2, window, min_periods)
                    
                    # Calculate rolling correlation using dask_cudf
                    df = df.map_partitions(compute_corr_for_pair, meta=df._meta)

        return df
    
    def _compute_rolling_corr_partition(self, pdf: cudf.DataFrame, col1: str, col2: str, 
                                       window: int, min_periods: int) -> cudf.DataFrame:
        """
        Compute rolling correlation for a single partition.
        
        Args:
            pdf: cuDF DataFrame partition
            col1: First column name
            col2: Second column name
            window: Rolling window size
            min_periods: Minimum periods required
            
        Returns:
            DataFrame with rolling correlation column added
        """
        try:
            # Calculate rolling correlation
            rolling_corr = pdf[[col1, col2]].rolling(
                window=window,
                min_periods=min_periods
            ).corr()
            
            # Extract the correlation between col1 and col2
            # The result is a Series with MultiIndex, we need to unstack and get the cross-correlation
            if not rolling_corr.empty:
                # Unstack to get correlation matrix
                corr_matrix = rolling_corr.unstack()
                
                # Get the correlation between col1 and col2
                # This will be in the position (col1, col2) of the correlation matrix
                if col2 in corr_matrix.columns:
                    corr_series = corr_matrix[col2]
                    if col1 in corr_series.index:
                        corr_values = corr_series[col1]
                    else:
                        # If col1 is not in index, try the other way around
                        corr_values = corr_matrix.loc[col1, col2] if col1 in corr_matrix.index else cudf.Series([np.nan] * len(pdf), dtype=np.float64)
                else:
                    corr_values = cudf.Series([np.nan] * len(pdf), dtype=np.float64)
            else:
                corr_values = cudf.Series([np.nan] * len(pdf), dtype=np.float64)
            
            # Ensure the series has numeric dtype
            corr_values = corr_values.astype(np.float64)
            
            # Rename the new column and add to DataFrame
            new_col_name = f"rolling_corr_{col1}_{col2}_{window}w"
            pdf[new_col_name] = corr_values
            
        except Exception as e:
            self._log_error(f"Error computing rolling correlation for {col1}-{col2}: {e}")
            # Add NaN column if computation fails
            new_col_name = f"rolling_corr_{col1}_{col2}_{window}w"
            pdf[new_col_name] = cudf.Series([np.nan] * len(pdf), dtype=np.float64)
        
        return pdf
    
    def _apply_fractional_differentiation(self, df: dask_cudf.DataFrame) -> dask_cudf.DataFrame:
        """
        Apply fractional differentiation to relevant columns.
        
        Args:
            df: Input dask_cudf DataFrame
            
        Returns:
            DataFrame with fractional differentiation features
        """
        self._log_progress("Applying fractional differentiation...")
        
        # Apply to close prices if available
        if 'close' in df.columns:
            def apply_frac_diff_close(pdf):
                return self._apply_frac_diff_to_partition(pdf, 'close')
            df = df.map_partitions(apply_frac_diff_close, meta=df._meta)
        
        # Apply to returns if available
        if 'returns' in df.columns:
            def apply_frac_diff_returns(pdf):
                return self._apply_frac_diff_to_partition(pdf, 'returns')
            df = df.map_partitions(apply_frac_diff_returns, meta=df._meta)
        
        return df
    
    def _apply_frac_diff_to_partition(self, pdf: cudf.DataFrame, column: str) -> cudf.DataFrame:
        """
        Apply fractional differentiation to a single partition.
        
        Args:
            pdf: cuDF DataFrame partition
            column: Column name to apply fractional differentiation to
            
        Returns:
            DataFrame with fractional differentiation features added
        """
        try:
            data = pdf[column].to_cupy()
            frac_diff_result = self.find_optimal_d(data, column)
            
            if frac_diff_result:
                # Add fractionally differentiated series
                pdf[f'frac_diff_{column}'] = frac_diff_result['differentiated_series']
                pdf[f'frac_diff_{column}_d'] = frac_diff_result['optimal_d']
                pdf[f'frac_diff_{column}_stationary'] = frac_diff_result['is_stationary']
                pdf[f'frac_diff_{column}_variance_ratio'] = frac_diff_result['variance_ratio']
                
                # Add statistics
                pdf[f'frac_diff_{column}_mean'] = frac_diff_result['differentiated_stats']['mean']
                pdf[f'frac_diff_{column}_std'] = frac_diff_result['differentiated_stats']['std']
                pdf[f'frac_diff_{column}_skewness'] = frac_diff_result['differentiated_stats']['skewness']
                pdf[f'frac_diff_{column}_kurtosis'] = frac_diff_result['differentiated_stats']['kurtosis']
        
        except Exception as e:
            self._log_error(f"Error applying fractional differentiation to {column}: {e}")
        
        return pdf
    
    def _apply_rolling_stationarization(self, df: dask_cudf.DataFrame) -> dask_cudf.DataFrame:
        """
        Apply rolling stationarization to relevant columns.
        
        Args:
            df: Input dask_cudf DataFrame
            
        Returns:
            DataFrame with rolling stationarization features
        """
        self._log_progress("Applying rolling stationarization...")
        
        # Apply to close prices if available
        if 'close' in df.columns:
            def apply_rolling_stationary_close(pdf):
                return self._apply_rolling_stationary_to_partition(pdf, 'close')
            df = df.map_partitions(apply_rolling_stationary_close, meta=df._meta)
        
        return df
    
    def _apply_rolling_stationary_to_partition(self, pdf: cudf.DataFrame, column: str) -> cudf.DataFrame:
        """
        Apply rolling stationarization to a single partition.
        
        Args:
            pdf: cuDF DataFrame partition
            column: Column name to apply rolling stationarization to
            
        Returns:
            DataFrame with rolling stationarization features added
        """
        try:
            data = pdf[column].to_cupy()
            rolling_stationary = self.rolling_stationarization(data, window=252, column_name=column)
            if rolling_stationary is not None:
                pdf[f'rolling_stationary_{column}'] = rolling_stationary
        
        except Exception as e:
            self._log_error(f"Error applying rolling stationarization to {column}: {e}")
        
        return pdf
    
    def _apply_variance_stabilization(self, df: dask_cudf.DataFrame) -> dask_cudf.DataFrame:
        """
        Apply variance stabilization to relevant columns.
        
        Args:
            df: Input dask_cudf DataFrame
            
        Returns:
            DataFrame with variance stabilization features
        """
        self._log_progress("Applying variance stabilization...")
        
        # Apply to close prices if available
        if 'close' in df.columns:
            def apply_variance_stabilization_close(pdf):
                return self._apply_variance_stabilization_to_partition(pdf, 'close')
            df = df.map_partitions(apply_variance_stabilization_close, meta=df._meta)
        
        return df
    
    def _apply_variance_stabilization_to_partition(self, pdf: cudf.DataFrame, column: str) -> cudf.DataFrame:
        """
        Apply variance stabilization to a single partition.
        
        Args:
            pdf: cuDF DataFrame partition
            column: Column name to apply variance stabilization to
            
        Returns:
            DataFrame with variance stabilization features added
        """
        try:
            data = pdf[column].to_cupy()
            log_stabilized = self.variance_stabilization(data, method="log", column_name=column)
            if log_stabilized is not None:
                pdf[f'log_stabilized_{column}'] = log_stabilized
        
        except Exception as e:
            self._log_error(f"Error applying variance stabilization to {column}: {e}")
        
        return pdf
    
    def process_cudf(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """
        Apply stationarization techniques to a cuDF DataFrame.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional stationarized features
        """
        try:
            self._log_info("Starting stationarization for cuDF DataFrame")
            
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Apply fractional differentiation to close prices
            if 'close' in df.columns:
                close_prices = df['close'].values
                frac_diff_result = self.find_optimal_d(close_prices, "close")
                
                if frac_diff_result:
                    # Add fractionally differentiated series
                    result_df['frac_diff_close'] = frac_diff_result['differentiated_series']
                    result_df['frac_diff_d'] = frac_diff_result['optimal_d']
                    result_df['frac_diff_stationary'] = frac_diff_result['is_stationary']
                    result_df['frac_diff_variance_ratio'] = frac_diff_result['variance_ratio']
                    
                    # Add statistics
                    result_df['frac_diff_mean'] = frac_diff_result['differentiated_stats']['mean']
                    result_df['frac_diff_std'] = frac_diff_result['differentiated_stats']['std']
                    result_df['frac_diff_skewness'] = frac_diff_result['differentiated_stats']['skewness']
                    result_df['frac_diff_kurtosis'] = frac_diff_result['differentiated_stats']['kurtosis']
            
            # Apply rolling correlations
            result_df = self._compute_rolling_correlations_cudf(result_df)
            
            # Apply rolling stationarization
            if 'close' in df.columns:
                rolling_stationary = self.rolling_stationarization(close_prices, window=252, column_name="close")
                if rolling_stationary is not None:
                    result_df['rolling_stationary_close'] = rolling_stationary
            
            # Apply variance stabilization
            if 'close' in df.columns:
                log_stabilized = self.variance_stabilization(close_prices, method="log", column_name="close")
                if log_stabilized is not None:
                    result_df['log_stabilized_close'] = log_stabilized
            
            # Apply to returns if volume exists
            if 'volume' in df.columns:
                returns = cp.diff(cp.log(close_prices))
                returns = cp.concatenate([cp.array([cp.nan]), returns])  # Align with original
                
                # Fractional differentiation of returns
                returns_frac_diff = self.find_optimal_d(returns[1:], "returns")  # Skip NaN
                if returns_frac_diff:
                    # Pad with NaN to match original length
                    diff_returns = cp.concatenate([cp.array([cp.nan]), returns_frac_diff['differentiated_series']])
                    result_df['frac_diff_returns'] = diff_returns
                    result_df['frac_diff_returns_d'] = returns_frac_diff['optimal_d']
                    result_df['frac_diff_returns_stationary'] = returns_frac_diff['is_stationary']
            
            self._log_info("Stationarization completed successfully")
            return result_df
            
        except Exception as e:
            self._log_error(f"Error in stationarization: {e}")
            return df
    
    def _compute_rolling_correlations_cudf(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """
        Calcula correlações rolantes para pares de features pré-definidos (cuDF version).
        
        Args:
            df: Input cuDF DataFrame
            
        Returns:
            DataFrame with rolling correlation features added
        """
        self._log_progress("Applying rolling correlations (cuDF)...")

        # Pares de features para analisar
        feature_pairs = [
            ('returns', 'tick_volume'),
            ('returns', 'ofi'),
            ('spread_rel', 'realized_vol'),
            ('close', 'volume'),
            ('returns', 'spread_rel'),
            ('tick_volume', 'spread_rel')
        ]

        # Janelas definidas no config.yaml
        windows = self.settings.features.rolling_windows
        min_periods = self.settings.features.rolling_min_periods

        for col1, col2 in feature_pairs:
            if col1 in df.columns and col2 in df.columns:
                for window in windows:
                    self._log_progress(f"  -> Corr({col1}, {col2}) | Window={window}")

                    try:
                        # Calculate rolling correlation
                        rolling_corr = df[[col1, col2]].rolling(
                            window=window,
                            min_periods=min_periods
                        ).corr()
                        
                        # Extract the correlation between col1 and col2
                        if not rolling_corr.empty:
                            # Unstack to get correlation matrix
                            corr_matrix = rolling_corr.unstack()
                            
                            # Get the correlation between col1 and col2
                            if col2 in corr_matrix.columns:
                                corr_series = corr_matrix[col2]
                                if col1 in corr_series.index:
                                    corr_values = corr_series[col1]
                                else:
                                    # If col1 is not in index, try the other way around
                                    corr_values = corr_matrix.loc[col1, col2] if col1 in corr_matrix.index else cudf.Series([np.nan] * len(df), dtype=np.float64)
                            else:
                                corr_values = cudf.Series([np.nan] * len(df), dtype=np.float64)
                        else:
                            corr_values = cudf.Series([np.nan] * len(df), dtype=np.float64)
                        
                        # Ensure the series has numeric dtype
                        corr_values = corr_values.astype(np.float64)
                        
                        # Rename the new column and add to DataFrame
                        new_col_name = f"rolling_corr_{col1}_{col2}_{window}w"
                        df[new_col_name] = corr_values
                        
                    except Exception as e:
                        self._log_error(f"Error computing rolling correlation for {col1}-{col2}: {e}")
                        # Add NaN column if computation fails
                        new_col_name = f"rolling_corr_{col1}_{col2}_{window}w"
                        df[new_col_name] = cudf.Series([np.nan] * len(df), dtype=np.float64)

        return df
    
    def process_currency_pair(
        self, 
        df: 'cudf.DataFrame'
    ) -> Optional['cudf.DataFrame']:
        """
        Apply stationarization techniques to a currency pair DataFrame.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional stationarized features, or None if failed
        """
        try:
            self._log_info("Starting stationarization for currency pair")
            
            # Use the new process_cudf method for consistency
            result_df = self.process_cudf(df)
            
            self._log_info("Stationarization completed successfully")
            return result_df
            
        except Exception as e:
            self._log_error(f"Error in stationarization: {e}")
            return None
    
    def get_stationarization_info(self) -> Dict[str, Any]:
        """Get information about the stationarization techniques."""
        return {
            'available_methods': ['FractionalDifferentiation', 'RollingStationarization', 'VarianceStabilization', 'RollingCorrelations'],
            'frac_diff_config': {
                'd_values': self.config.d_values,
                'threshold': self.config.threshold,
                'max_lag': self.config.max_lag,
                'min_periods': self.config.min_periods
            },
            'rolling_corr_config': {
                'windows': self.settings.features.rolling_windows,
                'min_periods': self.settings.features.rolling_min_periods,
                'feature_pairs': [
                    ('returns', 'tick_volume'),
                    ('returns', 'ofi'),
                    ('spread_rel', 'realized_vol'),
                    ('close', 'volume'),
                    ('returns', 'spread_rel'),
                    ('tick_volume', 'spread_rel')
                ]
            },
            'description': 'GPU-accelerated stationarization techniques for financial time series including rolling correlations for dynamic relationship analysis'
        }
