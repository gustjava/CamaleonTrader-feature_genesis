"""
Distance Correlation Module

This module contains functionality for computing distance correlation between variables,
including batch processing, rolling windows, and permutation testing for significance.
"""

import logging
import time
import numpy as np
import cupy as cp
import cudf
from typing import List, Dict, Any, Tuple
from .utils import _adaptive_tile

logger = logging.getLogger(__name__)


class DistanceCorrelation:
    """Class for computing distance correlation between variables."""
    
    def __init__(self, logger_instance=None, dcor_max_samples=10000, dcor_tile_size=2048):
        """Initialize distance correlation with configuration parameters."""
        self.logger = logger_instance or logger
        self.dcor_max_samples = dcor_max_samples
        self.dcor_tile_size = dcor_tile_size
    
    def _log_info(self, message: str, **kwargs):
        """Log info message with optional context."""
        if self.logger:
            self.logger.info(f"dCor: {message}", extra=kwargs)
    
    def _log_warn(self, message: str, **kwargs):
        """Log warning message with optional context."""
        if self.logger:
            self.logger.warning(f"dCor: {message}", extra=kwargs)
    
    def _log_error(self, message: str, **kwargs):
        """Log error message with optional context."""
        if self.logger:
            self.logger.error(f"dCor: {message}", extra=kwargs)
    
    def _critical_error(self, message: str, **kwargs):
        """Log critical error and raise exception."""
        self._log_error(message, **kwargs)
        raise RuntimeError(f"dCor Critical Error: {message}")
    
    def _distance_correlation_gpu(self, x: cp.ndarray, y: cp.ndarray, tile: int = 2048, max_n: int = None) -> float:
        """Distance correlation for 1D arrays on GPU using chunked centering.

        Implements a two-pass algorithm without forming full n×n matrices in memory.
        """
        try:
            # Remove NaNs and invalid values
            mask = ~(cp.isnan(x) | cp.isnan(y))
            x = x[mask]
            y = y[mask]
            n = int(x.size)
            if n < 2:  # Need at least 2 points for distance correlation
                return float('nan')
            if max_n is not None and n > int(max_n):  # Limit sample size for performance
                x = x[-int(max_n):]  # Take last max_n points
                y = y[-int(max_n):]
                n = int(x.size)
            tile = int(max(1, tile))  # Ensure tile size is at least 1

            # pass 1: row sums and grand means for distance matrices
            a_row_sums = cp.zeros(n, dtype=cp.float64)  # Row sums for x distance matrix
            b_row_sums = cp.zeros(n, dtype=cp.float64)  # Row sums for y distance matrix
            a_total_sum = cp.float64(0.0)  # Total sum for x distance matrix
            b_total_sum = cp.float64(0.0)  # Total sum for y distance matrix
            for i0 in range(0, n, tile):  # Process in tiles to manage memory
                i1 = min(i0 + tile, n)
                xi = x[i0:i1]  # Current tile of x
                yi = y[i0:i1]  # Current tile of y
                for j0 in range(0, n, tile):  # Compare with all other tiles
                    j1 = min(j0 + tile, n)
                    xj = x[j0:j1]  # Comparison tile of x
                    yj = y[j0:j1]  # Comparison tile of y
                    dx = cp.abs(xi[:, None] - xj[None, :])  # Distance matrix block for x
                    dy = cp.abs(yi[:, None] - yj[None, :])  # Distance matrix block for y
                    a_row_sums[i0:i1] += dx.sum(axis=1, dtype=cp.float64)  # Add row sums for current tile
                    b_row_sums[i0:i1] += dy.sum(axis=1, dtype=cp.float64)  # Add row sums for current tile
                    if j0 != i0:  # Avoid double counting diagonal blocks
                        a_row_sums[j0:j1] += dx.sum(axis=0, dtype=cp.float64)  # Add column sums for comparison tile
                        b_row_sums[j0:j1] += dy.sum(axis=0, dtype=cp.float64)  # Add column sums for comparison tile
                    a_total_sum += cp.sum(dx, dtype=cp.float64)  # Add to total sum
                    b_total_sum += cp.sum(dy, dtype=cp.float64)  # Add to total sum

            n_f = float(n)
            a_row_mean = a_row_sums / n_f  # Mean of each row in x distance matrix
            b_row_mean = b_row_sums / n_f  # Mean of each row in y distance matrix
            a_grand = float(a_total_sum / (n_f * n_f))  # Grand mean of x distance matrix
            b_grand = float(b_total_sum / (n_f * n_f))  # Grand mean of y distance matrix

            # pass 2: centered blocks and accumulations
            num = cp.float64(0.0)  # Numerator for distance covariance
            sumA2 = cp.float64(0.0)  # Sum of squared centered x distances
            sumB2 = cp.float64(0.0)  # Sum of squared centered y distances
            for i0 in range(0, n, tile):  # Process in tiles again
                i1 = min(i0 + tile, n)
                xi = x[i0:i1]  # Current tile of x
                yi = y[i0:i1]  # Current tile of y
                a_i_mean = a_row_mean[i0:i1]  # Row means for current tile
                b_i_mean = b_row_mean[i0:i1]  # Row means for current tile
                for j0 in range(0, n, tile):  # Compare with all other tiles
                    j1 = min(j0 + tile, n)
                    xj = x[j0:j1]  # Comparison tile of x
                    yj = y[j0:j1]  # Comparison tile of y
                    a_j_mean = a_row_mean[j0:j1]  # Row means for comparison tile
                    b_j_mean = b_row_mean[j0:j1]  # Row means for comparison tile
                    dx = cp.abs(xi[:, None] - xj[None, :])  # Distance matrix block for x
                    dy = cp.abs(yi[:, None] - yj[None, :])  # Distance matrix block for y
                    # Double centering: subtract row means, column means, and add grand mean
                    A = dx - a_i_mean[:, None] - a_j_mean[None, :] + a_grand
                    B = dy - b_i_mean[:, None] - b_j_mean[None, :] + b_grand
                    num += cp.sum(A * B, dtype=cp.float64)  # Accumulate distance covariance
                    sumA2 += cp.sum(A * A, dtype=cp.float64)  # Accumulate x distance variance
                    sumB2 += cp.sum(B * B, dtype=cp.float64)  # Accumulate y distance variance
            denom = cp.sqrt(sumA2 * sumB2)  # Denominator for distance correlation
            if denom == 0:  # Check for zero denominator
                return 0.0
            return float(num / denom)  # Return distance correlation
        except Exception:
            return float('nan')  # Return NaN if computation fails

    def _dcor_gpu_single(self, x: cp.ndarray, y: cp.ndarray, max_n: int = None) -> float:
        """
        Distance correlation (dCor) via bloco (chunked), estável em memória para séries 1D.
        Implementa duas passagens: (1) médias por linha e média global; (2) centragem e acumulação.
        """
        try:
            # Limpeza de NaNs
            mask = ~(cp.isnan(x) | cp.isnan(y))
            x = x[mask]
            y = y[mask]
            n = int(x.size)

            if n < 2:
                return float("nan")

            # Amostragem/decimação para limitar custo
            if max_n is None:
                max_n = self.dcor_max_samples
            elif n > max_n:
                # fallback: amostra cauda
                x = x[-max_n:]
                y = y[-max_n:]
                n = max_n

            # Garantir tipo consistente para performance/memória
            if x.dtype != cp.float32:
                x = x.astype(cp.float32, copy=False)
            if y.dtype != cp.float32:
                y = y.astype(cp.float32, copy=False)

            # Tamanho do bloco (trade-off memória/tempo)
            tile = _adaptive_tile(self.dcor_tile_size)
            tile = min(tile, n) if n >= 2 * tile else min(max(1024, tile), n)

            t0 = time.time()

            # Passo 1: somas por linha e somas globais (x e y)
            a_row_sums = cp.zeros(n, dtype=cp.float64)
            b_row_sums = cp.zeros(n, dtype=cp.float64)
            a_total_sum = cp.float64(0.0)
            b_total_sum = cp.float64(0.0)

            for i0 in range(0, n, tile):
                i1 = min(i0 + tile, n)
                xi = x[i0:i1]
                yi = y[i0:i1]
                for j0 in range(0, n, tile):
                    j1 = min(j0 + tile, n)
                    xj = x[j0:j1]
                    yj = y[j0:j1]

                    dx = cp.abs(xi[:, None] - xj[None, :])
                    dy = cp.abs(yi[:, None] - yj[None, :])

                    # acumula somas por linha para i-bloco
                    a_row_sums[i0:i1] += dx.sum(axis=1, dtype=cp.float64)
                    b_row_sums[i0:i1] += dy.sum(axis=1, dtype=cp.float64)

                    # para j-bloco, somas por linha equivalem às somas por coluna do bloco i
                    # acumular também para j quando blocos distintos (evita recomputar em outra iteração)
                    if j0 != i0:
                        a_row_sums[j0:j1] += dx.sum(axis=0, dtype=cp.float64)
                        b_row_sums[j0:j1] += dy.sum(axis=0, dtype=cp.float64)

                    a_total_sum += cp.sum(dx, dtype=cp.float64)
                    b_total_sum += cp.sum(dy, dtype=cp.float64)

            # médias
            n_f = float(n)
            a_row_mean = a_row_sums / n_f
            b_row_mean = b_row_sums / n_f
            a_grand = float(a_total_sum / (n_f * n_f))
            b_grand = float(b_total_sum / (n_f * n_f))

            # Passo 2: centragem por blocos e acumulação de somas
            num = cp.float64(0.0)
            sumA2 = cp.float64(0.0)
            sumB2 = cp.float64(0.0)

            for i0 in range(0, n, tile):
                i1 = min(i0 + tile, n)
                xi = x[i0:i1]
                yi = y[i0:i1]
                a_i_mean = a_row_mean[i0:i1]
                b_i_mean = b_row_mean[i0:i1]
                for j0 in range(0, n, tile):
                    j1 = min(j0 + tile, n)
                    xj = x[j0:j1]
                    yj = y[j0:j1]
                    a_j_mean = a_row_mean[j0:j1]
                    b_j_mean = b_row_mean[j0:j1]

                    dx = cp.abs(xi[:, None] - xj[None, :])
                    dy = cp.abs(yi[:, None] - yj[None, :])

                    A = dx - a_i_mean[:, None] - a_j_mean[None, :] + a_grand
                    B = dy - b_i_mean[:, None] - b_j_mean[None, :] + b_grand

                    num += cp.sum(A * B, dtype=cp.float64)
                    sumA2 += cp.sum(A * A, dtype=cp.float64)
                    sumB2 += cp.sum(B * B, dtype=cp.float64)

            denom = cp.sqrt(sumA2 * sumB2)
            if denom == 0:
                return 0.0
            dcor = float(num / denom)
            self._log_info("dCor computed", n=n, tile=int(tile), dcor=round(dcor, 6), elapsed=round(time.time()-t0, 3))
            return dcor
        except Exception as e:
            self._log_error(f"Error in chunked dCor computation: {e}")
            return float("nan")

    def _compute_distance_correlation_vectorized(self, x: cp.ndarray, y: cp.ndarray, max_samples: int = 10000) -> float:
        """
        Cálculo de dCor usando abordagem em blocos (evita matrizes n×n completas).
        """
        try:
            return self._dcor_gpu_single(x, y, max_n=max_samples)
        except Exception as e:
            self._critical_error(f"Error in distance correlation: {e}")

    def distance_correlation_with_permutation(self, x: cp.ndarray, y: cp.ndarray, n_perm: int = 1000) -> Tuple[float, float]:
        """
        Distance correlation with permutation test to estimate p-value.
        """
        dcor_obs = self._compute_distance_correlation_vectorized(x, y)
        
        # inicializa contador de valores permutados >= observado
        greater_count = 0
        # use cupy.random.permutation para embaralhar y em GPU
        for _ in range(n_perm):
            y_perm = cp.random.permutation(y)
            dcor_perm = self._compute_distance_correlation_vectorized(x, y_perm)
            if dcor_perm >= dcor_obs:
                greater_count += 1
        
        p_value = (greater_count + 1) / (n_perm + 1)  # correção de continuidade
        return float(dcor_obs), float(p_value)
    
    def compute_distance_correlation_batch(self, data_pairs: List[Tuple[cp.ndarray, cp.ndarray]], 
                                         max_samples: int = 10000, 
                                         include_permutation: bool = False,
                                         n_perm: int = 1000) -> List[Dict[str, float]]:
        """
        Compute distance correlation for multiple pairs of variables in batch.
        
        Args:
            data_pairs: List of (x, y) pairs
            max_samples: Maximum number of samples per pair
            include_permutation: Whether to include permutation test
            n_perm: Number of permutations for p-value calculation
            
        Returns:
            List of dictionaries with dcor value and optionally p-value
        """
        try:
            results = []
            
            for i, (x, y) in enumerate(data_pairs):
                try:
                    # Remove NaN values
                    valid_mask = ~(cp.isnan(x) | cp.isnan(y))
                    if cp.sum(valid_mask) < 10:
                        if include_permutation:
                            results.append({'dcor_value': 0.0, 'dcor_pvalue': 1.0})
                        else:
                            results.append({'dcor_value': 0.0})
                        continue
                    
                    x_clean = x[valid_mask]
                    y_clean = y[valid_mask]
                    
                    if include_permutation:
                        dcor_value, dcor_pvalue = self.distance_correlation_with_permutation(
                            x_clean, y_clean, n_perm
                        )
                        results.append({
                            'dcor_value': dcor_value,
                            'dcor_pvalue': dcor_pvalue,
                            'dcor_significant': dcor_pvalue < 0.05  # alpha = 0.05
                        })
                    else:
                        dcor_value = self._compute_distance_correlation_vectorized(x_clean, y_clean, max_samples)
                        results.append({'dcor_value': dcor_value})
                    
                except Exception as e:
                    self._log_warn(f"Error computing dCor for pair {i}: {e}")
                    if include_permutation:
                        results.append({'dcor_value': 0.0, 'dcor_pvalue': 1.0, 'dcor_significant': False})
                    else:
                        results.append({'dcor_value': 0.0})
            
            return results
            
        except Exception as e:
            self._critical_error(f"Error in batch distance correlation: {e}")

    def _dcor_partition_gpu(self, pdf: cudf.DataFrame, target: str, candidates: List[str], max_samples: int, tile: int) -> cudf.DataFrame:
        """Compute distance correlation between target and candidate features using GPU for a partition."""
        out = {}
        results_log = []  # Store results for logging
        
        # Get GPU info
        try:
            import cupy as cp
            import os as _os
            current_gpu = cp.cuda.runtime.getDevice()
            visible = _os.environ.get('CUDA_VISIBLE_DEVICES', '')
            gpu_info = f"GPU {current_gpu} (VISIBLE={visible})"
        except Exception:
            gpu_info = "GPU unknown"
        
        try:
            y = pdf[target].astype('f8').to_cupy()  # Convert target to CuPy array
            # Only log once per partition, not for every feature
            if len(candidates) > 0:
                self._log_info(f"dCor partition processing {len(candidates)} features on {gpu_info}", 
                              target=target, sample_size=len(pdf))
        except Exception:
            return cudf.DataFrame([{f"dcor_{c}": float('nan') for c in candidates}])  # Return NaN if conversion fails
        
        for i, c in enumerate(candidates):  # Iterate through candidate features
            try:
                x = pdf[c].astype('f8').to_cupy()  # Convert candidate to CuPy array
                dcor_value = self._distance_correlation_gpu(x, y, tile=tile, max_n=max_samples)  # Compute GPU distance correlation
                out[f"dcor_{c}"] = dcor_value
                
                # Store result for logging
                results_log.append(f"{c}: {dcor_value:.6f}")
                
                # Log progress every 20 features to reduce spam
                if (i + 1) % 20 == 0 or (i + 1) == len(candidates):
                    self._log_info(f"dCor progress: {i + 1}/{len(candidates)} features processed", 
                                  current_feature=c, last_dcor=dcor_value)
                
            except Exception:
                out[f"dcor_{c}"] = float('nan')  # Return NaN if computation fails
                results_log.append(f"{c}: NaN")
        
        # Store results in a way that can be accessed by the main process
        try:
            # Add results as metadata to the DataFrame
            result_df = cudf.DataFrame([out])
            result_df.attrs['dcor_results'] = results_log
            result_df.attrs['target'] = target
            result_df.attrs['n_features'] = len(candidates)
            result_df.attrs['gpu_info'] = gpu_info
            self._log_info(f"Completed dCor computation for {len(candidates)} features", gpu_info=gpu_info)
            return result_df
        except Exception:
            return cudf.DataFrame([out])  # Return results as DataFrame

    def _dcor_rolling_partition_gpu(
        self, pdf: cudf.DataFrame, target: str, candidates: List[str], window: int, step: int, 
        min_periods: int, min_valid_pairs: int, max_rows: int, max_windows: int, agg: str, 
        max_samples: int, tile: int
    ) -> cudf.DataFrame:
        """Compute rolling distance correlation for a partition."""
        # Limit rows to control memory
        if hasattr(pdf, 'head'):
            pdf = pdf.head(int(max_rows))
        try:
            y_all = pdf[target].astype('f8').to_cupy()
        except Exception:
            # Return NaNs and zero counts
            base = {f"dcor_roll_{c}": float('nan') for c in candidates}
            cnts = {f"dcor_roll_cnt_{c}": np.int64(0) for c in candidates}
            return cudf.DataFrame([{**base, **cnts}])
        n = int(y_all.size)
        if n < max(3, int(min_periods)):
            base = {f"dcor_roll_{c}": float('nan') for c in candidates}
            cnts = {f"dcor_roll_cnt_{c}": np.int64(0) for c in candidates}
            return cudf.DataFrame([{**base, **cnts}])

        starts = list(range(0, max(0, n - int(min_periods) + 1), max(1, int(step))))
        if len(starts) > int(max_windows):
            starts = starts[-int(max_windows):]

        score_map: Dict[str, float] = {}
        cnt_map: Dict[str, int] = {}

        # Pre-pull all candidate columns as CuPy
        X_cols: Dict[str, cp.ndarray] = {}
        for c in candidates:
            try:
                X_cols[c] = pdf[c].astype('f8').to_cupy()
            except Exception:
                X_cols[c] = None

        self._log_info(f"Processing {len(candidates)} features with window={window}")
        for i, c in enumerate(candidates):
            if i % 50 == 0:  # Log every 50 features to avoid spam
                self._log_info(f"Processing feature {i+1}/{len(candidates)}: {c}")
            x_all = X_cols.get(c, None)
            if x_all is None:
                score_map[f"dcor_roll_{c}"] = float('nan')
                cnt_map[f"dcor_roll_cnt_{c}"] = np.int64(0)
                continue
            vals: List[float] = []
            for s in starts:
                e = min(n, s + int(window))
                if e - s < int(min_periods):
                    continue
                xv = x_all[s:e]
                yv = y_all[s:e]
                m = ~(cp.isnan(xv) | cp.isnan(yv))
                if int(m.sum().item()) < int(min_valid_pairs):
                    continue
                xv2 = xv[m]
                yv2 = yv[m]
                vals.append(self._distance_correlation_gpu(xv2, yv2, tile=tile, max_n=max_samples))
            if not vals:
                score_map[f"dcor_roll_{c}"] = float('nan')
                cnt_map[f"dcor_roll_cnt_{c}"] = np.int64(0)
            else:
                arr = np.array([v for v in vals if np.isfinite(v)], dtype=float)
                cnt_map[f"dcor_roll_cnt_{c}"] = np.int64(arr.size)
                if arr.size == 0:
                    score_map[f"dcor_roll_{c}"] = float('nan')
                else:
                    if agg == 'mean':
                        score_map[f"dcor_roll_{c}"] = float(np.mean(arr))
                    elif agg == 'min':
                        score_map[f"dcor_roll_{c}"] = float(np.min(arr))
                    elif agg == 'max':
                        score_map[f"dcor_roll_{c}"] = float(np.max(arr))
                    elif agg == 'p25':
                        score_map[f"dcor_roll_{c}"] = float(np.percentile(arr, 25))
                    elif agg == 'p75':
                        score_map[f"dcor_roll_{c}"] = float(np.percentile(arr, 75))
                    else:
                        score_map[f"dcor_roll_{c}"] = float(np.median(arr))

        ordered = {}
        for c in candidates:
            ordered[f"dcor_roll_{c}"] = score_map.get(f"dcor_roll_{c}", float('nan'))
        for c in candidates:
            ordered[f"dcor_roll_cnt_{c}"] = cnt_map.get(f"dcor_roll_cnt_{c}", np.int64(0))
        return cudf.DataFrame([ordered])

    def _apply_comprehensive_distance_correlation(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """
        Apply comprehensive distance correlation analysis.
        
        Generates:
        - dcor_* for each feature vs target
        - dcor_roll_* for rolling distance correlation
        - dcor_pvalue_* for significance testing
        """
        try:
            self._log_info("Applying comprehensive distance correlation analysis...")
            
            # Find target column
            target_col = None
            for col in df.columns:
                if 'y_ret' in col or 'target' in col.lower():
                    target_col = col
                    break
            
            if target_col is None:
                self._log_warn("No target column found for distance correlation analysis")
                return df
            
            # Find candidate features (exclude target and other metrics)
            candidate_features = []
            for col in df.columns:
                if (col != target_col and 
                    not col.startswith(('dcor_', 'adf_', 'stage1_', 'cpcv_', 'y_ret_fwd_')) and
                    'frac_diff' in col):
                    candidate_features.append(col)
            
            if not candidate_features:
                self._log_warn("No candidate features found for distance correlation analysis")
                return df
            
            self._log_info(f"Computing distance correlation for {len(candidate_features)} features vs {target_col}")
            
            # Compute distance correlation for each feature
            for feature in candidate_features:
                try:
                    x = df[feature].to_cupy()
                    y = df[target_col].to_cupy()
                    
                    # Remove NaN values
                    valid_mask = ~(cp.isnan(x) | cp.isnan(y))
                    if cp.sum(valid_mask) > 50:  # Minimum sample size
                        x_clean = x[valid_mask]
                        y_clean = y[valid_mask]
                        
                        # Compute distance correlation
                        dcor_value = self._compute_distance_correlation_vectorized(x_clean, y_clean)
                        
                        # Add to DataFrame
                        base_name = feature.replace('frac_diff_', '')
                        df[f'dcor_{base_name}'] = float(dcor_value)
                        
                        # Compute rolling distance correlation if enabled
                        if hasattr(self, 'stage1_rolling_enabled') and self.stage1_rolling_enabled:
                            rolling_dcor = self._compute_rolling_distance_correlation(
                                df, target_col, feature, 
                                window=getattr(self, 'stage1_rolling_window', 2000),
                                step=getattr(self, 'stage1_rolling_step', 500)
                            )
                            if rolling_dcor is not None:
                                df[f'dcor_roll_{base_name}'] = rolling_dcor
                        
                except Exception as e:
                    self._log_warn(f"Error computing distance correlation for {feature}: {e}")
                    continue
            
            self._log_info("Comprehensive distance correlation analysis completed")
            return df
            
        except Exception as e:
            self._log_error(f"Error in comprehensive distance correlation: {e}")
            return df

    def _compute_rolling_distance_correlation(self, df: cudf.DataFrame, target: str, feature: str, 
                                            window: int = 2000, step: int = 500) -> float:
        """Compute rolling distance correlation for a single feature."""
        try:
            x = df[feature].to_cupy()
            y = df[target].to_cupy()
            
            n = len(x)
            if n < window:
                return None
            
            # Compute rolling windows
            dcor_values = []
            for start in range(0, n - window + 1, step):
                end = start + window
                x_window = x[start:end]
                y_window = y[start:end]
                
                # Remove NaN values
                valid_mask = ~(cp.isnan(x_window) | cp.isnan(y_window))
                if cp.sum(valid_mask) > 100:  # Minimum valid pairs
                    x_clean = x_window[valid_mask]
                    y_clean = y_window[valid_mask]
                    
                    dcor_val = self._compute_distance_correlation_vectorized(x_clean, y_clean)
                    if not cp.isnan(dcor_val):
                        dcor_values.append(float(dcor_val))
            
            if dcor_values:
                return float(np.median(dcor_values))  # Return median of rolling values
            else:
                return None
                
        except Exception as e:
            self._log_warn(f"Error in rolling distance correlation: {e}")
            return None
