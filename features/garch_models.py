"""
GARCH Models Module for Dynamic Stage 0 Pipeline

This module implements GPU-accelerated GARCH model fitting for volatility modeling.
Uses a hybrid CPU-GPU approach with CuPy for log-likelihood computation.
"""

import logging
import numpy as np
import cupy as cp
import dask_cudf
import cudf
from typing import Optional, Dict, Any
from scipy.optimize import minimize

from .base_engine import BaseFeatureEngine
from config.settings import Settings
from dask.distributed import Client

logger = logging.getLogger(__name__)


class GARCHModels(BaseFeatureEngine):
    """
    GPU-accelerated GARCH model fitting engine for volatility modeling.
    """

    def __init__(self, settings: Settings, client: Client):
        """Initialize the GARCH models engine with configuration."""
        super().__init__(settings, client)
        self.config = self.settings.features.garch
        self.max_samples = self.settings.features.distance_corr_max_samples # Reusing this for consistency

    def _log_likelihood_gpu(self, params: np.ndarray, returns: cp.ndarray) -> float:
        """
        Compute log-likelihood using GPU acceleration.
        This is the core function offloaded to the GPU.
        """
        try:
            omega, alpha, beta = cp.asarray(params, dtype=cp.float32)
            n = len(returns)
            h = cp.zeros(n, dtype=cp.float32)
            
            # Initialize with unconditional variance if possible, otherwise simple variance
            uncond_var = cp.var(returns)
            if (1 - alpha - beta) > 1e-8:
                uncond_var = omega / (1 - alpha - beta)
            h[0] = uncond_var

            # GARCH(1,1) recursion
            for t in range(1, n):
                h[t] = omega + alpha * returns[t-1]**2 + beta * h[t-1]

            h = cp.maximum(h, 1e-9) # Ensure positive variance

            log_likelihood = -0.5 * cp.sum(cp.log(2 * cp.pi) + cp.log(h) + returns**2 / h)
            
            # Return negative for minimization
            return -float(log_likelihood) if cp.isfinite(log_likelihood) else np.inf

        except Exception:
            return np.inf

    def fit_garch11(self, series: cudf.Series) -> Optional[Dict[str, Any]]:
        """
        Fit GARCH(1,1) model to a single time series (partition).
        """
        try:
            data = series.to_cupy()

            if len(data) < 100:
                self._log_warn("Insufficient data for GARCH, skipping.", series_len=len(data))
                return None
            
            if len(data) > self.max_samples:
                data = data[-self.max_samples:]
                self._log_info("Truncated data for GARCH fitting.", samples=len(data))
            
            returns = cp.diff(cp.log(data))
            
            initial_params = np.array([cp.var(returns) * 0.01, 0.1, 0.8])
            bounds = [(1e-8, None), (0.0, 1.0), (0.0, 1.0)]
            constraints = {'type': 'ineq', 'fun': lambda params: 1.0 - params[1] - params[2]}
            
            result = minimize(
                fun=self._log_likelihood_gpu,
                x0=initial_params,
                args=(returns,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': self.config.max_iter, 'ftol': self.config.tolerance}
            )

            if not result.success:
                self._log_warn("GARCH optimization failed.", message=result.message)
                return None

            # Return fitted parameters and diagnostics
            omega, alpha, beta = result.x
            persistence = alpha + beta
            log_likelihood = -result.fun
            n_obs = len(returns)
            aic = 2 * 3 - 2 * log_likelihood
            bic = 3 * np.log(n_obs) - 2 * log_likelihood
            is_stationary = persistence < 1.0
            
            self._log_info("GARCH(1,1) fitted successfully.", alpha=f"{alpha:.4f}", beta=f"{beta:.4f}")

            return {
                'garch_omega': omega, 'garch_alpha': alpha, 'garch_beta': beta,
                'garch_persistence': persistence, 'garch_log_likelihood': log_likelihood,
                'garch_aic': aic, 'garch_bic': bic, 'garch_is_stationary': float(is_stationary)
            }
        except Exception as e:
            self._log_error("Error during GARCH fitting.", error=str(e))
            return None

    def _fit_on_partition(self, part: cudf.DataFrame) -> Dict[str, Any]:
        """
        Execute GARCH fitting inside GPU worker (without leaving cluster).
        
        Args:
            part: DataFrame partition with 'close' column
            
        Returns:
            Dictionary with GARCH parameters
        """
        close_series = part["close"]
        res = self.fit_garch11(close_series)
        if res is None:
            # Return keys with NaN to maintain schema
            return {
                'garch_omega': np.nan, 'garch_alpha': np.nan, 'garch_beta': np.nan,
                'garch_persistence': np.nan, 'garch_log_likelihood': np.nan,
                'garch_aic': np.nan, 'garch_bic': np.nan, 'garch_is_stationary': np.nan
            }
        return res

    def process(self, df: dask_cudf.DataFrame) -> dask_cudf.DataFrame:
        """
        Executes the GARCH modeling pipeline (Dask version).
        """
        self._log_info("Starting GARCH (Dask)...")

        # (optional) ensure temporal order
        # df = self._ensure_sorted(df, by="ts")

        # Bring series to single partition in one GPU worker
        one = self._single_partition(df, cols=["close"])

        # Compute model ONCE inside worker and materialize dict in driver
        # (it's small - just scalars)
        meta = {
            'garch_omega': 'f8', 'garch_alpha': 'f8', 'garch_beta': 'f8',
            'garch_persistence': 'f8', 'garch_log_likelihood': 'f8',
            'garch_aic': 'f8', 'garch_bic': 'f8', 'garch_is_stationary': 'f8'
        }

        # Generate a Dask DataFrame with 1 row containing metrics
        params_ddf = one.map_partitions(
            lambda p: cudf.DataFrame([self._fit_on_partition(p)]),
            meta=cudf.DataFrame({k: cudf.Series([], dtype=v) for k, v in meta.items()})
        )

        params_pdf = params_ddf.compute().to_pandas()  # 1 row, cheap
        params = params_pdf.iloc[0].to_dict()

        self._log_info("GARCH fitted.", **{k: float(params[k]) if params[k] == params[k] else None for k in params})

        # Broadcast scalars to original DataFrame (all rows)
        df = self._broadcast_scalars(df, params)
        self._log_info("GARCH features attached.")
        return df

    def process_cudf(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        Executes the GARCH modeling pipeline (cuDF version).
        """
        self._log_info("Starting GARCH (cuDF)...")

        # Fit GARCH directly on cuDF DataFrame
        params = self._fit_on_partition(gdf)
        
        if params is None:
            # Return keys with NaN to maintain schema
            params = {
                'garch_omega': np.nan, 'garch_alpha': np.nan, 'garch_beta': np.nan,
                'garch_persistence': np.nan, 'garch_log_likelihood': np.nan,
                'garch_aic': np.nan, 'garch_bic': np.nan, 'garch_is_stationary': np.nan
            }

        self._log_info("GARCH fitted.", **{k: float(params[k]) if params[k] == params[k] else None for k in params})

        # Add GARCH parameters as columns
        for key, value in params.items():
            gdf[key] = value

        self._log_info("GARCH features attached.")
        return gdf
