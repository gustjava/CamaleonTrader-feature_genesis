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
from config.unified_config import UnifiedConfig
from dask.distributed import Client

logger = logging.getLogger(__name__)


class GARCHModels(BaseFeatureEngine):
    """
    GPU-accelerated GARCH model fitting engine for volatility modeling.
    """

    def __init__(self, settings: UnifiedConfig, client: Client):
        """Initialize the GARCH models engine with configuration."""
        super().__init__(settings, client)
        # Access individual GARCH settings instead of nested object
        self.p = self.settings.features.garch_p
        self.q = self.settings.features.garch_q
        self.max_iter = self.settings.features.garch_max_iter
        self.tolerance = self.settings.features.garch_tolerance
        self.max_samples = self.settings.features.distance_corr_max_samples # Reusing this for consistency

    def _log_likelihood_gpu(self, params: np.ndarray, returns: np.ndarray) -> float:
        """
        Compute log-likelihood using numpy (for scipy compatibility).
        """
        try:
            # Ensure params are float64 for scipy compatibility
            params = np.asarray(params, dtype=np.float64)
            omega, alpha, beta = params
            
            # Ensure returns are float64
            returns = np.asarray(returns, dtype=np.float64)
            n = len(returns)
            h = np.zeros(n, dtype=np.float64)
            
            # Initialize with unconditional variance if possible, otherwise simple variance
            uncond_var = float(np.var(returns))
            if (1 - alpha - beta) > 1e-8:
                uncond_var = omega / (1 - alpha - beta)
            h[0] = uncond_var

            # GARCH(1,1) recursion
            for t in range(1, n):
                h[t] = omega + alpha * returns[t-1]**2 + beta * h[t-1]

            h = np.maximum(h, 1e-9) # Ensure positive variance

            # Compute log-likelihood
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + returns**2 / h)
            
            # Return negative for minimization
            result = -float(log_likelihood) if np.isfinite(log_likelihood) else np.inf
            return result

        except Exception as e:
            self._log_error(f"Error in log likelihood computation: {e}")
            return np.inf

    def _garch_log_likelihood_gpu(self, returns: cp.ndarray, omega: float, alpha: float, beta: float) -> float:
        """
        GPU-accelerated log-likelihood for GARCH(1,1) without Python loops.
        """
        try:
            n = len(returns)
            # prepara vetor de variâncias: v[0] = var inicial
            h0 = cp.var(returns)
            # vetor de retornos ao quadrado
            r2 = returns[:-1]**2  # comprimento n-1

            # calcula recursivamente: h_t = omega + alpha*r_{t-1}^2 + beta*h_{t-1}
            # podemos obter h[1:] = omega/(1-beta) + (alpha*r2) ⊗ K + beta^i*h0,
            # onde K é um kernel de convolução decrescente beta^i.
            betas = beta ** cp.arange(n, dtype=cp.float64)
            # soma ponderada de alphas*r2 com potências de beta (convolução)
            # use cp.signal.convolve se cusignal não estiver disponível
            from cupyx.scipy.signal import lfilter
            # lfilter aplica soma recursiva: y[i] = alpha*r2[i] + beta*y[i-1]
            h_rec = lfilter([alpha], [1, -beta], r2)
            h = omega / (1 - beta) + cp.concatenate([cp.array([h0]), h_rec])
            h = cp.maximum(h, 1e-9)
            
            log_likelihood = -0.5 * cp.sum(cp.log(2*cp.pi) + cp.log(h) + returns**2 / h)
            return float(log_likelihood)
        except Exception as e:
            self._critical_error(f"Error in vectorized GARCH log-likelihood: {e}")
    
    def _garch_log_likelihood_cpu(self, returns: np.ndarray, omega: float, alpha: float, beta: float) -> float:
        """
        CPU fallback for GARCH log-likelihood computation.
        """
        try:
            n = len(returns)
            
            # Initialize variance series
            variance = np.zeros(n)
            variance[0] = np.var(returns)  # Initial variance
            
            # Compute variance series using GARCH recursion
            for t in range(1, n):
                variance[t] = omega + alpha * returns[t-1]**2 + beta * variance[t-1]
            
            # Compute log-likelihood (normal distribution assumption)
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * variance) + returns**2 / variance
            )
            
            return float(log_likelihood)
            
        except Exception as e:
            self._critical_error(f"Error in CPU GARCH log-likelihood: {e}")

    def fit_garch_gpu(self, returns: cp.ndarray, max_iter: int = 1000, tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Fit GARCH(1,1) model using GPU-accelerated log-likelihood computation.
        
        Args:
            returns: Return series on GPU
            max_iter: Maximum optimization iterations
            tolerance: Convergence tolerance
            
        Returns:
            Dictionary with GARCH parameters and fit statistics
        """
        try:
            # Convert to numpy for optimization (only the optimization loop runs on CPU)
            returns_np = cp.asnumpy(returns)
            
            # Initial parameter guesses
            initial_params = np.array([0.01, 0.1, 0.8])  # omega, alpha, beta
            
            # Parameter bounds (ensure stationarity and positivity)
            bounds = [(1e-6, None), (0, 1), (0, 1)]  # omega > 0, 0 <= alpha, beta <= 1
            
            # Constraint: alpha + beta < 1 for stationarity
            def constraint(params):
                return 1 - params[1] - params[2]  # alpha + beta < 1
            
            # Objective function (negative log-likelihood for minimization)
            def objective(params):
                omega, alpha, beta = params
                
                # Check stationarity constraint
                if alpha + beta >= 1:
                    return np.inf
                
                # Use GPU log-likelihood computation
                try:
                    log_lik = self._garch_log_likelihood_gpu(returns, omega, alpha, beta)
                    return -log_lik  # Negative for minimization
                except Exception:
                    # Fallback to CPU
                    log_lik = self._garch_log_likelihood_cpu(returns_np, omega, alpha, beta)
                    return -log_lik
            
            # Optimize using scipy
            from scipy.optimize import minimize
            
            result = minimize(
                objective,
                initial_params,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint},
                options={'maxiter': max_iter, 'ftol': tolerance}
            )
            
            if result.success:
                omega, alpha, beta = result.x
                
                # Compute final log-likelihood
                final_log_lik = -result.fun
                
                # Compute fitted variance series
                variance = cp.zeros(len(returns), dtype=cp.float64)
                variance[0] = cp.var(returns)
                for t in range(1, len(returns)):
                    variance[t] = omega + alpha * returns[t-1]**2 + beta * variance[t-1]
                
                return {
                    'omega': float(omega),
                    'alpha': float(alpha),
                    'beta': float(beta),
                    'log_likelihood': float(final_log_lik),
                    'variance': cp.asnumpy(variance),
                    'converged': True,
                    'iterations': result.nit
                }
            else:
                logger.warning(f"GARCH optimization failed: {result.message}")
                return {
                    'omega': None,
                    'alpha': None,
                    'beta': None,
                    'log_likelihood': None,
                    'variance': None,
                    'converged': False,
                    'iterations': result.nit
                }
                
        except Exception as e:
            self._critical_error(f"Error in GPU GARCH fitting: {e}")

    def fit_garch11(self, series: cudf.Series) -> Optional[Dict[str, Any]]:
        """
        Fit GARCH(1,1) model to a single time series (partition).
        """
        try:
            # Debug: Check input data type and shape
            self._log_info(f"GARCH input data type: {type(series)}, shape: {series.shape}")
            
            data = series.to_cupy()
            self._log_info(f"GARCH cupy data type: {type(data)}, shape: {data.shape}")

            if len(data) < 100:
                self._log_warn("Insufficient data for GARCH, skipping.", series_len=len(data))
                return None
            
            if len(data) > self.max_samples:
                data = data[-self.max_samples:]
                self._log_info("Truncated data for GARCH fitting.", samples=len(data))
            
            # Ensure data is float64 and handle any NaN values
            data = cp.asarray(data, dtype=cp.float64)
            data = cp.nan_to_num(data, nan=0.0)
            self._log_info(f"GARCH data after conversion: type={type(data)}, min={float(cp.min(data))}, max={float(cp.max(data))}")
            
            # Compute returns with proper type handling
            log_data = cp.log(cp.maximum(data, 1e-8))  # Avoid log(0)
            returns = cp.diff(log_data)
            self._log_info(f"GARCH returns after diff: type={type(returns)}, min={float(cp.min(returns))}, max={float(cp.max(returns))}")
            
            # Remove any remaining NaN or infinite values
            returns = returns[cp.isfinite(returns)]
            self._log_info(f"GARCH returns after filtering: type={type(returns)}, len={len(returns)}")
            
            if len(returns) < 50:
                self._log_warn("Insufficient valid returns for GARCH, skipping.", valid_returns=len(returns))
                return None
            
            # Use more conservative initial parameters without strict constraints
            returns_var = float(cp.var(returns))
            initial_params = np.array([returns_var * 0.01, 0.05, 0.85], dtype=np.float64)
            bounds = [(1e-8, None), (0.0, 1.0), (0.0, 1.0)]
            
            self._log_info(f"GARCH initial params: {initial_params}, returns_var: {returns_var}")
            
            # Convert to numpy for scipy compatibility
            returns_np = cp.asnumpy(returns).astype(np.float64)
            self._log_info(f"GARCH returns_np: type={type(returns_np)}, shape={returns_np.shape}, min={float(np.min(returns_np))}, max={float(np.max(returns_np))}")
            
            result = minimize(
                fun=self._log_likelihood_gpu,
                x0=initial_params,
                args=(returns_np,),
                method='L-BFGS-B',  # Use L-BFGS-B which is more robust
                bounds=bounds,
                options={'maxiter': int(self.max_iter), 'ftol': float(self.tolerance)}
            )

            if not result.success:
                self._log_warn("GARCH optimization failed.", message=result.message)
                return None

            # Return fitted parameters and diagnostics
            omega, alpha, beta = result.x
            persistence = alpha + beta
            log_likelihood = -result.fun
            n_obs = int(len(returns))  # Convert to int explicitly
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
            self._critical_error(f"Error during GARCH fitting: {e}")

    def _fit_on_partition(self, part: cudf.DataFrame) -> Dict[str, Any]:
        """
        Execute GARCH fitting inside GPU worker (without leaving cluster).
        
        Args:
            part: DataFrame partition with 'y_close' column
            
        Returns:
            Dictionary with GARCH parameters
        """
        close_series = part["y_close"]
        res = self.fit_garch11(close_series)
        # fit_garch11 now always returns a dict (either fitted or default values)
        return res

    def process(self, df: dask_cudf.DataFrame) -> dask_cudf.DataFrame:
        """
        Executes the GARCH modeling pipeline (Dask version).
        """
        self._log_info("Starting GARCH (Dask)...")

        # (optional) ensure temporal order
        # df = self._ensure_sorted(df, by="ts")

        # Bring series to single partition in one GPU worker
        one = self._single_partition(df, cols=["y_close"])

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
        Executes the comprehensive GARCH modeling pipeline (cuDF version).
        
        Implements the complete GARCH modeling pipeline as specified in the technical plan:
        - GARCH(1,1) parameter estimation
        - Conditional variance series estimation
        - Volatility statistics and autocorrelations
        - Residual statistics
        - Volatility forecasting
        
        Args:
            gdf: Input cuDF DataFrame
            
        Returns:
            DataFrame with comprehensive GARCH features
        """
        self._log_info("Starting comprehensive GARCH (cuDF)...")

        # Fit GARCH directly on cuDF DataFrame
        garch_result = self._fit_comprehensive_garch(gdf)
        
        if garch_result is None:
            # Return default values to maintain schema
            garch_result = self._get_default_garch_result()

        self._log_info("Comprehensive GARCH fitted successfully")

        # Add all GARCH features to DataFrame
        gdf = self._add_comprehensive_garch_features(gdf, garch_result)

        self._log_info("Comprehensive GARCH features attached.")
        return gdf
    
    def _fit_comprehensive_garch(self, df: cudf.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Fit comprehensive GARCH model with all additional features.
        """
        try:
            if 'y_ret_1m' not in df.columns:
                self._log_warn("y_ret_1m column not found for GARCH fitting")
                return None
            
            # Get returns data
            returns = df['y_ret_1m'].to_cupy()
            
            # Remove NaN values
            valid_mask = ~cp.isnan(returns)
            if cp.sum(valid_mask) < 100:
                self._log_warn("Insufficient valid returns for GARCH fitting")
                return None
            
            clean_returns = returns[valid_mask]
            
            # Fit GARCH model using GPU-accelerated method
            garch_result = self.fit_garch_gpu(clean_returns, max_iter=self.max_iter, tolerance=self.tolerance)
            
            if garch_result is None or not garch_result.get('converged', False):
                self._log_warn("GARCH fitting failed or did not converge")
                return None
            
            # Add comprehensive additional features
            comprehensive_result = self._add_comprehensive_garch_features_to_result(garch_result, clean_returns)
            
            return comprehensive_result
            
        except Exception as e:
            self._critical_error(f"Error in comprehensive GARCH fitting: {e}")
    
    def _add_comprehensive_garch_features_to_result(self, garch_result: Dict[str, Any], returns: cp.ndarray) -> Dict[str, Any]:
        """
        Add comprehensive GARCH features to the result dictionary.
        """
        try:
            # Extract basic GARCH parameters
            omega = garch_result['omega']
            alpha = garch_result['alpha']
            beta = garch_result['beta']
            variance_series = garch_result['variance']
            
            # 1. CONDITIONAL VARIANCE SERIES STATISTICS
            volatility_stats = self._calculate_volatility_statistics(variance_series)
            
            # 2. VOLATILITY AUTOCORRELATIONS
            volatility_autocorr = self._calculate_volatility_autocorrelations(variance_series)
            
            # 3. RESIDUAL STATISTICS
            residual_stats = self._calculate_residual_statistics(returns, variance_series)
            
            # 4. VOLATILITY FORECASTING
            volatility_forecast = self._calculate_volatility_forecast(omega, alpha, beta, variance_series)
            
            # 5. ADDITIONAL GARCH METRICS
            additional_metrics = self._calculate_additional_garch_metrics(omega, alpha, beta, returns, variance_series)
            
            # Combine all results
            comprehensive_result = {
                # Basic GARCH parameters
                'garch_omega': omega,
                'garch_alpha': alpha,
                'garch_beta': beta,
                'garch_persistence': alpha + beta,
                'garch_log_likelihood': garch_result['log_likelihood'],
                'garch_aic': garch_result.get('aic', np.nan),
                'garch_bic': garch_result.get('bic', np.nan),
                'garch_is_stationary': float(alpha + beta < 1.0),
                'garch_converged': garch_result['converged'],
                'garch_iterations': garch_result['iterations'],
                
                # Volatility statistics
                **volatility_stats,
                
                # Volatility autocorrelations
                **volatility_autocorr,
                
                # Residual statistics
                **residual_stats,
                
                # Volatility forecasting
                **volatility_forecast,
                
                # Additional metrics
                **additional_metrics
            }
            
            return comprehensive_result
            
        except Exception as e:
            self._critical_error(f"Error adding comprehensive GARCH features: {e}")
    
    def _calculate_volatility_statistics(self, variance_series: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive volatility statistics.
        """
        try:
            volatility = np.sqrt(variance_series)
            
            stats = {
                'garch_vol_mean': float(np.mean(volatility)),
                'garch_vol_std': float(np.std(volatility)),
                'garch_vol_skew': float(self._calculate_skewness(volatility)),
                'garch_vol_kurt': float(self._calculate_kurtosis(volatility)),
                'garch_vol_min': float(np.min(volatility)),
                'garch_vol_max': float(np.max(volatility)),
                'garch_vol_median': float(np.median(volatility)),
                'garch_vol_q25': float(np.percentile(volatility, 25)),
                'garch_vol_q75': float(np.percentile(volatility, 75)),
                'garch_vol_range': float(np.max(volatility) - np.min(volatility)),
                'garch_vol_cv': float(np.std(volatility) / np.mean(volatility)) if np.mean(volatility) > 0 else 0.0
            }
            
            return stats
            
        except Exception as e:
            self._critical_error(f"Error calculating volatility statistics: {e}")
    
    def _calculate_volatility_autocorrelations(self, variance_series: np.ndarray) -> Dict[str, float]:
        """
        Calculate volatility autocorrelations at different lags.
        """
        try:
            volatility = np.sqrt(variance_series)
            
            autocorr = {}
            lags = [1, 5, 10, 20, 50]
            
            for lag in lags:
                if len(volatility) > lag:
                    autocorr_value = self._calculate_autocorrelation_numpy(volatility, lag)
                    autocorr[f'garch_vol_autocorr_lag{lag}'] = autocorr_value
                else:
                    autocorr[f'garch_vol_autocorr_lag{lag}'] = np.nan
            
            return autocorr
            
        except Exception as e:
            self._critical_error(f"Error calculating volatility autocorrelations: {e}")
    
    def _calculate_residual_statistics(self, returns: cp.ndarray, variance_series: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive residual statistics.
        """
        try:
            # Calculate standardized residuals
            volatility = np.sqrt(variance_series)
            standardized_residuals = returns[1:] / volatility  # Skip first observation
            
            # Remove any infinite or NaN values
            valid_mask = np.isfinite(standardized_residuals)
            clean_residuals = standardized_residuals[valid_mask]
            
            if len(clean_residuals) == 0:
                return {f'garch_residual_{stat}': np.nan for stat in ['mean', 'std', 'skew', 'kurt', 'jarque_bera']}
            
            stats = {
                'garch_residual_mean': float(np.mean(clean_residuals)),
                'garch_residual_std': float(np.std(clean_residuals)),
                'garch_residual_skew': float(self._calculate_skewness(clean_residuals)),
                'garch_residual_kurt': float(self._calculate_kurtosis(clean_residuals)),
                'garch_residual_jarque_bera': float(self._calculate_jarque_bera_statistic(clean_residuals))
            }
            
            return stats
            
        except Exception as e:
            self._critical_error(f"Error calculating residual statistics: {e}")
    
    def _calculate_volatility_forecast(self, omega: float, alpha: float, beta: float, variance_series: np.ndarray) -> Dict[str, float]:
        """
        Calculate volatility forecasts at different horizons.
        """
        try:
            # Get the last variance value
            last_variance = variance_series[-1]
            
            # Calculate forecasts at different horizons
            forecasts = {}
            horizons = [1, 5, 10, 20]
            
            for h in horizons:
                # GARCH(1,1) forecast formula: E[σ²_{t+h}] = ω + (α + β)^h * (σ²_t - ω/(1-α-β))
                if alpha + beta < 1.0:
                    long_run_variance = omega / (1 - alpha - beta)
                    forecast_variance = long_run_variance + (alpha + beta)**h * (last_variance - long_run_variance)
                else:
                    # If not stationary, use simple extrapolation
                    forecast_variance = last_variance
                
                forecasts[f'garch_vol_forecast_h{h}'] = float(np.sqrt(forecast_variance))
            
            return forecasts
            
        except Exception as e:
            self._critical_error(f"Error calculating volatility forecast: {e}")
    
    def _calculate_additional_garch_metrics(self, omega: float, alpha: float, beta: float, 
                                          returns: cp.ndarray, variance_series: np.ndarray) -> Dict[str, float]:
        """
        Calculate additional GARCH metrics and diagnostics.
        """
        try:
            metrics = {}
            
            # Leverage effect (asymmetric response)
            negative_returns = returns[returns < 0]
            positive_returns = returns[returns > 0]
            
            if len(negative_returns) > 0 and len(positive_returns) > 0:
                neg_vol = float(cp.std(negative_returns))
                pos_vol = float(cp.std(positive_returns))
                leverage_effect = neg_vol / pos_vol if pos_vol > 0 else 1.0
                metrics['garch_leverage_effect'] = leverage_effect
            else:
                metrics['garch_leverage_effect'] = 1.0
            
            # Volatility clustering measure
            volatility = np.sqrt(variance_series)
            volatility_changes = np.diff(volatility)
            volatility_clustering = float(np.corrcoef(volatility[:-1], volatility[1:])[0, 1]) if len(volatility) > 1 else 0.0
            metrics['garch_volatility_clustering'] = volatility_clustering
            
            # Mean reversion speed
            if alpha + beta < 1.0:
                mean_reversion_speed = 1 - (alpha + beta)
                metrics['garch_mean_reversion_speed'] = float(mean_reversion_speed)
            else:
                metrics['garch_mean_reversion_speed'] = 0.0
            
            # Half-life of volatility shocks
            if alpha + beta < 1.0:
                half_life = np.log(0.5) / np.log(alpha + beta)
                metrics['garch_volatility_half_life'] = float(half_life)
            else:
                metrics['garch_volatility_half_life'] = np.inf
            
            return metrics
            
        except Exception as e:
            self._critical_error(f"Error calculating additional GARCH metrics: {e}")
    
    def _add_comprehensive_garch_features(self, df: cudf.DataFrame, garch_result: Dict[str, Any]) -> cudf.DataFrame:
        """
        Add all comprehensive GARCH features to the DataFrame.
        """
        try:
            # Add all GARCH features as columns
            for key, value in garch_result.items():
                df[key] = value
            
            return df
            
        except Exception as e:
            self._critical_error(f"Error adding GARCH features to DataFrame: {e}")
    
    def _get_default_garch_result(self) -> Dict[str, Any]:
        """
        Get default GARCH result with NaN values.
        """
        return {
            'garch_omega': np.nan, 'garch_alpha': np.nan, 'garch_beta': np.nan,
            'garch_persistence': np.nan, 'garch_log_likelihood': np.nan,
            'garch_aic': np.nan, 'garch_bic': np.nan, 'garch_is_stationary': np.nan,
            'garch_converged': False, 'garch_iterations': 0,
            'garch_vol_mean': np.nan, 'garch_vol_std': np.nan, 'garch_vol_skew': np.nan,
            'garch_vol_kurt': np.nan, 'garch_vol_min': np.nan, 'garch_vol_max': np.nan,
            'garch_vol_median': np.nan, 'garch_vol_q25': np.nan, 'garch_vol_q75': np.nan,
            'garch_vol_range': np.nan, 'garch_vol_cv': np.nan,
            'garch_vol_autocorr_lag1': np.nan, 'garch_vol_autocorr_lag5': np.nan,
            'garch_vol_autocorr_lag10': np.nan, 'garch_vol_autocorr_lag20': np.nan,
            'garch_vol_autocorr_lag50': np.nan,
            'garch_residual_mean': np.nan, 'garch_residual_std': np.nan,
            'garch_residual_skew': np.nan, 'garch_residual_kurt': np.nan,
            'garch_residual_jarque_bera': np.nan,
            'garch_vol_forecast_h1': np.nan, 'garch_vol_forecast_h5': np.nan,
            'garch_vol_forecast_h10': np.nan, 'garch_vol_forecast_h20': np.nan,
            'garch_leverage_effect': np.nan, 'garch_volatility_clustering': np.nan,
            'garch_mean_reversion_speed': np.nan, 'garch_volatility_half_life': np.nan
        }
    
    # Helper methods for statistical calculations
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 3))
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 4))
    
    def _calculate_autocorrelation_numpy(self, data: np.ndarray, lag: int) -> float:
        """Calculate autocorrelation at given lag."""
        if lag >= len(data):
            return 0.0
        
        data_lagged = data[lag:]
        data_original = data[:-lag]
        
        mean_original = np.mean(data_original)
        mean_lagged = np.mean(data_lagged)
        
        numerator = np.sum((data_original - mean_original) * (data_lagged - mean_lagged))
        denominator = np.sqrt(np.sum((data_original - mean_original)**2) * np.sum((data_lagged - mean_lagged)**2))
        
        if denominator > 1e-9:
            return float(numerator / denominator)
        else:
            return 0.0
    
    def _calculate_jarque_bera_statistic(self, data: np.ndarray) -> float:
        """Calculate Jarque-Bera test statistic."""
        n = len(data)
        skewness = self._calculate_skewness(data)
        kurtosis = self._calculate_kurtosis(data)
        
        jb_stat = n * (skewness**2 / 6 + (kurtosis - 3)**2 / 24)
        return float(jb_stat)
