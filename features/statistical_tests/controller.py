"""
Statistical Tests Controller

This module contains the main controller class that orchestrates all statistical tests
and feature selection operations, providing a unified interface to the modular components.
"""

import logging
import numpy as np
import cupy as cp
import cudf
import dask_cudf
import pandas as pd
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import re

from features.base_engine import BaseFeatureEngine
from .adf_tests import ADFTests
from .distance_correlation import DistanceCorrelation
from .feature_selection import FeatureSelection
from .statistical_analysis import StatisticalAnalysis
from .utils import _free_gpu_memory_worker, _tail_k, _tail_k_to_pandas
from utils.logging_utils import get_logger, info_event, warn_event, error_event, Events
from utils import log_context

logger = get_logger(__name__, "features.StatisticalTests")


class StatisticalTests(BaseFeatureEngine):
    """
    Main controller class for statistical tests and feature selection.
    
    This class orchestrates all statistical testing operations including:
    - ADF (Augmented Dickey-Fuller) stationarity tests
    - Distance correlation analysis
    - Feature selection (VIF, MI, clustering)
    - General statistical analysis
    """

    def __init__(self, settings, client):
        """Initialize StatisticalTests with configuration parameters."""
        super().__init__(settings, client)  # Initialize base class
        
        # Store settings for consistency with other engines
        self.settings = settings
        
        # Initialize component modules
        self.adf_tests = ADFTests(logger)
        self.distance_correlation = DistanceCorrelation(logger)
        self.feature_selection = FeatureSelection(logger)
        self.statistical_analysis = StatisticalAnalysis(logger)
        
        # Load configuration parameters
        self._load_configuration()
        
        # Initialize logging context
        self._log_info("StatisticalTests initialized", 
                      dcor_max_samples=self.dcor_max_samples,
                      dcor_tile_size=self.dcor_tile_size,
                      vif_threshold=self.vif_threshold,
                      mi_threshold=self.mi_threshold)

    def _load_configuration(self):
        """Load configuration parameters from settings."""
        try:
            # Configurable parameters with safe fallbacks
            self.dcor_max_samples = getattr(settings.features, 'distance_corr_max_samples', 10000)
        except Exception:
            self.dcor_max_samples = 10000  # Default fallback
            
        try:
            from config.unified_config import get_unified_config
            uc = get_unified_config()  # Load unified configuration
            
            # Distance correlation parameters
            self.dcor_tile_size = getattr(uc.features, 'distance_corr_tile_size', 2048)
            self.selection_target_column = getattr(uc.features, 'selection_target_column', 'y_ret_1m')
            self.dcor_top_k = int(getattr(uc.features, 'dcor_top_k', 50))
            self.dcor_include_permutation = bool(getattr(uc.features, 'dcor_include_permutation', False))
            self.dcor_permutations = int(getattr(uc.features, 'dcor_permutations', 0))
            
            # Feature selection parameters
            self.selection_max_rows = int(getattr(uc.features, 'selection_max_rows', 100000))
            self.vif_threshold = float(getattr(uc.features, 'vif_threshold', 5.0))
            self.mi_threshold = float(getattr(uc.features, 'mi_threshold', 0.3))
            self.stage3_top_n = int(getattr(uc.features, 'stage3_top_n', 50))
            self.dcor_min_threshold = float(getattr(uc.features, 'dcor_min_threshold', 0.0))
            self.dcor_min_percentile = float(getattr(uc.features, 'dcor_min_percentile', 0.0))
            self.stage1_top_n = int(getattr(uc.features, 'stage1_top_n', 0))
            
            # Additional Stage 1 gates
            self.correlation_min_threshold = float(getattr(uc.features, 'correlation_min_threshold', 0.0))
            self.pvalue_max_alpha = float(getattr(uc.features, 'pvalue_max_alpha', 1.0))
            self.dcor_fast_1d_enabled = bool(getattr(uc.features, 'dcor_fast_1d_enabled', False))
            self.dcor_fast_1d_bins = int(getattr(uc.features, 'dcor_fast_1d_bins', 2048))
            self.dcor_permutation_top_k = int(getattr(uc.features, 'dcor_permutation_top_k', 0))
            self.dcor_pvalue_alpha = float(getattr(uc.features, 'dcor_pvalue_alpha', 0.05))
            
            # Rolling dCor params
            self.stage1_rolling_enabled = bool(getattr(uc.features, 'stage1_rolling_enabled', False))
            self.stage1_rolling_window = int(getattr(uc.features, 'stage1_rolling_window', 2000))
            self.stage1_rolling_step = int(getattr(uc.features, 'stage1_rolling_step', 500))
            self.stage1_rolling_min_periods = int(getattr(uc.features, 'stage1_rolling_min_periods', 200))
            self.stage1_rolling_min_valid_pairs = int(getattr(uc.features, 'stage1_rolling_min_valid_pairs', self.stage1_rolling_min_periods))
            self.stage1_rolling_max_rows = int(getattr(uc.features, 'stage1_rolling_max_rows', 20000))
            self.stage1_rolling_max_windows = int(getattr(uc.features, 'stage1_rolling_max_windows', 20))
            self.stage1_agg = str(getattr(uc.features, 'stage1_agg', 'median')).lower()
            self.stage1_use_rolling_scores = bool(getattr(uc.features, 'stage1_use_rolling_scores', True))
            
            # Dashboard per-feature visibility
            self.stage1_dashboard_per_feature = bool(getattr(uc.features, 'stage1_dashboard_per_feature', False))
            
            # Gating and leakage control
            self.selection_target_columns = list(getattr(uc.features, 'selection_target_columns', []))
            self.dataset_target_columns = list(getattr(uc.features, 'dataset_target_columns', []))
            self.feature_allowlist = list(getattr(uc.features, 'feature_allowlist', []))
            self.feature_allow_prefixes = list(getattr(uc.features, 'feature_allow_prefixes', []))
            self.feature_denylist = list(getattr(uc.features, 'feature_denylist', []))
            self.feature_deny_prefixes = list(getattr(uc.features, 'feature_deny_prefixes', ['y_ret_fwd_']))
            self.feature_deny_regex = list(getattr(uc.features, 'feature_deny_regex', []))
            self.metrics_prefixes = list(getattr(uc.features, 'metrics_prefixes', ['dcor_', 'dcor_roll_', 'dcor_pvalue_', 'stage1_', 'cpcv_']))
            self.dataset_target_prefixes = list(getattr(uc.features, 'dataset_target_prefixes', []))
            
            # Protection lists (always keep)
            self.always_keep_features = list(getattr(uc.features, 'always_keep_features', []))
            self.always_keep_prefixes = list(getattr(uc.features, 'always_keep_prefixes', []))
            
            # Visibility/debug flags
            self.stage1_broadcast_scores = bool(getattr(uc.features, 'stage1_broadcast_scores', False))
            self.stage1_broadcast_rolling = bool(getattr(uc.features, 'stage1_broadcast_rolling', False))
            self.debug_write_artifacts = bool(getattr(uc.features, 'debug_write_artifacts', True))
            self.artifacts_dir = str(getattr(uc.features, 'artifacts_dir', 'artifacts'))
            
            # Stage 3 LightGBM params
            self.stage3_task = str(getattr(uc.features, 'stage3_task', 'auto'))
            self.stage3_random_state = int(getattr(uc.features, 'stage3_random_state', 42))
            self.stage3_lgbm_enabled = bool(getattr(uc.features, 'stage3_lgbm_enabled', True))
            self.stage3_lgbm_num_leaves = int(getattr(uc.features, 'stage3_lgbm_num_leaves', 31))
            self.stage3_lgbm_max_depth = int(getattr(uc.features, 'stage3_lgbm_max_depth', -1))
            self.stage3_lgbm_n_estimators = int(getattr(uc.features, 'stage3_lgbm_n_estimators', 200))
            self.stage3_lgbm_learning_rate = float(getattr(uc.features, 'stage3_lgbm_learning_rate', 0.05))
            self.stage3_lgbm_feature_fraction = float(getattr(uc.features, 'stage3_lgbm_feature_fraction', 0.8))
            self.stage3_lgbm_bagging_fraction = float(getattr(uc.features, 'stage3_lgbm_bagging_fraction', 0.8))
            self.stage3_lgbm_bagging_freq = int(getattr(uc.features, 'stage3_lgbm_bagging_freq', 0))
            self.stage3_lgbm_early_stopping_rounds = int(getattr(uc.features, 'stage3_lgbm_early_stopping_rounds', 0))
            
            # MI clustering params (Stage 2 scalable)
            self.mi_cluster_enabled = bool(getattr(uc.features, 'mi_cluster_enabled', True))
            self.mi_cluster_method = str(getattr(uc.features, 'mi_cluster_method', 'agglo'))
            self.mi_cluster_threshold = float(getattr(uc.features, 'mi_cluster_threshold', 0.3))
            self.mi_max_candidates = int(getattr(uc.features, 'mi_max_candidates', 400))
            self.mi_chunk_size = int(getattr(uc.features, 'mi_chunk_size', 128))
            
        except Exception:
            # Fallback defaults when unified config is not available
            self.dcor_tile_size = 2048
            self.selection_target_column = 'y_ret_1m'
            self.dcor_top_k = 50
            self.dcor_include_permutation = False
            self.dcor_permutations = 0
            
            # Gating defaults when unified config missing
            self.selection_target_columns = []
            self.dataset_target_columns = []
            self.feature_allowlist = []
            self.feature_allow_prefixes = []
            self.feature_denylist = []
            self.feature_deny_prefixes = ['y_ret_fwd_']
            self.feature_deny_regex = []
            self.metrics_prefixes = ['dcor_', 'dcor_roll_', 'dcor_pvalue_', 'stage1_', 'cpcv_']
            self.dataset_target_prefixes = []
            self.selection_max_rows = 100000
            self.vif_threshold = 5.0
            self.mi_threshold = 0.3
            self.stage3_top_n = 50
            
            # Rolling dCor defaults when unified config missing
            self.stage1_rolling_enabled = False
            self.stage1_rolling_window = 2000
            self.stage1_rolling_step = 500
            self.stage1_rolling_min_periods = 200
            self.stage1_rolling_min_valid_pairs = self.stage1_rolling_min_periods
            self.stage1_rolling_max_rows = 20000
            self.stage1_rolling_max_windows = 20
            self.stage1_agg = 'median'
            self.stage1_use_rolling_scores = True
            self.stage1_dashboard_per_feature = False

    def _log_info(self, message: str, **kwargs):
        """Log info message with optional context."""
        logger.info(f"StatisticalTests: {message}", extra=kwargs)
    
    def _log_warn(self, message: str, **kwargs):
        """Log warning message with optional context."""
        logger.warning(f"StatisticalTests: {message}", extra=kwargs)
    
    def _log_error(self, message: str, **kwargs):
        """Log error message with optional context."""
        logger.error(f"StatisticalTests: {message}", extra=kwargs)
    
    def _critical_error(self, message: str, **kwargs):
        """Log critical error and raise exception."""
        self._log_error(message, **kwargs)
        raise RuntimeError(f"StatisticalTests Critical Error: {message}")

    def process_cudf(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """
        Process cuDF DataFrame with comprehensive statistical tests.
        
        Implements the complete statistical tests pipeline:
        - ADF tests in batch for all frac_diff_* series
        - Distance correlation with significance testing
        - Stationarity flags and comprehensive statistics
        
        Args:
            df: Input cuDF DataFrame
            
        Returns:
            DataFrame with statistical test results
        """
        try:
            self._log_info("Starting comprehensive statistical tests pipeline...")
            
            # 1. ADF TESTS IN BATCH
            df = self.adf_tests._apply_comprehensive_adf_tests(df)
            
            # 2. DISTANCE CORRELATION TESTS
            df = self.distance_correlation._apply_comprehensive_distance_correlation(df)
            
            # 3. ADDITIONAL STATISTICAL TESTS
            df = self.statistical_analysis.apply_comprehensive_statistical_analysis(df)
            
            self._log_info("Comprehensive statistical tests pipeline completed successfully")
            return df
            
        except Exception as e:
            self._critical_error(f"Error in comprehensive statistical tests pipeline: {e}")

    def process(self, df: dask_cudf.DataFrame) -> dask_cudf.DataFrame:
        """
        Executes the statistical tests pipeline (Dask version).
        
        Args:
            df: Input Dask cuDF DataFrame
            
        Returns:
            DataFrame with statistical test results
        """
        self._log_info("Starting StatisticalTests (Dask)...")

        # Validate primary target; support computing log-forward returns if configured
        try:
            primary_target = getattr(self, 'selection_target_column', None)
        except Exception:
            primary_target = None
            
        if primary_target:
            try:
                if primary_target not in df.columns:
                    # If target looks like y_logret_fwd_{Nm} and y_close exists, compute it
                    m = re.match(r"^y_logret_fwd_(\d+)m$", str(primary_target))
                    if m and ('y_close' in df.columns):
                        horizon = int(m.group(1))
                        self._log_info("Primary target missing; computing log-forward return from y_close",
                                      target=primary_target, horizon=horizon)
                        meta = cudf.DataFrame({primary_target: cudf.Series([], dtype='f4')})
                        df = df.assign(**{
                            primary_target: df[['y_close']].map_partitions(
                                self._compute_forward_log_return_partition,
                                'y_close',
                                horizon,
                                primary_target,
                                meta=meta
                            )[primary_target]
                        })
                    else:
                        # Strict error otherwise
                        try:
                            sample_cols = list(df.head(50).columns)
                        except Exception:
                            sample_cols = []
                        self._critical_error(
                            "Target column not found for dCor ranking",
                            target=primary_target,
                            hint="Check config.features.selection_target_column and dataset labeling",
                            sample_schema=sample_cols[:20]
                        )
            except Exception:
                # If columns access fails unexpectedly, keep going and let later code raise
                pass

        # Multi-target sweep (if configured): run selection per target and persist comparison
        try:
            mt_list = list(getattr(self, 'selection_target_columns', []))
        except Exception:
            mt_list = []
            
        if mt_list:
            self._log_info("[MT] Multi-target sweep enabled", targets=mt_list)
            results: Dict[str, Any] = {}
            ccy = self._mt_currency_pair(df)
            for tgt in mt_list:
                res = self._mt_run_for_target(df, tgt)
                results[tgt] = res
            # Persist comparison report per currency pair
            try:
                out_root = Path(getattr(self.settings.output, 'output_path', './output'))
                out_dir = out_root / ccy / 'targets'
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / 'comparison.json', 'w') as f:
                    json.dump(results, f, indent=2)
                self._log_info("[MT] Comparison persisted", path=str(out_dir / 'comparison.json'))
            except Exception as e:
                self._log_warn("[MT] Persist comparison failed", error=str(e))

        # --- ADF rolling window on fractional difference columns ---
        adf_cols = [c for c in df.columns if "frac_diff" in c]

        for col in adf_cols:
            self._log_info(f"ADF rolling on '{col}'...")
            out = f"adf_stat_{col.split('_')[-1]}"
            df[out] = df[col].map_partitions(
                self.adf_tests._apply_adf_rolling,
                252,  # Window size (1 year of trading days)
                200,  # Minimum periods
                meta=(out, "f8"),
            )

        # --- Distance Correlation Analysis ---
        if primary_target and primary_target in df.columns:
            self._log_info("Starting distance correlation analysis...")
            
            # Find candidate features
            candidate_features = self._find_candidate_features(df)
            
            if candidate_features:
                self._log_info(f"Found {len(candidate_features)} candidate features for distance correlation")
                
                # Apply distance correlation analysis
                df = self._apply_distance_correlation_analysis(df, primary_target, candidate_features)
            else:
                self._log_warn("No candidate features found for distance correlation analysis")

        # --- Feature Selection Pipeline ---
        if primary_target and primary_target in df.columns:
            self._log_info("Starting feature selection pipeline...")
            
            # Apply feature selection
            selection_results = self._apply_feature_selection_pipeline(df, primary_target)
            
            if selection_results:
                self._log_info("Feature selection pipeline completed", 
                              final_count=len(selection_results.get('stage3_final_selected', [])))

        # --- Additional Statistical Analysis ---
        self._log_info("Applying additional statistical analysis...")
        df = df.map_partitions(self.statistical_analysis.apply_comprehensive_statistical_analysis)

        self._log_info("StatisticalTests (Dask) completed successfully")
        return df

    def _find_candidate_features(self, df: dask_cudf.DataFrame) -> List[str]:
        """Find candidate features for analysis."""
        try:
            # Get sample of columns
            sample_df = df.head(100)
            all_columns = list(sample_df.columns)
            
            # Filter candidate features
            candidates = []
            for col in all_columns:
                # Skip target columns and metrics
                if (col.startswith(('y_ret_fwd_', 'dcor_', 'adf_', 'stage1_', 'cpcv_')) or
                    col in self.feature_denylist or
                    any(col.startswith(prefix) for prefix in self.feature_deny_prefixes)):
                    continue
                
                # Include frac_diff columns and other relevant features
                if ('frac_diff' in col or 
                    col.startswith(('y_', 'x_')) or
                    any(col.startswith(prefix) for prefix in self.feature_allow_prefixes)):
                    candidates.append(col)
            
            return candidates
            
        except Exception as e:
            self._log_warn(f"Error finding candidate features: {e}")
            return []

    def _apply_distance_correlation_analysis(self, df: dask_cudf.DataFrame, target: str, candidates: List[str]) -> dask_cudf.DataFrame:
        """Apply distance correlation analysis to the DataFrame."""
        try:
            # Apply distance correlation computation
            dcor_results = df.map_partitions(
                self.distance_correlation._dcor_partition_gpu,
                target,
                candidates,
                self.dcor_max_samples,
                self.dcor_tile_size,
                meta=cudf.DataFrame([{f"dcor_{c}": float('nan') for c in candidates}])
            )
            
            # Merge results back to main DataFrame
            for col in candidates:
                dcor_col = f"dcor_{col}"
                if dcor_col in dcor_results.columns:
                    df[dcor_col] = dcor_results[dcor_col]
            
            return df
            
        except Exception as e:
            self._log_warn(f"Error in distance correlation analysis: {e}")
            return df

    def _apply_feature_selection_pipeline(self, df: dask_cudf.DataFrame, target: str) -> Dict[str, Any]:
        """Apply the complete feature selection pipeline."""
        try:
            # Get candidate features
            candidates = self._find_candidate_features(df)
            
            if not candidates:
                return {}
            
            # Apply feature selection to a sample of the data
            sample_df = df.head(self.selection_max_rows)
            
            # Get distance correlation scores
            dcor_scores = {}
            for col in candidates:
                dcor_col = f"dcor_{col}"
                if dcor_col in sample_df.columns:
                    dcor_scores[col] = float(sample_df[dcor_col].iloc[0]) if len(sample_df) > 0 else 0.0
            
            # Apply feature selection pipeline
            selection_results = self.feature_selection.apply_feature_selection_pipeline(
                sample_df, target, candidates, dcor_scores
            )
            
            return selection_results
            
        except Exception as e:
            self._log_warn(f"Error in feature selection pipeline: {e}")
            return {}

    def _compute_forward_log_return_partition(self, pdf: cudf.DataFrame, price_col: str, horizon: int, out_col: str) -> cudf.DataFrame:
        """Compute forward log returns for a partition."""
        try:
            if price_col not in pdf.columns:
                return cudf.DataFrame({out_col: cudf.Series([], dtype='f4')})
            
            prices = pdf[price_col].values
            n = len(prices)
            log_returns = cp.full(n, cp.nan, dtype=cp.float32)
            
            for i in range(n - horizon):
                if cp.isfinite(prices[i]) and cp.isfinite(prices[i + horizon]):
                    log_returns[i] = cp.log(prices[i + horizon] / prices[i])
            
            return cudf.DataFrame({out_col: cudf.Series(log_returns, index=pdf.index)})
        except Exception:
            return cudf.DataFrame({out_col: cudf.Series([], dtype='f4')})

    def _mt_currency_pair(self, df: dask_cudf.DataFrame) -> str:
        """Extract currency pair identifier from DataFrame."""
        try:
            # Try to get currency pair from metadata or column names
            sample_cols = list(df.head(10).columns)
            for col in sample_cols:
                if 'currency' in col.lower() or 'pair' in col.lower():
                    return str(col)
            return 'unknown'
        except Exception:
            return 'unknown'

    def _mt_run_for_target(self, df: dask_cudf.DataFrame, target: str) -> Dict[str, Any]:
        """Run feature selection for a specific target."""
        try:
            self._log_info(f"[MT] Running selection for target: {target}")
            
            # Apply feature selection for this target
            selection_results = self._apply_feature_selection_pipeline(df, target)
            
            return {
                'target': target,
                'selection_results': selection_results,
                'timestamp': str(pd.Timestamp.now())
            }
            
        except Exception as e:
            self._log_warn(f"[MT] Error running selection for target {target}: {e}")
            return {
                'target': target,
                'error': str(e),
                'timestamp': str(pd.Timestamp.now())
            }
