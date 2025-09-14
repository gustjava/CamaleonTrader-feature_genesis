"""
Statistical Tests Controller

This module contains the main controller class that orchestrates all statistical tests
and feature selection operations, providing a unified interface to the modular components.
"""

import logging
import uuid
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
            self.dcor_max_samples = getattr(self.settings.features, 'distance_corr_max_samples', 10000)
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
            # Feature toggles
            self.enable_adf_rolling = bool(getattr(uc.features, 'enable_adf_rolling', False))
            # JB removed: no configuration loaded
            
            # Feature selection parameters
            self.selection_max_rows = int(getattr(uc.features, 'selection_max_rows', 100000))
            self.vif_threshold = float(getattr(uc.features, 'vif_threshold', 5.0))
            self.mi_threshold = float(getattr(uc.features, 'mi_threshold', 0.3))
            self.stage3_top_n = int(getattr(uc.features, 'stage3_top_n', 50))
            self.dcor_min_threshold = float(getattr(uc.features, 'dcor_min_threshold', 0.05))
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
            # Stage 3 CatBoost (explicit) with fallback to LGBM fields
            self.stage3_catboost_iterations = int(getattr(uc.features, 'stage3_catboost_iterations', self.stage3_lgbm_n_estimators))
            self.stage3_catboost_learning_rate = float(getattr(uc.features, 'stage3_catboost_learning_rate', self.stage3_lgbm_learning_rate))
            # If lgbm_max_depth == -1 (auto), use 6 as CatBoost default depth
            _cb_depth_default = 6 if int(self.stage3_lgbm_max_depth) == -1 else int(self.stage3_lgbm_max_depth)
            self.stage3_catboost_depth = int(getattr(uc.features, 'stage3_catboost_depth', _cb_depth_default))
            self.stage3_catboost_devices = str(getattr(uc.features, 'stage3_catboost_devices', '0'))
            self.stage3_catboost_task_type = str(getattr(uc.features, 'stage3_catboost_task_type', 'GPU'))
            self.stage3_catboost_thread_count = int(getattr(uc.features, 'stage3_catboost_thread_count', 1))
            self.stage3_catboost_loss_regression = str(getattr(uc.features, 'stage3_catboost_loss_regression', 'RMSE'))
            self.stage3_catboost_loss_classification = str(getattr(uc.features, 'stage3_catboost_loss_classification', 'Logloss'))
            # Temporal CV / early stopping
            self.stage3_cv_splits = int(getattr(uc.features, 'stage3_cv_splits', 3))
            self.stage3_cv_min_train = int(getattr(uc.features, 'stage3_cv_min_train', 200))
            self.stage3_catboost_early_stopping_rounds = int(getattr(uc.features, 'stage3_catboost_early_stopping_rounds', self.stage3_lgbm_early_stopping_rounds))
            self.stage3_catboost_use_full_dataset = bool(getattr(uc.features, 'stage3_catboost_use_full_dataset', False))
            # Additional Stage 3 knobs (not always present in dataclass; read via getattr)
            self.stage3_importance_threshold = getattr(uc.features, 'stage3_importance_threshold', 'median')
            self.stage3_stratified_sampling = bool(getattr(uc.features, 'stage3_stratified_sampling', True))
            self.stage3_classification_max_classes = int(getattr(uc.features, 'stage3_classification_max_classes', 10))

            # CPCV controls
            self.cpcv_enabled = bool(getattr(uc.features, 'cpcv_enabled', True))
            self.cpcv_n_groups = int(getattr(uc.features, 'cpcv_n_groups', 6))
            self.cpcv_k_leave_out = int(getattr(uc.features, 'cpcv_k_leave_out', 2))
            self.cpcv_purge = int(getattr(uc.features, 'cpcv_purge', 0))
            self.cpcv_embargo = int(getattr(uc.features, 'cpcv_embargo', 0))

            # Propagate key Stage 3 and CPCV configs to FeatureSelection instance
            try:
                fs = self.feature_selection
                for _name in [
                    'selection_max_rows',
                    'stage3_task', 'stage3_random_state',
                    'stage3_catboost_iterations', 'stage3_catboost_learning_rate', 'stage3_catboost_depth',
                    'stage3_catboost_devices', 'stage3_catboost_task_type', 'stage3_catboost_thread_count',
                    'stage3_catboost_loss_regression', 'stage3_catboost_use_full_dataset',
                    'stage3_cv_splits', 'stage3_cv_min_train', 'stage3_catboost_early_stopping_rounds',
                    'stage3_importance_threshold', 'stage3_stratified_sampling', 'stage3_classification_max_classes',
                    'cpcv_enabled', 'cpcv_n_groups', 'cpcv_k_leave_out', 'cpcv_purge', 'cpcv_embargo',
                    # GPU policy flags
                    'force_gpu_usage', 'gpu_fallback_enabled',
                ]:
                    try:
                        setattr(fs, _name, getattr(self, _name))
                    except Exception:
                        pass
            except Exception:
                # Best-effort propagation only
                pass
            
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
            # JB removed: no defaults
            # Feature toggles
            self.enable_adf_rolling = False

    def _log_info(self, message: str, **kwargs):
        """Log info message with optional context and phase/session markers."""
        sid = getattr(self, '_sid', None)
        stage = getattr(self, '_stage', None)
        prefix = []
        if sid:
            prefix.append(f"sid={sid}")
        if stage:
            prefix.append(f"stage={stage}")
        pfx = ("[" + "][".join(prefix) + "] ") if prefix else ""
        logger.info(f"StatisticalTests: {pfx}{message}", extra=kwargs)
    
    def _log_warn(self, message: str, **kwargs):
        """Log warning message with optional context and phase/session markers."""
        sid = getattr(self, '_sid', None)
        stage = getattr(self, '_stage', None)
        prefix = []
        if sid:
            prefix.append(f"sid={sid}")
        if stage:
            prefix.append(f"stage={stage}")
        pfx = ("[" + "][".join(prefix) + "] ") if prefix else ""
        logger.warning(f"StatisticalTests: {pfx}{message}", extra=kwargs)
    
    def _log_error(self, message: str, **kwargs):
        """Log error message with optional context and phase/session markers."""
        sid = getattr(self, '_sid', None)
        stage = getattr(self, '_stage', None)
        prefix = []
        if sid:
            prefix.append(f"sid={sid}")
        if stage:
            prefix.append(f"stage={stage}")
        pfx = ("[" + "][".join(prefix) + "] ") if prefix else ""
        logger.error(f"StatisticalTests: {pfx}{message}", extra=kwargs)
    
    def _log_debug(self, message: str, **kwargs):
        """Log debug message with optional context and phase/session markers."""
        sid = getattr(self, '_sid', None)
        stage = getattr(self, '_stage', None)
        prefix = []
        if sid:
            prefix.append(f"sid={sid}")
        if stage:
            prefix.append(f"stage={stage}")
        pfx = ("[" + "][".join(prefix) + "] ") if prefix else ""
        logger.debug(f"StatisticalTests: {pfx}{message}", extra=kwargs)

    def _set_stage(self, stage: str):
        """Set current stage label for logging context."""
        try:
            self._stage = str(stage)
        except Exception:
            self._stage = stage
    
    def _critical_error(self, message: str, **kwargs):
        """Log critical error and raise exception."""
        self._log_error(message, **kwargs)
        raise RuntimeError(f"StatisticalTests Critical Error: {message}")

    def _persist_and_wait(self, df: dask_cudf.DataFrame, stage: str, timeout_s: int = 600) -> dask_cudf.DataFrame:
        """Persist the Dask DataFrame and wait with timeout, logging stage progress.

        Intended to provide mid-stage visibility for long-running Statistical Tests.
        Safe no-op if client is unavailable; always returns a (possibly) persisted df.
        """
        try:
            self._log_info(f"Persist start: statistical_tests:{stage}")
            try:
                # Try to use dask.persist for better graph control when debug dashboard is enabled
                import dask
                debug_dash = bool(getattr(self, 'debug_write_artifacts', True))  # reuse flag as proxy
                if debug_dash:
                    df, = dask.persist(df, optimize_graph=False)
                else:
                    df = df.persist()
            except Exception:
                df = df.persist()

            self._log_info(f"Persist returned: statistical_tests:{stage}")

            # Wait with timeout (best-effort); continue even if timeout
            try:
                from dask.distributed import wait as _wait
                self._log_info(f"Waiting for statistical_tests:{stage} to complete (timeout: {timeout_s}s)...")
                _wait(df, timeout=timeout_s)
                self._log_info(f"statistical_tests:{stage} computation completed successfully")
            except Exception as wait_err:
                self._log_warn(f"Wait timeout or error for statistical_tests:{stage}: {wait_err}")
                self._log_info(f"Continuing without waiting - data is still persisted for statistical_tests:{stage}")

            # Optional: free GPU pools on all workers to reduce memory pressure
            try:
                if getattr(self, 'client', None):
                    self.client.run(lambda: (__import__('gc').collect(), __import__('cupy').get_default_memory_pool().free_all_blocks()))
            except Exception:
                pass
            return df
        except Exception as e:
            self._log_warn(f"persist_and_wait failed for stage {stage}: {e}")
            return df

    def _sample_head_across_partitions(self, df: dask_cudf.DataFrame, n: int, max_parts: int = 16) -> cudf.DataFrame:
        """Collect up to n rows by sampling the head of several partitions safely.

        Avoids Dask head() alignment issues with non-unique indices by computing
        partition heads separately and concatenating on GPU with ignore_index.
        """
        try:
            nparts = int(getattr(df, 'npartitions', 1))
            k = max(1, min(int(max_parts), nparts))
            per = max(1, int(np.ceil(float(n) / float(k))))
            frames = []
            for i in range(k):
                try:
                    part_head = df.get_partition(i).head(per)
                    # If still lazy, compute to cuDF
                    if not isinstance(part_head, cudf.DataFrame):
                        part_head = part_head.compute()
                    frames.append(part_head)
                except Exception:
                    continue
            if not frames:
                return cudf.DataFrame()
            # Ensure all frames are cuDF
            cu_frames = []
            for g in frames:
                try:
                    if isinstance(g, cudf.DataFrame):
                        cu_frames.append(g)
                    else:
                        cu_frames.append(cudf.from_pandas(g))
                except Exception:
                    continue
            if not cu_frames:
                return cudf.DataFrame()
            sample = cudf.concat(cu_frames, ignore_index=True)
            if len(sample) > int(n):
                sample = sample.head(int(n))
            return sample
        except Exception as e:
            self._log_warn("sample_head_across_partitions failed; falling back to head", error=str(e))
            try:
                out = df.head(int(n))
                # Ensure cuDF
                if not isinstance(out, cudf.DataFrame):
                    out = cudf.from_pandas(out)
                return out
            except Exception:
                try:
                    # Last resort: first partition only
                    out = df.get_partition(0).head(int(n)).compute()
                    if not isinstance(out, cudf.DataFrame):
                        out = cudf.from_pandas(out)
                    return out
                except Exception:
                    return cudf.DataFrame()

    def _sample_tail_across_partitions(self, df: dask_cudf.DataFrame, n: int, max_parts: int = 16) -> cudf.DataFrame:
        """Collect up to n rows by sampling the tail of the last partitions safely.

        Similar to head sampling, but walks partitions from the end, getting tail(per)
        from each until reaching approximately n rows. Ensures cuDF output.
        """
        try:
            nparts = int(getattr(df, 'npartitions', 1))
        except Exception:
            nparts = 1
        try:
            k = max(1, min(int(max_parts), nparts))
            per = max(1, int(np.ceil(float(n) / float(k))))
            frames = []
            for idx in range(nparts - 1, max(-1, nparts - 1 - k), -1):
                try:
                    part_tail = df.get_partition(idx).tail(per)
                    if not isinstance(part_tail, cudf.DataFrame):
                        part_tail = part_tail.compute()
                    frames.append(part_tail)
                except Exception:
                    continue
            if not frames:
                return cudf.DataFrame()
            cu_frames = []
            for g in frames:
                try:
                    if isinstance(g, cudf.DataFrame):
                        cu_frames.append(g)
                    else:
                        cu_frames.append(cudf.from_pandas(g))
                except Exception:
                    continue
            if not cu_frames:
                return cudf.DataFrame()
            sample = cudf.concat(cu_frames, ignore_index=True)
            # Keep the last n rows of the concatenated sample
            if len(sample) > int(n):
                sample = sample.tail(int(n))
            return sample
        except Exception as e:
            self._log_warn("sample_tail_across_partitions failed; falling back to tail of first partition", error=str(e))
            try:
                out = df.get_partition(max(0, nparts - 1)).tail(int(n)).compute()
                if not isinstance(out, cudf.DataFrame):
                    out = cudf.from_pandas(out)
                return out
            except Exception:
                return cudf.DataFrame()

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
            try:
                self._sid = uuid.uuid4().hex[:8]
            except Exception:
                self._sid = None
            self._set_stage('init')
            
            # Extract currency pair information
            currency_pair = self._mt_currency_pair(df)
            currency_info = f" for {currency_pair}" if currency_pair else ""
            
            self._log_info(f"Starting comprehensive statistical tests pipeline{currency_info}...")
            self._log_info("Input DataFrame info", 
                          currency_pair=currency_pair,
                          rows=len(df), 
                          cols=len(df.columns),
                          memory_usage_mb=df.memory_usage(deep=True).sum() / 1024**2)
            
            # 1. ADF TESTS IN BATCH
            self._set_stage('adf')
            self._log_info(f"🔍 Starting ADF (Augmented Dickey-Fuller) stationarity tests{currency_info}...")
            cols_before = len(df.columns)
            df = self.adf_tests._apply_comprehensive_adf_tests(df)
            cols_after = len(df.columns)
            self._log_info("✅ ADF tests completed", 
                          currency_pair=currency_pair,
                          features_added=cols_after - cols_before,
                          total_features=cols_after)
            
            # JB removed: no rolling JB features
            
            # 2. DISTANCE CORRELATION TESTS
            self._set_stage('dcor')
            self._log_info(f"🔗 Starting Distance Correlation (dCor) analysis{currency_info}...")
            self._log_info("📊 PROCESSING: Distance Correlation - This is the main computational step")
            cols_before_dcor = len(df.columns)
            df = self.distance_correlation._apply_comprehensive_distance_correlation(df)
            cols_after_dcor = len(df.columns)
            self._log_info("✅ Distance Correlation analysis completed", 
                          currency_pair=currency_pair,
                          features_added=cols_after_dcor - cols_before_dcor,
                          total_features=cols_after_dcor)
            
            # 3. ADDITIONAL STATISTICAL TESTS
            self._set_stage('final_stats')
            self._log_info(f"📈 Starting additional statistical analysis{currency_info}...")
            self._log_info("📊 PROCESSING: Statistical Analysis - Computing summaries and correlations")
            cols_before_final = len(df.columns)
            df = self.statistical_analysis.apply_comprehensive_statistical_analysis(df)
            cols_after_final = len(df.columns)
            self._log_info("✅ Additional statistical analysis completed", 
                          currency_pair=currency_pair,
                          features_added=cols_after_final - cols_before_final,
                          total_features=cols_after_final)
            
            self._log_info("🎉 Comprehensive statistical tests pipeline completed successfully",
                          currency_pair=currency_pair,
                          total_features_created=cols_after_final - len(df.columns) + (cols_after_final - cols_before_final),
                          final_feature_count=cols_after_final)
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
        # Initialize session id and stage markers for this run
        try:
            self._sid = uuid.uuid4().hex[:8]
        except Exception:
            self._sid = None
        self._set_stage('init')
        self._log_info("Starting StatisticalTests (Dask)...")
        self._log_info("📊 PROCESSING: Dask Statistical Tests Pipeline - This is the main processing stage")

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
            except Exception as e:
                # If columns access fails unexpectedly, keep going and let later code raise
                self._log_error(f"Failed to access columns for dCor ranking: {e}")
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

    # --- ADF Stage ---
        from .stage_adf import run as run_adf
        self._set_stage('adf')
        self._log_info("Stage start: ADF rolling on frac_diff*")
        df = run_adf(self, df, window=252, min_periods=200)
        df = self._persist_and_wait(df, stage="adf", timeout_s=600)
        self._log_info("Stage end: ADF")

    # JB removed: no rolling JB features

        # --- Pre-selection Statistical Analysis (broadcast constants; no corr matrix) ---
        # (Pre-selection statistical broadcasting removed per request)

        # --- dCor Stage ---
        self._set_stage('dcor')
        if primary_target and primary_target in df.columns:
            from .stage_dcor import run as run_dcor
            self._log_info("Stage start: dCor analysis")
            candidate_features = self._find_candidate_features(df)
            if candidate_features:
                df = run_dcor(self, df, primary_target, candidate_features)
            else:
                self._log_warn("No candidate features found for dCor analysis")
            self._log_info("Stage end: dCor")

        # --- Selection Stage ---
        self._set_stage('selection')
        if primary_target and primary_target in df.columns:
            from .stage_selection import run as run_selection
            self._log_info("Stage start: Selection (VIF/MI/CatBoost)")
            selection_results = run_selection(self, df, primary_target)
            if selection_results:
                self._log_info("Stage end: Selection", final_count=len(selection_results.get('stage3_final_selected', [])))

        # --- Final Stats Stage ---
        # DISABLED: Final stats stage removed - unnecessary after feature selection
        self._log_info("Final Stats stage skipped - feature selection already completed")

        self._log_info("StatisticalTests (Dask) completed successfully")
        return df

    def _find_candidate_features(self, df: dask_cudf.DataFrame) -> List[str]:
        """Find candidate features for analysis."""
        try:
            # Use schema directly; avoid triggering compute via head()
            all_columns = list(df.columns)
            # Build a dtype map to filter non-numeric columns
            dtype_map = {}
            try:
                dts = getattr(df, 'dtypes', None)
                if dts is not None:
                    # dask-cuDF returns a pandas Series-like object
                    dtype_map = {str(k): str(v).lower() for k, v in dts.items()}
            except Exception:
                dtype_map = {}
            def _is_numeric_dt(dt: str) -> bool:
                s = (dt or '').lower()
                if not s:
                    # Unknown dtype in meta: treat as potentially numeric (avoid over-filtering)
                    return True
                bad = ('object', 'str', 'string', 'category', 'datetime', 'timedelta', 'date')
                if any(b in s for b in bad):
                    return False
                ok = (
                    'int', 'uint', 'float', 'double', 'bool',  # common dtype names
                    'f4', 'f8', 'i4', 'i8', 'u4', 'u8'          # short dtype mnemonics seen in some metas
                )
                return any(k in s for k in ok)
            
            # Filter candidate features
            candidates = []
            target_skipped = []
            denylist_skipped = []
            prefix_skipped = []
            non_numeric_skipped = []
            not_included_skipped = []
            
            self._log_info(f"Feature filtering: Starting with {len(all_columns)} total columns")
            
            for col in all_columns:
                # Skip target columns and metrics
                if (col.startswith(('y_ret_fwd_', 'dcor_', 'adf_', 'stage1_', 'cpcv_'))):
                    target_skipped.append(col)
                    continue
                
                # Skip denylist columns
                if col in self.feature_denylist:
                    denylist_skipped.append(col)
                    continue
                
                # Skip deny prefix columns
                if any(col.startswith(prefix) for prefix in self.feature_deny_prefixes):
                    prefix_skipped.append(col)
                    continue
                
                # Include frac_diff columns and other relevant features, but only numeric types
                if ('frac_diff' in col or col.startswith(('y_', 'x_')) or any(col.startswith(prefix) for prefix in self.feature_allow_prefixes)):
                    dt = dtype_map.get(col, '')
                    if _is_numeric_dt(dt):
                        candidates.append(col)
                    else:
                        non_numeric_skipped.append(f"{col}({dt})")
                else:
                    not_included_skipped.append(col)
            
            # Log detailed filtering results
            self._log_info(f"Feature filtering results:", 
                          total_columns=len(all_columns),
                          target_skipped=len(target_skipped),
                          denylist_skipped=len(denylist_skipped),
                          prefix_skipped=len(prefix_skipped),
                          non_numeric_skipped=len(non_numeric_skipped),
                          not_included_skipped=len(not_included_skipped),
                          final_candidates=len(candidates))
            
            if target_skipped:
                self._log_info(f"Target/metrics columns skipped: {target_skipped[:10]}")
            if denylist_skipped:
                self._log_info(f"Denylist columns skipped: {denylist_skipped[:10]}")
            if prefix_skipped:
                self._log_info(f"Deny prefix columns skipped: {prefix_skipped[:10]}")
            if non_numeric_skipped:
                self._log_info(f"Non-numeric columns skipped: {non_numeric_skipped[:10]}")
            if not_included_skipped:
                self._log_info(f"Not included columns (no matching patterns): {not_included_skipped[:10]}")
            
            self._log_info(f"Final candidates for dCor computation: {candidates[:15]}")
            self._log_info(f"Total features entering dCor computation: {len(candidates)}")
            self._log_info(f"Note: These {len(candidates)} features will be used for distance correlation analysis")
            
            return candidates
            
        except Exception as e:
            self._log_warn(f"Error finding candidate features: {e}")
            return []

    def _apply_distance_correlation_analysis(self, df: dask_cudf.DataFrame, target: str, candidates: List[str]) -> dask_cudf.DataFrame:
        """Compute distance correlation on a global sample and keep scores in-memory only.

        Note: does NOT add `dcor_*` columns to the DataFrame to avoid column explosion.
        Stores results in `self._last_dcor_scores` for downstream selection.
        """
        try:
            # Build a tail sample across last partitions to represent recent behavior - Fixed 100k sample
            sample_n = 100000
            sample = self._sample_tail_across_partitions(df, sample_n, max_parts=16)
            if sample is None or len(sample) == 0 or target not in sample.columns:
                self._log_warn("dCor tail sample empty or missing target; trying head sample")
                sample = self._sample_head_across_partitions(df, sample_n, max_parts=16)
                if sample is None or len(sample) == 0 or target not in sample.columns:
                    self._log_warn("dCor sampling failed; skipping dCor broadcast")
                    return df
            # Convert target to CuPy
            try:
                y = sample[target].astype('f8').to_cupy()
                # Log GPU context for transparency
                try:
                    import os as _os
                    dev = int(cp.cuda.runtime.getDevice())
                    vis = _os.environ.get('CUDA_VISIBLE_DEVICES', '')
                    self._log_info("dCor GPU context", gpu_device=dev, visible_devices=vis, sample_rows=len(sample))
                except Exception:
                    pass
            except Exception as e:
                self._log_warn("dCor tail sample: failed to pull target to GPU", error=str(e))
                return df
            # Optional: fast prefilter using Pearson corr on GPU to shrink candidate set
            dcor_map: Dict[str, float] = {}
            try:
                pre_k = int(min(max(10, getattr(self, 'dcor_top_k', 50) * 3), 200))
            except Exception:
                pre_k = 150

            pref_candidates = candidates
            try:
                # Compute quick Pearson correlations (O(n) per feature) to select top-K for full dCor (O(n^2))
                scores = []
                yf = y.astype(cp.float32, copy=False)
                y_mask = cp.isfinite(yf)
                yv = yf[y_mask]
                if int(yv.size) >= 50:
                    for c in candidates:
                        if c not in sample.columns:
                            continue
                        try:
                            xv = sample[c].astype('f4').to_cupy()
                            m = y_mask & cp.isfinite(xv)
                            if int(m.sum().item()) < 50:
                                continue
                            xa = xv[m]
                            ya = yf[m]
                            # Pearson corr in a few ops
                            xm = xa.mean()
                            ym = ya.mean()
                            xs = xa - xm
                            ys = ya - ym
                            denom = cp.sqrt((xs * xs).sum() * (ys * ys).sum())
                            if float(denom) == 0.0:
                                continue
                            corr = float((xs * ys).sum() / denom)
                            scores.append((c, abs(corr)))
                        except Exception:
                            continue
                    if scores:
                        scores.sort(key=lambda kv: kv[1], reverse=True)
                        pref_candidates = [name for name, _ in scores[:pre_k]]
                        self._log_info("Prefilter via Pearson corr applied", total=len(candidates), kept=len(pref_candidates), pre_k=pre_k)
            except Exception as _pf_err:
                self._log_warn("Prefilter via Pearson corr failed; proceeding with full candidate set", error=str(_pf_err))

            # Compute dCor for prefiltered candidates on the sample
            total = len(pref_candidates)
            last_log = -1
            for i, c in enumerate(pref_candidates, start=1):
                if c not in sample.columns:
                    dcor_map[f"dcor_{c}"] = float('nan')
                    continue
                try:
                    x = sample[c].astype('f8').to_cupy()
                    val = self.distance_correlation._distance_correlation_gpu(
                        x, y, tile=int(self.dcor_tile_size), max_n=int(self.dcor_max_samples)
                    )
                    dcor_map[f"dcor_{c}"] = float(val)
                except Exception:
                    dcor_map[f"dcor_{c}"] = float('nan')
                # Progress log every ~25 features or at end
                try:
                    if total > 0:
                        step = max(1, min(25, total // 10 or 1))
                        if (i == total) or (i // step) > last_log:
                            last_log = i // step
                            self._log_info("dCor progress", processed=i, total=total)
                except Exception:
                    pass
            # Optional: log top-K dCor scores for visibility
            try:
                topk = int(getattr(self, 'dcor_permutation_top_k', 0) or getattr(self, 'dcor_top_k', 20))
            except Exception:
                topk = 20
            try:
                items = [(k.replace('dcor_', ''), v) for k, v in dcor_map.items() if np.isfinite(v)]
                items.sort(key=lambda kv: kv[1], reverse=True)
                show = items[:max(1, min(topk, 10))]
                # Inline the list in the message so it always appears in logs
                top_pairs = ", ".join([f"{name}:{val:.4f}" for name, val in show])
                self._log_info(f"dCor tail sample top{len(show)}: {top_pairs}")
            except Exception:
                pass

            # Keep in memory only for selection; avoid adding columns to df
            try:
                # Convert to feature->score mapping for convenience in selection
                self._last_dcor_scores = {k.replace('dcor_', ''): v for k, v in dcor_map.items()}
            except Exception:
                self._last_dcor_scores = {}
            return df
        except Exception as e:
            self._log_warn(f"Error in distance correlation analysis (tail-based): {e}")
            return df

    def _apply_dcor_filters(self, candidates: List[str], dcor_scores: Dict[str, float]) -> List[str]:
        """
        Apply dCor-based filtering (Stage 1 filters) to candidates.
        
        Args:
            candidates: List of candidate feature names
            dcor_scores: Dictionary of distance correlation scores
            
        Returns:
            List of filtered candidate feature names
        """
        try:
            self._log_info(f"DEBUG: _apply_dcor_filters called with {len(candidates)} candidates and {len(dcor_scores)} scores")
            self._log_info(f"DEBUG: dcor_min_threshold = {self.dcor_min_threshold}, stage1_top_n = {self.stage1_top_n}")
            
            if not candidates or not dcor_scores:
                self._log_info("dCor filtering: No candidates or scores available")
                return candidates
            
            original_count = len(candidates)
            filtered_candidates = []
            
            # Step 1: Apply dcor_min_threshold filter
            threshold_filtered = []
            for candidate in candidates:
                score = dcor_scores.get(candidate, 0.0)
                if score >= self.dcor_min_threshold:
                    threshold_filtered.append(candidate)
                else:
                    self._log_debug(f"dCor filter: {candidate} removed (score={score:.4f} < threshold={self.dcor_min_threshold})")
            
            threshold_count = len(threshold_filtered)
            self._log_info(f"dCor threshold filter: {original_count} → {threshold_count} candidates (threshold={self.dcor_min_threshold})")
            
            # Step 2: Apply stage1_top_n filter (if enabled)
            if self.stage1_top_n > 0 and threshold_count > self.stage1_top_n:
                # Sort by dCor score (descending) and take top N
                scored_candidates = [(candidate, dcor_scores.get(candidate, 0.0)) for candidate in threshold_filtered]
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                
                top_candidates = [candidate for candidate, score in scored_candidates[:self.stage1_top_n]]
                filtered_candidates = top_candidates
                
                self._log_info(f"dCor top_n filter: {threshold_count} → {len(filtered_candidates)} candidates (top_n={self.stage1_top_n})")
                
                # Log the top candidates for visibility
                top_scores = [(candidate, score) for candidate, score in scored_candidates[:min(10, len(scored_candidates))]]
                self._log_info(f"dCor top candidates: {[(c, f'{s:.4f}') for c, s in top_scores]}")
            else:
                filtered_candidates = threshold_filtered
                if self.stage1_top_n > 0:
                    self._log_info(f"dCor top_n filter: Not applied (candidates={threshold_count} <= top_n={self.stage1_top_n})")
                else:
                    self._log_info(f"dCor top_n filter: Disabled (top_n={self.stage1_top_n})")
            
            # Step 3: Apply dcor_min_percentile filter (if enabled)
            if self.dcor_min_percentile > 0.0 and len(filtered_candidates) > 1:
                # Calculate percentile threshold
                scores = [dcor_scores.get(candidate, 0.0) for candidate in filtered_candidates]
                scores.sort()
                percentile_index = int(len(scores) * self.dcor_min_percentile)
                percentile_threshold = scores[percentile_index] if percentile_index < len(scores) else scores[-1]
                
                percentile_filtered = [candidate for candidate in filtered_candidates 
                                     if dcor_scores.get(candidate, 0.0) >= percentile_threshold]
                
                self._log_info(f"dCor percentile filter: {len(filtered_candidates)} → {len(percentile_filtered)} candidates (percentile={self.dcor_min_percentile}, threshold={percentile_threshold:.4f})")
                filtered_candidates = percentile_filtered
            
            final_count = len(filtered_candidates)
            self._log_info(f"dCor filtering complete: {original_count} → {final_count} candidates")
            
            return filtered_candidates
            
        except Exception as e:
            self._log_warn(f"Error in dCor filtering: {e}")
            return candidates  # Return original candidates on error

    def _apply_feature_selection_pipeline(self, df: dask_cudf.DataFrame, target: str) -> Dict[str, Any]:
        """Apply the complete feature selection pipeline."""
        try:
            # Get candidate features
            candidates = self._find_candidate_features(df)
            if not candidates:
                self._critical_error("No candidate features available for selection")
            
            # Log initial candidates to verify filtering worked
            self._log_info(f"Feature selection pipeline: Starting with {len(candidates)} candidates from filtering")
            self._log_info(f"First 10 candidates: {candidates[:10]}")
            
            # Check if any forbidden features are in candidates (this should not happen)
            forbidden_in_candidates = []
            for col in candidates:
                if (col.startswith(('y_ret_fwd_', 'dcor_', 'adf_', 'stage1_', 'cpcv_')) or 
                    col in self.feature_denylist or
                    any(col.startswith(prefix) for prefix in self.feature_deny_prefixes)):
                    forbidden_in_candidates.append(col)
            
            if forbidden_in_candidates:
                self._log_warn(f"WARNING: Found {len(forbidden_in_candidates)} forbidden features in candidates: {forbidden_in_candidates}")
            else:
                self._log_info("✓ All candidates passed initial filtering (no forbidden features found)")

            # Apply feature selection to a sample of the data (span more partitions safely)
            sample_df = self._sample_head_across_partitions(df, self.selection_max_rows, max_parts=16)

            # Sanity-check candidates against the sample: keep only columns castable to float
            valid_candidates: List[str] = []
            invalid_candidates: List[str] = []
            
            self._log_info(f"Sanitizing {len(candidates)} candidate features", 
                          sample_size=len(sample_df), candidates=candidates[:10])
            
            missing_columns = []
            dtype_failures = []
            
            for c in candidates:
                try:
                    # Check if column exists in sample
                    if c not in sample_df.columns:
                        missing_columns.append(c)
                        invalid_candidates.append(c)
                        continue
                    
                    # Attempt a lightweight dtype cast on sample to ensure numeric compatibility
                    _ = sample_df[c].astype('f4')
                    valid_candidates.append(c)
                except Exception as e:
                    dtype_failures.append(f"{c}: {str(e)}")
                    invalid_candidates.append(c)
                    self._log_debug(f"Column {c} failed numeric validation: {e}")
            
            # Log detailed sanitization results
            self._log_info(
                f"Sanitization results: original={len(candidates)}, valid={len(valid_candidates)}, "
                f"missing={len(missing_columns)}, dtype_failures={len(dtype_failures)}"
            )
            
            if missing_columns:
                self._log_info(f"Sample missing columns (first 10): {missing_columns[:10]}")
            if dtype_failures:
                self._log_info(f"Dtype conversion failures (first 10): {dtype_failures[:10]}")
            
            if invalid_candidates:
                # Trim to avoid downstream MI/GPU failures (informational)
                self._log_info("Dropping non-numeric candidates prior to selection", 
                              dropped=len(invalid_candidates), 
                              invalid_candidates=invalid_candidates[:10])

            if not valid_candidates:
                self._log_error("No valid numeric candidates after sanitization", 
                               total_candidates=len(candidates),
                               invalid_count=len(invalid_candidates),
                               sample_columns=list(sample_df.columns)[:20])
                
                # Fallback: try a more lenient validation approach
                self._log_info("Attempting fallback validation with more lenient criteria...")
                fallback_candidates = []
                fallback_failures = []
                
                self._log_info(f"Fallback: Testing {len(candidates)} candidates against sample with {len(sample_df)} rows")
                
                for c in candidates:
                    try:
                        if c not in sample_df.columns:
                            fallback_failures.append(f"{c}: not in sample columns")
                            continue
                            
                        # Try to get a small sample and check if it's numeric
                        sample_values = sample_df[c].head(10)
                        self._log_debug(f"Fallback testing {c}: sample values = {list(sample_values)[:5]}")
                        
                        # Check if we can convert to numeric
                        numeric_count = 0
                        non_numeric_values = []
                        for val in sample_values:
                            try:
                                float(val)
                                numeric_count += 1
                            except (ValueError, TypeError):
                                non_numeric_values.append(str(val))
                        
                        # If at least 30% of sample values are numeric, consider it valid (more lenient)
                        if numeric_count >= len(sample_values) * 0.3:
                            fallback_candidates.append(c)
                            self._log_info(f"Fallback validation passed for {c} ({numeric_count}/{len(sample_values)} numeric values)")
                        else:
                            fallback_failures.append(f"{c}: only {numeric_count}/{len(sample_values)} numeric, non-numeric: {non_numeric_values[:3]}")
                            
                    except Exception as e:
                        fallback_failures.append(f"{c}: exception - {str(e)}")
                        self._log_debug(f"Fallback validation failed for {c}: {e}")
                
                self._log_info(f"Fallback results: {len(fallback_candidates)} passed, {len(fallback_failures)} failed")
                if fallback_failures:
                    self._log_info(f"Fallback failures: {fallback_failures[:10]}")
                
                if fallback_candidates:
                    self._log_info(f"Fallback validation found {len(fallback_candidates)} valid candidates")
                    valid_candidates = fallback_candidates
                else:
                    # Last resort: accept all candidates and let downstream handle errors
                    self._log_warn("All fallback validations failed. Accepting all candidates as last resort...")
                    valid_candidates = candidates
                    self._log_info(f"Last resort: using all {len(candidates)} original candidates")
                    
                    # Double-check: ensure we're not reintroducing forbidden features
                    forbidden_in_last_resort = []
                    for col in valid_candidates:
                        if (col.startswith(('y_ret_fwd_', 'dcor_', 'adf_', 'stage1_', 'cpcv_')) or 
                            col in self.feature_denylist or
                            any(col.startswith(prefix) for prefix in self.feature_deny_prefixes)):
                            forbidden_in_last_resort.append(col)
                    
                    if forbidden_in_last_resort:
                        self._log_error(f"CRITICAL: Last resort reintroduced {len(forbidden_in_last_resort)} forbidden features: {forbidden_in_last_resort}")
                        # Remove forbidden features even in last resort
                        valid_candidates = [c for c in valid_candidates if c not in forbidden_in_last_resort]
                        self._log_info(f"Removed forbidden features from last resort. Final count: {len(valid_candidates)}")
            
            # Get distance correlation scores (prefer in-memory scores; fallback to columns if present)
            dcor_scores: Dict[str, float] = {}
            # First, try the in-memory scores computed during dCor phase
            try:
                mem_scores = getattr(self, '_last_dcor_scores', {}) or {}
                self._log_info(f"DEBUG: Retrieved {len(mem_scores)} dCor scores from memory")
                if mem_scores:
                    for col in valid_candidates:
                        dcor_scores[col] = float(mem_scores.get(col, 0.0))
                    self._log_info(f"DEBUG: Mapped {len(dcor_scores)} scores for valid candidates")
                else:
                    self._log_warn("DEBUG: No dCor scores found in memory")
            except Exception as e:
                self._log_warn(f"DEBUG: Error getting dCor scores from memory: {e}")
                pass
            # If still empty, fallback to reading from sample columns (legacy path)
            if not dcor_scores and len(sample_df) > 0:
                for col in valid_candidates:
                    dcor_col = f"dcor_{col}"
                    if dcor_col in sample_df.columns:
                        try:
                            val = sample_df[dcor_col].iloc[0]
                            score = float(val) if val is not None else 0.0
                        except Exception:
                            score = 0.0
                        dcor_scores[col] = score
            
            # Apply dCor-based filtering (Stage 1 filters)
            filtered_candidates = self._apply_dcor_filters(valid_candidates, dcor_scores)
            
            # Apply feature selection pipeline
            selection_results = self.feature_selection.apply_feature_selection_pipeline(
                sample_df, target, filtered_candidates, dcor_scores, full_ddf=df
            )
            return selection_results
        except Exception as e:
            # Escalate: this is a critical failure for the stats engine
            self._critical_error(f"Feature selection pipeline failed: {e}")

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
            # Try to get currency pair from column names without triggering compute
            cols = list(df.columns)
            for col in cols:
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
