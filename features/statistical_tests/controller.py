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
from .utils import _free_gpu_memory_worker, _tail_k, _tail_k_to_pandas, _jb_rolling_partition
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
            # Rolling JB configuration
            self.jb_base_column = getattr(uc.features, 'jb_base_column', None)
            try:
                self.jb_windows = list(getattr(uc.features, 'jb_windows', []) or [])
            except Exception:
                self.jb_windows = []
            
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
            # JB defaults
            self.jb_base_column = 'y_ret_1m'
            self.jb_windows = []

    def _log_info(self, message: str, **kwargs):
        """Log info message with optional context."""
        logger.info(f"StatisticalTests: {message}", extra=kwargs)
    
    def _log_warn(self, message: str, **kwargs):
        """Log warning message with optional context."""
        logger.warning(f"StatisticalTests: {message}", extra=kwargs)
    
    def _log_error(self, message: str, **kwargs):
        """Log error message with optional context."""
        logger.error(f"StatisticalTests: {message}", extra=kwargs)
    
    def _log_debug(self, message: str, **kwargs):
        """Log debug message with optional context."""
        logger.debug(f"StatisticalTests: {message}", extra=kwargs)
    
    def _critical_error(self, message: str, **kwargs):
        """Log critical error and raise exception."""
        self._log_error(message, **kwargs)
        raise RuntimeError(f"StatisticalTests Critical Error: {message}")

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
            self._log_info("Starting comprehensive statistical tests pipeline...")
            
            # 1. ADF TESTS IN BATCH
            df = self.adf_tests._apply_comprehensive_adf_tests(df)
            
            # 1b. JB rolling p-values (cuDF path)
            try:
                jb_windows = list(getattr(self, 'jb_windows', []) or [])
            except Exception:
                jb_windows = []
            if jb_windows:
                base = getattr(self, 'jb_base_column', None) or ('y_ret_1m' if 'y_ret_1m' in df.columns else None)
                if base is None:
                    # fallback: first column containing 'ret'
                    for c in df.columns:
                        if 'ret' in str(c).lower():
                            base = c
                            break
                if base is not None and base in df.columns:
                    self._log_info(f"JB (cuDF): Computing rolling JB for base={base} windows={jb_windows}")
                    for w in jb_windows:
                        try:
                            win = int(w)
                            minp = max(10, int(0.5 * win))
                            safe = re.sub(r"[^0-9a-zA-Z_]+", "_", str(base))
                            out = f"jb_p_{safe}_w{win}"
                            if out in df.columns:
                                continue
                            df[out] = _jb_rolling_partition(df[base], win, minp)
                        except Exception as e:
                            self._log_warn("JB (cuDF) rolling failed", window=str(w), error=str(e))
            
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

        # --- ADF rolling window on fractional difference columns ---
        adf_cols = [c for c in df.columns if "frac_diff" in c]
        
        if adf_cols:
            self._log_info(f"ADF: Found {len(adf_cols)} frac_diff features to process", 
                          features=adf_cols[:10],  # Show first 10 features
                          total_count=len(adf_cols))
        else:
            self._log_info("ADF: No frac_diff features found to process")

        processed_count = 0
        skipped_count = 0

        for col in adf_cols:
            # Use a unique, safe output column name to avoid collisions across frac_diff variants
            try:
                safe = re.sub(r"[^0-9a-zA-Z_]+", "_", str(col))
            except Exception:
                safe = str(col)
            out = f"adf_stat_{safe}"
            # Skip if the column already exists to prevent redundant recomputation
            try:
                if out in df.columns:
                    skipped_count += 1
                    continue
            except Exception as e:
                self._log_error(f"Failed to check if column exists: {e}")
                pass
            
            df[out] = df[col].map_partitions(
                self.adf_tests._apply_adf_rolling,
                252,  # Window size (1 year of trading days)
                200,  # Minimum periods
                meta=(out, "f8"),
            )
            processed_count += 1
            
            # Log progress every 10 features to reduce spam
            if processed_count % 10 == 0:
                self._log_info(f"ADF: Processed {processed_count}/{len(adf_cols)} features...")

        if adf_cols:
            self._log_info(f"ADF: Completed processing {processed_count} features, skipped {skipped_count} (already exist)")

        # --- Rolling JB p-value on base return column(s) ---
        try:
            jb_windows = list(getattr(self, 'jb_windows', []) or [])
        except Exception:
            jb_windows = []
        if jb_windows:
            # Determine base column
            try:
                cols = list(df.columns)
            except Exception:
                cols = []
            base = None
            cfg_base = getattr(self, 'jb_base_column', None)
            if cfg_base and (cfg_base in cols):
                base = cfg_base
            elif 'y_ret_1m' in cols:
                base = 'y_ret_1m'
            else:
                # fallback: first column containing 'ret'
                for c in cols:
                    if 'ret' in str(c).lower():
                        base = c
                        break
            if base is None:
                self._log_warn("JB: No suitable base return column found; skipping rolling JB features")
            else:
                self._log_info(f"JB: Computing rolling JB p-values for base={base} windows={jb_windows}")
                for w in jb_windows:
                    try:
                        win = int(w)
                        minp = max(10, int(0.5 * win))
                        safe = re.sub(r"[^0-9a-zA-Z_]+", "_", str(base))
                        out = f"jb_p_{safe}_w{win}"
                        if out in df.columns:
                            continue
                        # Apply mask for DXY base if available (mask closed periods to null)
                        base_series = df[base]
                        try:
                            if ('is_dxy_open' in df.columns) and ('dxy' in str(base).lower()):
                                base_series = base_series.where(df['is_dxy_open'].astype('bool'), None)
                        except Exception:
                            pass
                        df[out] = base_series.map_partitions(
                            _jb_rolling_partition,
                            win,
                            minp,
                            meta=(out, 'f8')
                        )
                    except Exception as e:
                        self._log_warn("JB rolling failed for a window", window=str(w), error=str(e))

        # --- Pre-selection Statistical Analysis (broadcast constants; no corr matrix) ---
        try:
            self._log_info("Pre-selection: computing statistical analysis features on tail sample…")
            stats_sample = self._sample_tail_across_partitions(df, min(self.selection_max_rows, 50000), max_parts=8)
            if stats_sample is not None and len(stats_sample) > 0:
                before_cols = set(map(str, stats_sample.columns))
                # Use lightweight preselection stats (no correlation matrices here)
                stats_enriched = self.statistical_analysis.apply_preselection_stat_features(stats_sample)
                after_cols = set(map(str, stats_enriched.columns))
                added = sorted(list(after_cols - before_cols))
                consts = {}
                for c in added:
                    try:
                        col = stats_enriched[c]
                        val = None
                        if hasattr(col, 'iloc') and len(col) > 0:
                            val = col.iloc[0]
                        if val is None:
                            continue
                        try:
                            consts[c] = float(val)
                        except Exception:
                            consts[c] = val
                    except Exception:
                        continue
                if consts:
                    df = df.assign(**consts)
                    self._log_info(f"Pre-selection: broadcasted {len(consts)} statistical features")
                    try:
                        df = df.persist()
                    except Exception:
                        pass
            else:
                self._log_warn("Pre-selection: tail sample empty; skipping statistical features broadcast")
        except Exception as e:
            self._log_warn("Pre-selection statistical analysis failed; continuing", error=str(e))

        # --- Distance Correlation Analysis ---
        if primary_target and primary_target in df.columns:
            self._log_info("Starting distance correlation analysis...")
            
            # Find candidate features
            candidate_features = self._find_candidate_features(df)
            
            if candidate_features:
                self._log_info(f"Found {len(candidate_features)} candidate features for distance correlation", 
                              candidates=candidate_features[:15],  # Show first 15 candidates
                              total_count=len(candidate_features))
                
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
                    # Unknown dtype: be conservative and exclude
                    return False
                bad = ('object', 'str', 'string', 'category', 'datetime', 'timedelta', 'date')
                if any(b in s for b in bad):
                    return False
                ok = ('int', 'float', 'double', 'bool')
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
        """Apply distance correlation once on a global tail sample, then broadcast constants."""
        try:
            # Build a tail sample across last partitions to represent recent behavior
            sample_n = int(max(1000, min(self.dcor_max_samples * 2, 200000)))
            sample = self._sample_tail_across_partitions(df, sample_n, max_parts=16)
            if sample is None or len(sample) == 0 or target not in sample.columns:
                self._log_warn("dCor tail sample empty or target missing; skipping dCor broadcast")
                return df
            # Convert target to CuPy
            try:
                y = sample[target].astype('f8').to_cupy()
            except Exception as e:
                self._log_warn("dCor tail sample: failed to pull target to GPU", error=str(e))
                return df
            # Compute dCor for each candidate on the sample
            dcor_map: Dict[str, float] = {}
            for c in candidates:
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

            # Broadcast constants as new columns across the Dask DataFrame
            if dcor_map:
                df = df.assign(**dcor_map)
                # Persist to avoid recomputation of upstream graph in subsequent stages
                try:
                    df = df.persist()
                except Exception:
                    pass
            return df
        except Exception as e:
            self._log_warn(f"Error in distance correlation analysis (tail-based): {e}")
            return df

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
            
            # Get distance correlation scores (robust to NA/pd.NA/cudf.NA)
            dcor_scores = {}
            if len(sample_df) > 0:
                for col in valid_candidates:
                    dcor_col = f"dcor_{col}"
                    if dcor_col in sample_df.columns:
                        try:
                            val = sample_df[dcor_col].iloc[0]
                            try:
                                score = float(val)
                            except Exception:
                                score = 0.0
                            dcor_scores[col] = float(score)
                        except Exception:
                            dcor_scores[col] = 0.0
            
            # Apply feature selection pipeline
            selection_results = self.feature_selection.apply_feature_selection_pipeline(
                sample_df, target, valid_candidates, dcor_scores
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
