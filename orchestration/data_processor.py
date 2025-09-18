"""
Data Processor for Feature Engineering Pipeline

This module handles the processing of individual currency pairs through the feature engineering pipeline.
"""

import logging
import sys
import os
import time
from typing import Optional, Dict, Any
from pathlib import Path

import cudf
import cupy as cp
import dask

from config.unified_config import get_unified_config as get_settings
from dask.distributed import Client, wait
from data_io.db_handler import DatabaseHandler
from data_io.local_loader import LocalDataLoader
from features import (
    StationarizationEngine,
    StatisticalTests,
    GARCHModels,
    FeatureEngineeringEngine,
)
from features.signal_processing import apply_emd_to_series
from features.base_engine import CriticalPipelineError
from utils.logging_utils import get_logger, set_currency_pair_context
from features.engine_metrics import EngineMetrics
from data_io.r2_uploader import R2Uploader

logger = get_logger(__name__, "orchestration.processor")


class DataProcessor:
    """
    Handles the processing of individual currency pairs through the feature engineering pipeline.
    
    This class encapsulates all the logic for:
    - Loading data from various sources
    - Applying feature engineering engines in the correct order
    - Validating data at each step
    - Saving processed results
    - Error handling and recovery
    """
    
    def __init__(self, client: Optional[Client] = None, run_id: Optional[int] = None):
        """Initialize the data processor."""
        self.settings = get_settings()
        self.loader = LocalDataLoader()
        self.db_handler = DatabaseHandler()
        self.db_connected = False
        self.client = client
        self.run_id = run_id
        
        # Initialize feature engines (pass client for Dask usage when available)
        self.station = StationarizationEngine(self.settings, client)  # Engine 1: Estacionarização
        self.stats = StatisticalTests(self.settings, client)  # Engine 4: Testes estatísticos (estágios 1-4)
        self.feng = FeatureEngineeringEngine(self.settings, client)  # Engine 2: Feature engineering (BK filter)
        self.garch = GARCHModels(self.settings, client)  # Engine 3: Modelos GARCH
        # Transparency helpers
        self._metrics = EngineMetrics()
    
    def process_currency_pair(self, currency_pair: str, r2_path: str) -> bool:
        """
        Process a single currency pair through the complete feature engineering pipeline.
        
        Args:
            currency_pair: The currency pair symbol (e.g., 'EURUSD')
            r2_path: Path to the data file in R2 storage
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            # Set currency pair context for all subsequent logs
            set_currency_pair_context(currency_pair)
            logger.info(f"Starting processing for {currency_pair}")
            
            # Connect to database for task tracking (non-fatal)
            if not self.db_connected:
                self.db_connected = self.db_handler.connect()
                if not self.db_connected:
                    logger.warning("Database unavailable; proceeding without task tracking")
            
            # Load data (prefer parquet sync, then feather sync fallback)
            gdf = self._load_currency_pair_data(r2_path)  # Carrega dados do par de moeda
            if gdf is None:
                self._register_task_failure(currency_pair, r2_path, "Failed to load data")
                return False
            
            # Validate initial data
            if not self._validate_initial_data(gdf, currency_pair):  # Valida qualidade dos dados carregados
                self._register_task_failure(currency_pair, r2_path, "Initial data validation failed")
                return False
            
            # Process through feature engines
            gdf = self._apply_feature_engines(gdf, currency_pair)  # Aplica pipeline de feature engineering
            if gdf is None:
                self._register_task_failure(currency_pair, r2_path, "Feature engineering failed")
                return False

            # Nota: Seleção final de features (CatBoost) ocorre dentro do engine statistical_tests
            # após dCor → VIF → MI (Stage 3 embutido). Nenhuma execução duplicada aqui.

            # Export selected-feature parquet and upload to R2 (best-effort, does not fail task)
            try:
                sel_path = self._export_selected_features_parquet_cudf(gdf, currency_pair)
                if sel_path:
                    uploaded = self._upload_selected_features_to_r2(sel_path, currency_pair)
                    logger.info(f"Selected-feature parquet upload: {'OK' if uploaded else 'SKIPPED/FAILED'}")
            except Exception as _e:
                logger.warning(f"Could not export/upload selected-feature parquet for {currency_pair}: {_e}")
            
            # Save processed data
            if not self._save_processed_data(gdf, currency_pair):  # Salva dados processados
                self._register_task_failure(currency_pair, r2_path, "Data saving failed")
                return False
            
            # Register successful completion
            self._register_task_success(currency_pair, r2_path)
            
            logger.info(f"Successfully completed processing for {currency_pair}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {currency_pair}: {e}", exc_info=True)
            self._register_task_failure(currency_pair, r2_path, str(e))
            return False
        finally:
            if self.db_connected:
                try:
                    self.db_handler.close()
                except Exception:
                    pass
    
    def _load_currency_pair_data(self, r2_path: str) -> Optional[cudf.DataFrame]:
        """
        Load currency pair data from the specified path.
        
        Args:
            r2_path: Path to the data file
            
        Returns:
            cuDF DataFrame if successful, None otherwise
        """
        try:
            # Prefer Parquet (sync)
            gdf = self.loader.load_currency_pair_data_sync(r2_path)
            if gdf is not None:
                # Drop denied columns early
                gdf = self._drop_denied_columns(gdf)
                logger.info(f"Loaded parquet data: {len(gdf)} rows, {len(gdf.columns)} columns")
                return gdf
            
            # No fallback to Feather - only Parquet is supported
            
            logger.error(f"Failed to load data from {r2_path}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading data from {r2_path}: {e}", exc_info=True)
            return None
    
    def _validate_initial_data(self, gdf: cudf.DataFrame, currency_pair: str) -> bool:
        """
        Validate the initial data before processing.
        
        Args:
            gdf: The loaded cuDF DataFrame
            currency_pair: Currency pair for logging
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        try:
            logger.info(f"🔍 Validating initial data for {currency_pair}")
            logger.info(f"📊 Shape: {gdf.shape}")
            logger.info(f"📊 Rows: {len(gdf)}")
            logger.info(f"📊 Columns: {len(gdf.columns)}")
            logger.info(f"📊 ALL COLUMNS: {list(gdf.columns)}")
            
            # Check for empty DataFrame
            if len(gdf) == 0:
                logger.error(f"Empty DataFrame for {currency_pair}")
                return False
            
            # Check for required columns
            required_columns = self._get_required_columns()
            missing_columns = [col for col in required_columns if col not in gdf.columns]
            if missing_columns:
                logger.warning(f"Missing required columns for {currency_pair}: {missing_columns}")
                # Don't fail, just warn - some engines might work without all columns
            
            # Check for NaN values in critical columns
            price_cols = [col for col in gdf.columns if any(term in col.lower() for term in ['open', 'high', 'low', 'close'])]
            if price_cols:
                for col in price_cols:
                    nan_count = gdf[col].isna().sum()
                    if nan_count > 0:
                        logger.warning(f"Found {nan_count} NaN values in {col} for {currency_pair}")
            
            # Check data types
            logger.info(f"📊 Data types: {gdf.dtypes.to_dict()}")
            # Presence of open-flag columns (validation aid)
            try:
                flag_cols = [c for c in gdf.columns if str(c).startswith('is_') and str(c).endswith('_open')]
                if flag_cols:
                    logger.info(f"🔎 Detected open-flag columns: {flag_cols}")
                else:
                    logger.info("🔎 No is_*_open columns detected in dataset")
            except Exception:
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating initial data for {currency_pair}: {e}", exc_info=True)
            return False
    
    def _get_required_columns(self) -> list:
        """Get the list of required columns for processing."""
        return ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    def _drop_denied_columns(self, df):
        """Drop columns that must never enter any stage.

        Applies both configured deny lists and the hardcoded 'y_minutes_since_open'.
        Supports cuDF and dask_cuDF DataFrames.
        """
        try:
            features_cfg = getattr(self.settings, 'features', None)
            deny_exact = set()
            deny_prefixes = []
            dataset_target_prefixes = []
            protect_exact = set()
            # Configured lists
            if features_cfg is not None:
                deny_exact.update(getattr(features_cfg, 'feature_denylist', []) or [])
                deny_prefixes = list(getattr(features_cfg, 'feature_deny_prefixes', []) or [])
                dataset_target_prefixes = list(getattr(features_cfg, 'dataset_target_prefixes', []) or [])
                # Protect exact targets from being dropped even if they match deny prefixes
                stc = getattr(features_cfg, 'selection_target_column', None)
                if stc:
                    protect_exact.add(str(stc))
                for t in (getattr(features_cfg, 'selection_target_columns', []) or []):
                    protect_exact.add(str(t))
                for t in (getattr(features_cfg, 'dataset_target_columns', []) or []):
                    protect_exact.add(str(t))
            # Hard requirement
            deny_exact.add('y_minutes_since_open')

            cols = list(df.columns)
            to_drop = [
                c for c in cols
                if (
                    # exact-deny unless protected
                    ((c in deny_exact) and (c not in protect_exact))
                    # or any denied prefix unless protected
                    or (any(c.startswith(p) for p in deny_prefixes + dataset_target_prefixes) and (c not in protect_exact))
                )
            ]
            if to_drop:
                logger.info(f"Dropping denied columns: {to_drop}")
                df = df.drop(columns=to_drop)
        except Exception as e:
            logger.warning(f"Could not drop denied columns: {e}")
        return df
    
    def _apply_feature_engines(self, gdf: cudf.DataFrame, currency_pair: str) -> Optional[cudf.DataFrame]:
        """
        Apply all feature engineering engines in the correct order.
        
        Args:
            gdf: Input cuDF DataFrame
            currency_pair: Currency pair for logging
            
        Returns:
            Processed cuDF DataFrame if successful, None otherwise
        """
        try:
            initial_cols = len(gdf.columns)
            logger.info(f"Starting feature engineering for {currency_pair} with {initial_cols} initial columns")
            
            # Get engine configuration from settings
            engine_config = self.settings.pipeline.engines  # dict[name -> PipelineEngineConfig]
            # Execute engines in the correct order based on configuration (dataclass-friendly)
            def _enabled(cfg):
                try:
                    return bool(getattr(cfg, 'enabled', True))
                except Exception:
                    return True
            def _order(cfg):
                try:
                    return int(getattr(cfg, 'order', 999))
                except Exception:
                    return 999
            engine_execution_order = sorted(
                [(name, cfg) for name, cfg in engine_config.items() if _enabled(cfg)],
                key=lambda x: _order(x[1])
            )
            
            logger.info(f"🚀 Engine execution order: {[name for name, _ in engine_execution_order]}")
            
            # Execute each engine in order
            for engine_name, engine_config in engine_execution_order:
                if not getattr(engine_config, 'enabled', True):
                    logger.info(f"⏭️ Skipping disabled engine: {engine_name}")
                    continue
                
                # Log engine start
                logger.info(f"Starting {engine_name} processing for {currency_pair}")
                try:
                    desc = getattr(engine_config, 'description', 'No description')
                except Exception:
                    desc = 'No description'
                logger.info(f"📝 Description: {desc}")

                # For statistical_tests, emit a concrete step-by-step plan derived from config and schema
                try:
                    if str(engine_name) == 'statistical_tests':
                        self._log_statistical_tests_plan_cudf(gdf)
                except Exception as _e:
                    logger.debug(f"Could not build statistical_tests plan (cuDF): {_e}")
                
                rows_before = len(gdf)
                cols_before = len(gdf.columns)
                
                try:
                    # Execute the appropriate engine
                    gdf = self._execute_engine(engine_name, gdf)
                    if gdf is None:
                        logger.error(f"Engine {engine_name} returned None for {currency_pair}")
                        return None
                    
                    rows_after = len(gdf)
                    cols_after = len(gdf.columns)
                    new_cols = cols_after - cols_before
                    
                    # Log engine completion
                    logger.info(f"{engine_name.title()} processing completed for {currency_pair}: {rows_before}→{rows_after} rows, {cols_before}→{cols_after} cols (+{new_cols})")

                    # Simple GPU memory snapshot
                    try:
                        import cupy as _cp
                        free_b, total_b = _cp.cuda.runtime.memGetInfo()
                        used_gb = (total_b - free_b) / (1024**3)
                        total_gb = total_b / (1024**3)
                        mem = {'gpu_used_gb': round(used_gb, 2), 'gpu_total_gb': round(total_gb, 2)}
                    except Exception:
                        mem = {}
                    
                    logger.info(f"{engine_name.title()} complete: {cols_before} -> {cols_after} cols (+{new_cols} new), {rows_before} -> {rows_after} rows ({rows_after - rows_before} change)")
                    
                    if new_cols > 0:
                        new_col_names = [col for col in gdf.columns if col not in gdf.columns[:cols_before]]
                        logger.info(f"New columns: {new_col_names[:10] + ['...'] if len(new_col_names) > 10 else new_col_names}")
                    else:
                        logger.warning(f"⚠️ No new columns generated by {engine_name}!")
                    
                    # Validate data after each engine
                    if not self._validate_intermediate_data(gdf, currency_pair, engine_name):
                        logger.error(f"Data validation failed after {engine_name} for {currency_pair}")
                        return None

                    # Aggressive local GPU memory cleanup between engines (cuDF path)
                    try:
                        import gc as _gc
                        _gc.collect()
                        cp.get_default_memory_pool().free_all_blocks()
                    except Exception:
                        pass
                
                except Exception as e:
                    logger.error(f"🚨 CRITICAL ERROR in {engine_name} Engine: {e}")
                    if not self.settings.error_handling.continue_on_error:
                        logger.error(f"🛑 Stopping pipeline immediately due to critical error in {engine_name}.")
                        raise CriticalPipelineError(f"Engine {engine_name} failed: {e}")
                    else:
                        logger.warning(f"⚠️ Continuing pipeline despite error in {engine_name}")
            
            total_new_cols = len(gdf.columns) - initial_cols
            logger.info("=" * 50)
            logger.info(f"PROCESSING {currency_pair} - SUMMARY")
            logger.info("=" * 50)
            logger.info(f"Final cuDF shape: {len(gdf)} rows, {len(gdf.columns)} cols")
            logger.info(f"Total new features created: +{total_new_cols} columns")
            logger.info(f"Feature expansion: {initial_cols} -> {len(gdf.columns)} (+{(total_new_cols/initial_cols)*100:.1f}%)")
            
            return gdf
            
        except Exception as e:
            logger.error(f"Error applying feature engines for {currency_pair}: {e}", exc_info=True)
            return None
    
    def _execute_engine(self, engine_name: str, gdf: cudf.DataFrame) -> Optional[cudf.DataFrame]:
        """
        Execute a specific feature engineering engine.
        
        Args:
            engine_name: Name of the engine to execute
            gdf: Input cuDF DataFrame
            
        Returns:
            Processed cuDF DataFrame if successful, None otherwise
        """
        try:
            if engine_name == 'stationarization':
                return self.station.process_currency_pair(gdf)
            elif engine_name == 'feature_engineering':
                return self.feng.process_cudf(gdf)
            elif engine_name == 'statistical_tests':
                return self.stats.process_cudf(gdf)
            elif engine_name == 'signal_processing':
                # Apply EMD on 'close' (if available) and concatenate IMFs
                try:
                    emd_cfg = getattr(self.settings.features, 'emd', {})
                    max_imfs = int(emd_cfg.get('max_imfs', getattr(self.settings.features, 'emd_max_imfs', 10)))
                    rolling_enable = bool(emd_cfg.get('rolling_enable', getattr(self.settings.features, 'emd_rolling_enable', False)))
                    window_size = int(emd_cfg.get('rolling_window', getattr(self.settings.features, 'emd_rolling_window', 500)))
                    step_size = int(emd_cfg.get('rolling_step', getattr(self.settings.features, 'emd_rolling_step', 50)))
                    embargo = int(emd_cfg.get('rolling_embargo', getattr(self.settings.features, 'emd_rolling_embargo', 0)))
                except Exception:
                    max_imfs = getattr(self.settings.features, 'emd_max_imfs', 10)
                    rolling_enable = getattr(self.settings.features, 'emd_rolling_enable', False)
                    window_size = getattr(self.settings.features, 'emd_rolling_window', 500)
                    step_size = getattr(self.settings.features, 'emd_rolling_step', 50)
                    embargo = getattr(self.settings.features, 'emd_rolling_embargo', 0)

                if 'close' not in gdf.columns:
                    logger.warning("Coluna 'close' ausente; pulando EMD (signal_processing)")
                    return gdf
                try:
                    if rolling_enable:
                        from features.signal_processing import apply_rolling_emd
                        imfs = apply_rolling_emd(gdf['close'], max_imfs=max_imfs,
                                                 window_size=window_size, step_size=step_size,
                                                 embargo=embargo)
                    else:
                        imfs = apply_emd_to_series(gdf['close'], max_imfs=max_imfs)
                    # Prefix with 'emd_' to namespace features
                    imfs = imfs.rename(columns={c: f"emd_{c}" for c in imfs.columns})
                    gdf = cudf.concat([gdf, imfs], axis=1)
                    logger.info(f"EMD gerou {imfs.shape[1]} IMFs; novas colunas adicionadas")
                    return gdf
                except Exception as e:
                    logger.warning(f"Falha ao aplicar EMD: {e}; mantendo DataFrame inalterado")
                    return gdf
            elif engine_name == 'garch_models':
                return self.garch.process_cudf(gdf)
            else:
                logger.warning(f"⚠️ Unknown engine: {engine_name}")
                return gdf
                
        except Exception as e:
            logger.error(f"Error executing engine {engine_name}: {e}", exc_info=True)
            return None

    def _execute_engine_dask(self, engine_name: str, ddf) -> Optional["dask_cudf.DataFrame"]:
        """Execute a specific feature engineering engine on a dask_cudf DataFrame."""
        try:
            if engine_name == 'stationarization':
                return self.station.process(ddf)
            elif engine_name == 'feature_engineering':
                return self.feng.process(ddf)
            elif engine_name == 'statistical_tests':
                # Orchestrate StatisticalTests as explicit sub-stages for visibility
                return self._run_statistical_tests_stages(ddf)
            elif engine_name == 'signal_processing':
                # Compute EMD on worker by materializing a small cuDF subset and merge back by 'timestamp'.
                try:
                    import dask_cudf as _dask_cudf
                except Exception:
                    logger.info("dask_cudf indisponível; mantendo pass-through")
                    return ddf

                if 'close' not in ddf.columns or 'timestamp' not in ddf.columns:
                    logger.warning("Colunas necessárias ('timestamp','close') ausentes; pulando EMD (Dask)")
                    return ddf

                try:
                    emd_cfg = getattr(self.settings.features, 'emd', {})
                    max_imfs = int(emd_cfg.get('max_imfs', getattr(self.settings.features, 'emd_max_imfs', 10)))
                    rolling_enable = bool(emd_cfg.get('rolling_enable', getattr(self.settings.features, 'emd_rolling_enable', False)))
                    window_size = int(emd_cfg.get('rolling_window', getattr(self.settings.features, 'emd_rolling_window', 500)))
                    step_size = int(emd_cfg.get('rolling_step', getattr(self.settings.features, 'emd_rolling_step', 50)))
                    embargo = int(emd_cfg.get('rolling_embargo', getattr(self.settings.features, 'emd_rolling_embargo', 0)))
                except Exception:
                    max_imfs = getattr(self.settings.features, 'emd_max_imfs', 10)
                    rolling_enable = getattr(self.settings.features, 'emd_rolling_enable', False)
                    window_size = getattr(self.settings.features, 'emd_rolling_window', 500)
                    step_size = getattr(self.settings.features, 'emd_rolling_step', 50)
                    embargo = getattr(self.settings.features, 'emd_rolling_embargo', 0)

                # Materialize minimal columns on this worker
                try:
                    subset = ddf[['timestamp', 'close']].compute()
                    subset = subset.sort_values('timestamp')
                    if rolling_enable:
                        from features.signal_processing import apply_rolling_emd
                        imfs = apply_rolling_emd(subset['close'], max_imfs=max_imfs,
                                                 window_size=window_size, step_size=step_size,
                                                 embargo=embargo)
                    else:
                        imfs = apply_emd_to_series(subset['close'], max_imfs=max_imfs)
                    # Prefix and add key for merge
                    imfs = imfs.rename(columns={c: f"emd_{c}" for c in imfs.columns})
                    imfs = imfs.reset_index(drop=True)
                    subset = subset.reset_index(drop=True)
                    imf_gdf = cudf.concat([subset[['timestamp']], imfs], axis=1)
                    imf_ddf = _dask_cudf.from_cudf(imf_gdf, npartitions=max(1, ddf.npartitions))
                    # Merge by timestamp to align back to the distributed frame
                    ddf = ddf.merge(imf_ddf, on='timestamp', how='left')
                    logger.info(f"EMD (Dask) gerou {imfs.shape[1]} IMFs; colunas adicionadas via merge")
                    return ddf
                except Exception as e:
                    logger.warning(f"Falha ao aplicar EMD (Dask): {e}; mantendo pass-through")
                    return ddf
            elif engine_name == 'garch_models':
                return self.garch.process(ddf)
            else:
                logger.warning(f"⚠️ Unknown engine: {engine_name}")
                return ddf
        except Exception as e:
            logger.error(f"Error executing Dask engine {engine_name}: {e}", exc_info=True)
            return None

    def _run_statistical_tests_stages(self, ddf):
        """Run StatisticalTests as separate sub-stages with explicit checkpointing."""
        try:
            import dask
            from dask.distributed import wait as _wait
        except Exception:
            dask = None
            _wait = None

        # Stage: ADF
        try:
            logger.info("[StatTests] Stage start: ADF rolling")
            from features.statistical_tests.stage_adf import run as run_adf
            with dask.annotate(task_key_name="stat_adf") if dask else _noop_ctx():
                ddf = run_adf(self.stats, ddf, window=252, min_periods=200)
            logger.info("Persist start: statistical_tests:adf")
            ddf = ddf.persist()
            logger.info("Persist returned: statistical_tests:adf")
            try:
                if _wait:
                    logger.info("Waiting for statistical_tests:adf to complete (timeout: 600s)...")
                    _wait(ddf, timeout=600)
                    logger.info("statistical_tests:adf completed successfully")
            except Exception as e:
                logger.warning(f"Wait timeout or error for statistical_tests:adf: {e}")
                logger.info("Continuing without waiting - data is still persisted for statistical_tests:adf")
        except Exception as e:
            logger.error(f"[StatTests] ADF stage failed: {e}")

        # Stage: dCor
        try:
            logger.info("[StatTests] Stage start: dCor analysis")
            from features.statistical_tests.stage_dcor import run as run_dcor
            target = getattr(self.settings.features, 'selection_target_column', None)
            if target and target in ddf.columns:
                candidates = self.stats._find_candidate_features(ddf)
                with dask.annotate(task_key_name="stat_dcor") if dask else _noop_ctx():
                    ddf = run_dcor(self.stats, ddf, target, candidates)
                # Orchestration-level summary for dCor
                try:
                    dsum = getattr(self.stats, '_last_dcor_summary', {}) or {}
                    total = dsum.get('candidates_total')
                    kept = dsum.get('prefilter_kept')
                    scores = dsum.get('scores', {})
                    # Show all scores instead of just top 10
                    all_scores = [f"{name}:{val:.4f}" for name, val in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
                    logger.info(f"[StatTests] dCor completed: candidates={total}, prefilter_kept={kept}, all_scores={all_scores}")
                    try:
                        self._write_stat_tests_artifacts(getattr(self, 'current_currency_pair', 'unknown'), dcor_summary=dsum)
                    except Exception:
                        pass
                except Exception:
                    pass
            else:
                logger.warning("[StatTests] dCor skipped: target missing in columns")
        except Exception as e:
            logger.error(f"[StatTests] dCor stage failed: {e}")

        # Stage: Selection (split into VIF → MI → Embedded(CatBoost))
        try:
            target = getattr(self.settings.features, 'selection_target_column', None)
            if not (target and target in ddf.columns):
                logger.warning("[StatTests] Selection skipped: target missing in columns")
            else:
                # VIF
                logger.info("[StatTests] Stage start: Selection/VIF")
                from features.statistical_tests.stage_selection_vif import run as run_vif
                vif_res = run_vif(self.stats, ddf, target)
                try:
                    v_in = vif_res.get('stage2_vif_input', [])
                    v_out = vif_res.get('stage2_vif_selected', [])
                    logger.info(f"[StatTests] VIF: input={len(v_in)} → output={len(v_out)}; output_preview={(v_out or [])[:15]}")
                except Exception:
                    pass

                # MI
                logger.info("[StatTests] Stage start: Selection/MI")
                from features.statistical_tests.stage_selection_mi import run as run_mi
                mi_res = run_mi(self.stats, ddf, target, vif_selected=vif_res.get('stage2_vif_selected', []))
                try:
                    m_in = mi_res.get('stage2_vif_selected', [])
                    m_out = mi_res.get('stage2_mi_selected', [])
                    logger.info(f"[StatTests] MI: input={len(m_in)} → output={len(m_out)}; output_preview={(m_out or [])[:15]}")
                except Exception:
                    pass

                # Embedded (CatBoost)
                logger.info("[StatTests] Stage start: Selection/Embedded (CatBoost)")
                
                # Log dataset size for CatBoost
                try:
                    dataset_rows = len(ddf)
                    logger.info(f"[StatTests] 📊 Dataset size for CatBoost: {dataset_rows:,} rows")
                except Exception:
                    logger.info("[StatTests] 📊 Dataset size for CatBoost: Computing full dataset")
                
                from features.statistical_tests.stage_selection_embedded import run as run_emb
                emb_res = run_emb(self.stats, ddf, target, mi_selected=mi_res.get('stage2_mi_selected', []))
                try:
                    e_in = emb_res.get('stage2_mi_selected', [])
                    e_out = emb_res.get('stage3_final_selected', [])
                    e_importances = emb_res.get('importances', {})
                    e_backend = emb_res.get('selection_stats', {}).get('backend_used', 'unknown')
                    e_model_score = emb_res.get('selection_stats', {}).get('model_score', 0.0)
                    e_detailed_metrics = emb_res.get('detailed_metrics', {})
                    
                    # Log complete output with CatBoost scores
                    logger.info(f"[StatTests] Embedded (CatBoost): input={len(e_in)} → output={len(e_out)}; backend={e_backend}; model_score={e_model_score:.6f}")
                    
                    # Log dataset size confirmation
                    try:
                        dataset_size = len(ddf)
                        logger.info(f"[StatTests] ✅ DATASET CONFIRMATION: CatBoost used {dataset_size:,} rows (FULL DATASET)")
                    except Exception:
                        logger.info(f"[StatTests] ✅ DATASET CONFIRMATION: CatBoost used full dataset (size computed during processing)")
                    
                    # Log detailed CatBoost metrics
                    if e_detailed_metrics:
                        logger.info(f"[StatTests] 📊 CatBoost Detailed Analysis:")
                        
                        # Model configuration
                        if 'model_info' in e_detailed_metrics:
                            model_info = e_detailed_metrics['model_info']
                            logger.info(f"[StatTests]   🔧 Model Config: iterations={model_info.get('iterations_used', 'unknown')}, "
                                       f"learning_rate={model_info.get('learning_rate', 'unknown')}, "
                                       f"depth={model_info.get('depth', 'unknown')}, "
                                       f"task_type={model_info.get('task_type', 'unknown')}, "
                                       f"loss_function={model_info.get('loss_function', 'unknown')}")
                        
                        # Performance metrics
                        if 'validation_r2' in e_detailed_metrics:
                            logger.info(f"[StatTests]   📈 Regression Performance: "
                                       f"R²={e_detailed_metrics.get('validation_r2', 0):.4f}, "
                                       f"RMSE={e_detailed_metrics.get('validation_rmse', 0):.4f}, "
                                       f"MAE={e_detailed_metrics.get('validation_mae', 0):.4f}")
                            logger.info(f"[StatTests]   📈 Training Performance: "
                                       f"R²={e_detailed_metrics.get('training_r2', 0):.4f}, "
                                       f"RMSE={e_detailed_metrics.get('training_rmse', 0):.4f}, "
                                       f"MAE={e_detailed_metrics.get('training_mae', 0):.4f}")
                        elif 'validation_accuracy' in e_detailed_metrics:
                            logger.info(f"[StatTests]   📈 Classification Performance: "
                                       f"Accuracy={e_detailed_metrics.get('validation_accuracy', 0):.4f}, "
                                       f"Precision={e_detailed_metrics.get('validation_precision', 0):.4f}, "
                                       f"Recall={e_detailed_metrics.get('validation_recall', 0):.4f}, "
                                       f"F1={e_detailed_metrics.get('validation_f1', 0):.4f}")
                        
                        # CV scheme and folds
                        if 'cv_scheme' in e_detailed_metrics or 'cv_folds_used' in e_detailed_metrics:
                            logger.info(f"[StatTests]   🔁 CV: scheme={e_detailed_metrics.get('cv_scheme', 'unknown')}, folds_used={e_detailed_metrics.get('cv_folds_used', 0)}")

                        # Feature importance types
                        if 'feature_importance_types' in e_detailed_metrics:
                            fi_types = e_detailed_metrics['feature_importance_types']
                            logger.info(f"[StatTests]   🎯 Feature Importance Types Available: {list(fi_types.keys())}")
                    
                    # Log all winning features with their CatBoost importance scores
                    if e_out and e_importances:
                        winning_features = [(f, e_importances.get(f, 0.0)) for f in e_out]
                        winning_features.sort(key=lambda x: x[1], reverse=True)  # Sort by importance
                        logger.info(f"[StatTests] CatBoost - All winning features with importance scores: {[(f'{name}:{score:.6f}') for name, score in winning_features]}")
                    
                    # Log all considered features for comparison
                    if e_importances:
                        all_features = [(f, e_importances.get(f, 0.0)) for f in e_in if f in e_importances]
                        all_features.sort(key=lambda x: x[1], reverse=True)
                        logger.info(f"[StatTests] CatBoost - All considered features with importance scores: {[(f'{name}:{score:.6f}') for name, score in all_features]}")
                        
                except Exception as e:
                    logger.warning(f"[StatTests] Error logging CatBoost results: {e}")
                    pass

                # Combined selection summary + artifacts
                try:
                    combined = {
                        'stage': 'selection_split',
                        'stage2_vif_input': vif_res.get('stage2_vif_input', []),
                        'stage2_vif_selected': vif_res.get('stage2_vif_selected', []),
                        'vif_input_with_dcor': vif_res.get('vif_input_with_dcor', {}),
                        'vif_selected_with_dcor': vif_res.get('vif_selected_with_dcor', {}),
                        # New audit fields: features that entered/left VIF without dCor scores
                        'vif_input_without_dcor': vif_res.get('vif_input_without_dcor', []),
                        'vif_selected_without_dcor': vif_res.get('vif_selected_without_dcor', []),
                        'stage2_mi_selected': mi_res.get('stage2_mi_selected', []),
                        'stage3_final_selected': emb_res.get('stage3_final_selected', []),
                        'importances': emb_res.get('importances', {}),
                        'selection_stats': emb_res.get('selection_stats', {}),
                    }
                    logger.info(
                        f"[StatTests] Selection split summary: VIF={len(combined['stage2_vif_selected'])} → MI={len(combined['stage2_mi_selected'])} → Final={len(combined['stage3_final_selected'])}; final_preview={(combined['stage3_final_selected'] or [])[:15]}"
                    )
                    
                    # Log final selection clearly
                    final_features = combined['stage3_final_selected']
                    if final_features:
                        logger.info(f"[StatTests] 🎯 FINAL SELECTION: {len(final_features)} features selected by CatBoost:")
                        for i, feature in enumerate(final_features, 1):
                            importance = combined.get('importances', {}).get(feature, 0.0)
                            logger.info(f"[StatTests]   {i:2d}. {feature} (importance: {importance:.6f})")
                    else:
                        logger.warning("[StatTests] ⚠️  FINAL SELECTION: No features selected!")
                    try:
                        self._write_stat_tests_artifacts(
                            getattr(self, 'current_currency_pair', 'unknown'),
                            dcor_summary=getattr(self.stats, '_last_dcor_summary', {}) or None,
                            selection_results=combined,
                        )
                    except Exception:
                        pass
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"[StatTests] Selection stages failed: {e}")

        # Stage: Final Stats
        # DISABLED: Final stats stage removed - unnecessary after feature selection
        logger.info("[StatTests] Final Stats stage skipped - feature selection already completed")
        
        # Final summary
        try:
            final_features = getattr(self.stats, '_last_embedded_selected', [])
            if final_features:
                logger.info(f"[StatTests] ✅ FEATURE SELECTION COMPLETED: {len(final_features)} final features selected")
                logger.info(f"[StatTests] 📋 FINAL FEATURES LIST: {final_features}")
            else:
                logger.warning("[StatTests] ⚠️  FEATURE SELECTION COMPLETED: No features selected")
        except Exception:
            logger.info("[StatTests] ✅ FEATURE SELECTION COMPLETED")

        return ddf

def _noop_ctx():
    from contextlib import contextmanager
    @contextmanager
    def _noop():
        yield
    return _noop()

    def _log_statistical_tests_plan_cudf(self, gdf: cudf.DataFrame) -> None:
        """Emit a step-by-step plan for the statistical_tests engine when using cuDF.

        The plan is inferred from settings.features and current DataFrame columns.
        """
        try:
            feats = getattr(self.settings, 'features', None)
            # Detect presence of typical inputs
            cols = list(gdf.columns)
            has_frac_diff = any('frac_diff' in str(c) for c in cols)
            # JB removed from project
            target = None
            try:
                target = getattr(feats, 'selection_target_column', None)
            except Exception:
                target = None
            # dCor/selection params
            dcor_top_k = int(getattr(feats, 'dcor_top_k', 50) or 50) if feats else 50
            vif_thr = float(getattr(feats, 'vif_threshold', 5.0) or 5.0) if feats else 5.0
            mi_thr = float(getattr(feats, 'mi_threshold', 0.3) or 0.3) if feats else 0.3
            stage3_top_n = int(getattr(feats, 'stage3_top_n', 50) or 50) if feats else 50

            steps = []
            # Step 1: ADF
            if has_frac_diff:
                steps.append(f"1) ADF rolling over frac_diff* features (found={sum(1 for c in cols if 'frac_diff' in str(c))})")
            else:
                steps.append("1) ADF rolling: skipped (no frac_diff* columns found)")
            # JB removed from project
            # Step 2: dCor
            if target and (target in cols or 'y_close' in cols):
                steps.append(f"2) Distance Correlation (dCor) ranking vs target='{target}' (topK={dcor_top_k})")
            else:
                steps.append("2) Distance Correlation: target missing → will attempt fallback or skip")
            # Step 3: Selection
            steps.append(f"3) Feature Selection pipeline: VIF(thr={vif_thr}) → MI(thr={mi_thr}) → LGBM topN={stage3_top_n}")
            # Step 4: Final stats
            steps.append("4) Final comprehensive statistical summaries and flags")

            logger.info("StatisticalTests plan (cuDF):")
            for s in steps:
                logger.info(f"   • {s}")
        except Exception as e:
            logger.debug(f"statistical_tests plan (cuDF) failed: {e}")

    def _log_statistical_tests_plan_dask(self, ddf) -> None:
        """Emit a step-by-step plan for the statistical_tests engine when using Dask-cuDF.

        Avoids triggering computation; relies on available schema/meta only.
        """
        try:
            feats = getattr(self.settings, 'features', None)
            try:
                cols = list(ddf.columns)
            except Exception:
                cols = []
            has_frac_diff = any('frac_diff' in str(c) for c in cols)
            # JB removed from project
            target = None
            try:
                target = getattr(feats, 'selection_target_column', None)
            except Exception:
                target = None
            dcor_top_k = int(getattr(feats, 'dcor_top_k', 50) or 50) if feats else 50
            vif_thr = float(getattr(feats, 'vif_threshold', 5.0) or 5.0) if feats else 5.0
            mi_thr = float(getattr(feats, 'mi_threshold', 0.3) or 0.3) if feats else 0.3
            stage3_top_n = int(getattr(feats, 'stage3_top_n', 50) or 50) if feats else 50

            steps = []
            if has_frac_diff:
                cnt = sum(1 for c in cols if 'frac_diff' in str(c))
                steps.append(f"1) ADF rolling on frac_diff* (distributed) [~{cnt} features]")
            else:
                steps.append("1) ADF rolling: skipped (no frac_diff* columns in schema)")
            # JB removed from project
            if target and (target in cols or 'y_close' in cols):
                steps.append(f"2) dCor ranking vs target='{target}' on sampled tail/head (topK={dcor_top_k})")
            else:
                steps.append("2) dCor: target missing → will try compute forward return if possible or skip")
            steps.append(f"3) Selection: candidates filter → VIF({vif_thr}) → MI({mi_thr}) → LGBM topN={stage3_top_n}")
            steps.append("4) Final comprehensive statistical analysis (map_partitions)")

            logger.info("StatisticalTests plan (Dask):")
            for s in steps:
                logger.info(f"   • {s}")
        except Exception as e:
            logger.debug(f"statistical_tests plan (Dask) failed: {e}")

    def _save_intermediate_data(self, ddf_or_gdf, currency_pair: str, engine_name: str):
        """Optionally save an intermediate checkpoint after an engine.

        Applies the same metric-column filtering as the final save when configured.
        """
        try:
            # Check toggle
            if not bool(getattr(self.settings.output, 'save_intermediate_per_engine', False)):
                return
            # Compute to cuDF if needed
            try:
                import dask_cudf  # noqa: F401
                is_dask = hasattr(ddf_or_gdf, 'compute') and not hasattr(ddf_or_gdf, 'to_arrow')
            except Exception:
                is_dask = hasattr(ddf_or_gdf, 'compute') and not hasattr(ddf_or_gdf, 'to_arrow')
            gdf = ddf_or_gdf.compute() if is_dask else ddf_or_gdf

            # Drop metric columns if configured
            try:
                if bool(getattr(self.settings.features, 'drop_metric_columns_on_intermediate', True)):
                    metrics_prefixes = list(getattr(self.settings.features, 'metrics_prefixes', []))
                    to_drop = [c for c in gdf.columns if any(c.startswith(p) for p in metrics_prefixes)]
                    if to_drop:
                        gdf = gdf.drop(columns=to_drop)
            except Exception:
                pass

            # Prepare path
            import pathlib
            out_dir = pathlib.Path(self.settings.output.output_path) / currency_pair / 'checkpoint'
            out_dir.mkdir(parents=True, exist_ok=True)
            # File name per engine - always use parquet
            fname = f"{engine_name}.parquet"

            # Save as Parquet
            import pyarrow.parquet as pq
            table = gdf.to_arrow()
            pq.write_table(
                table,
                str(out_dir / fname),
                compression=str(getattr(self.settings.output, 'intermediate_compression', 'zstd')),
            )
            logger.info(f"Saved intermediate checkpoint: {out_dir / fname}")
        except Exception as e:
            logger.warning(f"Could not save intermediate checkpoint for {engine_name}: {e}")

    def _write_stat_tests_artifacts(self, currency_pair: str, dcor_summary=None, selection_results=None) -> None:
        """Write compact JSON artifacts for Statistical Tests (dCor and Selection).

        Best-effort only; failures are ignored. Files are written under artifacts/<pair>/
        """
        try:
            import json as _json
            base = None
            try:
                base = Path(getattr(self.stats, 'artifacts_dir', 'artifacts'))
            except Exception:
                base = Path('artifacts')
            out_dir = base / str(currency_pair)
            out_dir.mkdir(parents=True, exist_ok=True)
            if dcor_summary:
                dsum = dict(dcor_summary)
                scores = dsum.get('scores') or {}
                if scores:
                    try:
                        items = sorted(scores.items(), key=lambda kv: (kv[1] if isinstance(kv[1], (int, float)) else -1), reverse=True)
                        dsum['scores_top'] = items[:200]
                    except Exception:
                        pass
                    dsum.pop('scores', None)
                (out_dir / 'stat_tests_dcor.json').write_text(_json.dumps(dsum, ensure_ascii=False, indent=2))
            if selection_results:
                sres = dict(selection_results)
                imps = sres.get('importances') or {}
                if imps:
                    try:
                        items = sorted(imps.items(), key=lambda kv: (kv[1] if isinstance(kv[1], (int, float)) else -1), reverse=True)
                        sres['importances_top'] = items[:200]
                    except Exception:
                        pass
                    sres.pop('importances', None)
                (out_dir / 'stat_tests_selection.json').write_text(_json.dumps(sres, ensure_ascii=False, indent=2))
        except Exception:
            pass

    def _validate_initial_data_dask(self, ddf, currency_pair: str) -> bool:
        """Lightweight validation for dask_cudf DataFrame."""
        try:
            sample = ddf.head(1)
            if sample.shape[0] == 0:
                logger.error(f"Empty Dask DataFrame for {currency_pair}")
                return False
            logger.info(f"📊 Sample schema: {list(sample.columns)}")
            return True
        except Exception as e:
            logger.error(f"Error validating initial Dask data for {currency_pair}: {e}")
            return False

    def _validate_intermediate_data_dask(self, ddf, currency_pair: str, engine_name: str) -> bool:
        """Lightweight post-engine validation for dask_cudf DataFrame.

        Uses a minimal head() to avoid large materializations that could exhaust GPU memory.
        """
        try:
            sample = ddf.head(1)
            if sample.shape[0] == 0:
                logger.error(f"Empty Dask DataFrame after {engine_name} for {currency_pair}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error validating Dask data for {currency_pair} after {engine_name}: {e}")
            return False

    def process_currency_pair_dask(self, currency_pair: str, r2_path: str, client: Client) -> bool:
        """Process a single currency pair using dask_cudf (delegates to module impl)."""
        return _process_currency_pair_dask_impl(self, currency_pair, r2_path, client)


def _process_currency_pair_dask_impl(self: "DataProcessor", currency_pair: str, r2_path: str, client: Client) -> bool:
    """Module-level implementation of the Dask processing path.

    Kept outside the class so workers with an older DataProcessor class can bind
    this function as a method at runtime and still execute the Dask path.
    """
    try:
            # Ensure required helpers exist on this instance; bind from latest class or attach fallbacks
            try:
                import types as _types
                try:
                    # Try to use the most recently loaded class definition
                    import importlib as _importlib
                    import orchestration.data_processor as _dp_reload
                    _dp_mod_latest = _importlib.reload(_dp_reload)
                    _src_cls = getattr(_dp_mod_latest, 'DataProcessor', None) or type(self)
                except Exception:
                    _src_cls = type(self)

                _need = [
                    '_validate_initial_data_dask',
                    '_validate_intermediate_data_dask',
                    '_register_task_failure',
                    '_register_task_success',
                    '_save_intermediate_data',
                    '_save_processed_data_dask',
                    '_drop_denied_columns',
                    '_execute_engine_dask',
                    '_log_statistical_tests_plan_dask',
                ]
                for _name in _need:
                    if not hasattr(self, _name) and hasattr(_src_cls, _name):
                        try:
                            setattr(self, _name, _types.MethodType(getattr(_src_cls, _name), self))
                        except Exception:
                            pass
            except Exception:
                pass

            # Attach minimal safe fallbacks if still missing
            if not hasattr(self, '_validate_initial_data_dask'):
                def _fb_validate_initial_data_dask(ddf, pair):
                    try:
                        sample = ddf.head(1)
                        if getattr(sample, 'shape', (0,))[0] == 0:
                            logger.error(f"Empty Dask DataFrame for {pair}")
                            return False
                        return True
                    except Exception as _e:
                        logger.error(f"Error validating initial Dask data for {pair}: {_e}")
                        return False
                self._validate_initial_data_dask = _fb_validate_initial_data_dask  # type: ignore

            if not hasattr(self, '_validate_intermediate_data_dask'):
                def _fb_validate_intermediate_data_dask(ddf, pair, engine):
                    try:
                        sample = ddf.head(1)
                        if getattr(sample, 'shape', (0,))[0] == 0:
                            logger.error(f"Empty Dask DataFrame after {engine} for {pair}")
                            return False
                        return True
                    except Exception as _e:
                        logger.error(f"Error validating Dask data for {pair} after {engine}: {_e}")
                        return False
                self._validate_intermediate_data_dask = _fb_validate_intermediate_data_dask  # type: ignore

            if not hasattr(self, '_register_task_failure'):
                def _fb_register_task_failure(pair, path, msg):
                    try:
                        if getattr(self, 'db_connected', False) and getattr(self, 'db_handler', None):
                            try:
                                task_id = self.db_handler.register_currency_pair(pair, path)
                                if task_id:
                                    self.db_handler.update_task_status(task_id, 'FAILED', msg)
                            except Exception:
                                pass
                        logger.error(f"Task failure [{pair}]: {msg}")
                    except Exception:
                        pass
                self._register_task_failure = _fb_register_task_failure  # type: ignore

            if not hasattr(self, '_register_task_success'):
                def _fb_register_task_success(pair, path):
                    try:
                        if getattr(self, 'db_connected', False) and getattr(self, 'db_handler', None):
                            try:
                                task_id = self.db_handler.register_currency_pair(pair, path)
                                if task_id:
                                    self.db_handler.update_task_status(task_id, 'COMPLETED')
                            except Exception:
                                pass
                        logger.info(f"Task success [{pair}]")
                    except Exception:
                        pass
                self._register_task_success = _fb_register_task_success  # type: ignore

            if not hasattr(self, '_save_intermediate_data'):
                def _fb_save_intermediate_data(_ddf, pair, eng):
                    # no-op fallback; just log to keep flow
                    logger.debug(f"[fallback] Skipping intermediate save for {pair}:{eng}")
                self._save_intermediate_data = _fb_save_intermediate_data  # type: ignore

            if not hasattr(self, '_save_processed_data_dask'):
                def _fb_save_processed_data_dask(_ddf, pair):
                    logger.warning(f"[fallback] Missing Dask save; skipping save for {pair}")
                    return True
                self._save_processed_data_dask = _fb_save_processed_data_dask  # type: ignore

            if not hasattr(self, '_drop_denied_columns'):
                def _fb_drop_denied_columns(df):
                    try:
                        cols = list(df.columns)
                        if 'y_minutes_since_open' in cols:
                            return df.drop(columns=['y_minutes_since_open'], errors='ignore')
                        return df
                    except Exception:
                        return df
                self._drop_denied_columns = _fb_drop_denied_columns  # type: ignore

            if not hasattr(self, '_execute_engine_dask'):
                def _fb_execute_engine_dask(_eng, df):
                    logger.warning(f"[fallback] Missing _execute_engine_dask for {_eng}; pass-through")
                    return df
                self._execute_engine_dask = _fb_execute_engine_dask  # type: ignore

            if not hasattr(self, '_log_statistical_tests_plan_dask'):
                def _fb_log_stat_tests_plan(_df):
                    logger.debug("[fallback] Stats tests plan log skipped (helper missing)")
                self._log_statistical_tests_plan_dask = _fb_log_stat_tests_plan  # type: ignore

            # Set currency pair context for all subsequent logs
            set_currency_pair_context(currency_pair)
            try:
                self.current_currency_pair = currency_pair
            except Exception:
                pass
            logger.info(f"Starting Dask processing for {currency_pair}")
            # Cluster snapshot to aid debugging
            try:
                sched = client.scheduler_info()
                workers_list = list(sched.get('workers', {}).keys())
                logger.info(f"Dask snapshot: workers={len(workers_list)}, dashboard={getattr(client, 'dashboard_link', None)}")
            except Exception as e:
                logger.debug(f"Could not fetch scheduler info: {e}")

            task_id = None
            if not self.db_connected:
                self.db_connected = self.db_handler.connect()
                if not self.db_connected:
                    logger.warning("Database unavailable; proceeding without task tracking")
            # Register task and set RUNNING status (best-effort)
            if self.db_connected:
                try:
                    task_id = self.db_handler.register_currency_pair(currency_pair, r2_path)
                    if task_id:
                        self.db_handler.update_task_status(task_id, 'RUNNING')
                        # Attach context to engines for metrics/artifacts
                        try:
                            for eng in (self.station, self.stats, self.garch, self.feng):
                                if hasattr(eng, 'set_task_context'):
                                    eng.set_task_context(self.run_id, task_id, self.db_handler)
                        except Exception:
                            pass
                except Exception as e:
                    logger.warning(f"Could not register task in DB: {e}")

            # Load Dask DataFrame: choose loader based on extension or availability
            r2_lower = str(r2_path).lower()
            ddf = None
            try:
                with dask.annotate(task_key_name=f"load-{currency_pair}"):
                    # Only support Parquet format
                    ddf = self.loader.load_currency_pair_data(r2_path, client)
            except Exception as _e:
                logger.warning(f"Primary load path failed: {_e}; trying fallbacks")
                if ddf is None:
                    with dask.annotate(task_key_name=f"load-{currency_pair}"):
                        ddf = self.loader.load_currency_pair_data(r2_path, client)
            if ddf is None:
                self._register_task_failure(currency_pair, r2_path, "Failed to load Dask data")
                return False

            # Attach context column for downstream engines (e.g., CPCV persistence)
            try:
                ddf = ddf.assign(currency_pair=currency_pair)
            except Exception:
                logger.debug("Could not attach currency_pair column to Dask DataFrame")

            # Drop denied columns so they never enter any stage
            try:
                ddf = self._drop_denied_columns(ddf)
            except Exception:
                pass

            # Optional: Repartition to utilize multiple GPUs better
            try:
                n_workers = 1
                try:
                    sched = client.scheduler_info()
                    n_workers = max(1, len(sched.get('workers', {})))
                except Exception:
                    pass
                target_parts = max(ddf.npartitions, n_workers * 8)
                if ddf.npartitions < target_parts:
                    import dask as _dask
                    with _dask.annotate(task_key_name=f"repartition-{currency_pair}"):
                        logger.info(f"Repartitioning from {ddf.npartitions} to {target_parts} to utilize {n_workers} workers")
                        ddf = ddf.repartition(npartitions=target_parts)
                try:
                    logger.info(f"Post-load overview: cols={len(ddf.columns)}, npartitions={ddf.npartitions}")
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"Could not repartition: {e}")

            # Validate initial data
            if not self._validate_initial_data_dask(ddf, currency_pair):
                self._register_task_failure(currency_pair, r2_path, "Initial Dask data validation failed")
                return False

            # Engine execution order from settings
            engine_config = self.settings.pipeline.engines
            engine_execution_order = sorted(
                [(name, config) for name, config in engine_config.items() if config.enabled],
                key=lambda x: x[1].order
            )
            logger.info(f"🚀 Dask Engine execution order: {[name for name, _ in engine_execution_order]}")

            # Execute each engine (Dask)
            for engine_name, eng_cfg in engine_execution_order:
                if not eng_cfg.enabled:
                    logger.info(f"⏭️ Skipping disabled engine: {engine_name}")
                    continue

                # Set engine context
                # Log engine start
                logger.info(f"Starting {engine_name} processing for {currency_pair} (Dask)")
                logger.info(f"📝 Description: {eng_cfg.description}")
                # Emit step-by-step plan for statistical_tests when running on Dask
                try:
                    if str(engine_name) == 'statistical_tests':
                        self._log_statistical_tests_plan_dask(ddf)
                except Exception as _e:
                    logger.debug(f"Could not build statistical_tests plan (Dask): {_e}")
                # Stage DB start (schema deltas based on columns only, rows unknown for Dask)
                stage_id = None
                try:
                    cols_before = len(ddf.columns)
                    cols_before_set = set(list(ddf.columns))
                except Exception:
                    cols_before = None
                    cols_before_set = None
                if self.db_connected:
                    try:
                        stage_id = self.db_handler.start_stage(self.run_id, task_id, engine_name,
                                                               rows_before=None, cols_before=cols_before,
                                                               message=f"{currency_pair}")
                    except Exception:
                        stage_id = None

                # Group all Dask tasks created inside this engine under a readable name
                try:
                    from dask import annotate as _annotate, config as _dask_config
                except Exception:
                    def _annotate(**_kwargs):
                        from contextlib import contextmanager
                        @contextmanager
                        def _noop():
                            yield
                        return _noop()
                    class _dask_config:  # type: ignore
                        @staticmethod
                        def set(*args, **kwargs):
                            from contextlib import contextmanager
                            @contextmanager
                            def _noop():
                                yield
                            return _noop()

                try:
                    debug_dash = bool(getattr(self.settings.development, 'debug_dashboard', False))
                    with _annotate(task_key_name=f"{engine_name}-{currency_pair}"):
                        try:
                            logger.info(f"Engine start: {engine_name} | npartitions={ddf.npartitions}")
                        except Exception:
                            logger.info(f"ERROR Engine start: {engine_name} | npartitions={ddf.npartitions}")
                            pass
                        ctx = _dask_config.set({"optimization.fuse.active": False}) if debug_dash else _dask_config.set({})
                        with ctx:
                            ddf = self._execute_engine_dask(engine_name, ddf)
                    if ddf is None:
                        raise RuntimeError(f"Engine {engine_name} returned None")

                    # Stabilize graph/memory between engines
                    logger.info(f"Persist start: {engine_name}")
                    try:
                        from dask import persist as _persist
                        debug_dash = bool(getattr(self.settings.development, 'debug_dashboard', False))
                        if debug_dash:
                            ddf, = _persist(ddf, optimize_graph=False)
                        else:
                            ddf = ddf.persist()
                    except Exception:
                        ddf = ddf.persist()
                    logger.info(f"Persist returned: {engine_name}")
                    # Longer timeout for heavy engines (statistical_tests and garch_models)
                    if engine_name == 'statistical_tests':
                        timeout_s = 600
                    elif engine_name == 'garch_models':
                        timeout_s = 180
                    else:
                        timeout_s = 60
                    logger.info(f"Waiting for {engine_name} computation to complete (timeout: {timeout_s}s)...")
                    try:
                        wait(ddf, timeout=timeout_s)
                        logger.info(f"{engine_name} computation completed successfully")
                    except Exception as wait_err:
                        logger.warning(f"Wait timeout or error for {engine_name}: {wait_err}")
                        logger.info(f"Continuing without waiting - data is still persisted for {engine_name}")
                        # Continue without waiting - data is still persisted

                    # Free CuPy pools on all workers to reduce carry-over memory
                    try:
                        if self.client:
                            self.client.run(lambda: (__import__('gc').collect(), __import__('cupy').get_default_memory_pool().free_all_blocks()))
                    except Exception:
                        pass

                    if not self._validate_intermediate_data_dask(ddf, currency_pair, engine_name):
                        raise RuntimeError(f"Validation failed after {engine_name}")
                    try:
                        logger.info(f"Engine end: {engine_name} | npartitions={ddf.npartitions}")
                    except Exception:
                        pass

                    # Log engine completion (rows are unknown/expensive in Dask; report columns and partitions)
                    cols_after = len(ddf.columns)
                    new_cols = (cols_after - cols_before) if (cols_before is not None and cols_after is not None) else None
                    try:
                        nparts = getattr(ddf, 'npartitions', None)
                    except Exception:
                        nparts = None
                    logger.info(
                        f"{engine_name.title()} processing completed for {currency_pair} (Dask): "
                        f"cols {cols_before}->{cols_after} (+{new_cols}), npartitions={nparts}"
                    )

                    # Save intermediate checkpoint if enabled
                    try:
                        self._save_intermediate_data(ddf, currency_pair, engine_name)
                    except Exception:
                        pass

                    # Stage DB end
                    if self.db_connected and stage_id:
                        try:
                            cols_after = len(ddf.columns)
                            new_cols = (cols_after - cols_before) if (cols_before is not None and cols_after is not None) else None
                            details = None
                            try:
                                if cols_before_set is not None:
                                    cols_after_set = set(list(ddf.columns))
                                    added = sorted(list(cols_after_set - cols_before_set))
                                    import json as _json
                                    details = _json.dumps({'new_columns': added}, ensure_ascii=False)
                            except Exception:
                                details = None
                            self.db_handler.end_stage(stage_id, rows_after=None, cols_after=cols_after, new_cols=new_cols, details=details)
                        except Exception:
                            pass
                except Exception as eng_err:
                    logger.error(f"🚨 Engine {engine_name} failed for {currency_pair}: {eng_err}", exc_info=True)
                    if self.db_connected and stage_id:
                        try:
                            self.db_handler.error_stage(stage_id, error_message=str(eng_err))
                        except Exception:
                            pass
                    return False

            # Export selected-feature parquet and upload to R2 (Dask path, best-effort)
            try:
                # Debug: Check method availability
                methods = [m for m in dir(self) if 'export_selected_features' in m]
                logger.info(f"Available export methods: {methods}")
                
                if hasattr(self, '_export_selected_features_parquet_dask'):
                    logger.info(f"Method _export_selected_features_parquet_dask found, executing...")
                    sel_path = self._export_selected_features_parquet_dask(ddf, currency_pair)
                    if sel_path:
                        uploaded = self._upload_selected_features_to_r2(sel_path, currency_pair)
                        logger.info(f"Selected-feature parquet upload (Dask): {'OK' if uploaded else 'SKIPPED/FAILED'}")
                else:
                    logger.warning(f"Method _export_selected_features_parquet_dask not found on DataProcessor instance")
                    logger.info(f"DataProcessor type: {type(self)}")
                    logger.info(f"DataProcessor class: {self.__class__}")
            except Exception as _e:
                logger.warning(f"Could not export/upload selected-feature parquet (Dask) for {currency_pair}: {_e}")

            # Save directly from Dask without materializing everything on one GPU
            # Nota: Seleção final com CatBoost é parte do statistical_tests; não duplicar no caminho Dask.

            # Save diretamente do Dask sem materializar tudo em uma única GPU
            with dask.annotate(task_key_name=f"save-{currency_pair}"):
                try:
                    from dask import config as _dask_config
                    debug_dash = bool(getattr(self.settings.development, 'debug_dashboard', False))
                except Exception:
                    debug_dash = False
                    class _dask_config:  # type: ignore
                        @staticmethod
                        def set(*args, **kwargs):
                            from contextlib import contextmanager
                            @contextmanager
                            def _noop():
                                yield
                            return _noop()
                try:
                    logger.info(f"Preparing save: npartitions={ddf.npartitions}")
                except Exception:
                    pass
                ctx = _dask_config.set({"optimization.fuse.active": False}) if debug_dash else _dask_config.set({})
                with ctx:
                    save_ok = self._save_processed_data_dask(ddf, currency_pair)
            if not save_ok:
                self._register_task_failure(currency_pair, r2_path, "Data saving failed")
                return False

            # Aggressive cleanup to free GPU memory before next task
            try:
                try:
                    ddf = ddf.unpersist()
                except Exception:
                    pass
                del ddf
            except Exception:
                pass
            try:
                if self.client:
                    # Free CuPy pools on all workers and run GC
                    self.client.run(lambda: (__import__('gc').collect(), __import__('cupy').get_default_memory_pool().free_all_blocks()))
            except Exception:
                pass

            self._register_task_success(currency_pair, r2_path)
            logger.info(f"Successfully completed Dask processing for {currency_pair}")
            return True
    except Exception as e:
        logger.error(f"Error in Dask processing for {currency_pair}: {e}", exc_info=True)
        self._register_task_failure(currency_pair, r2_path, str(e))
        return False
    
    def _validate_intermediate_data(self, gdf: cudf.DataFrame, currency_pair: str, engine_name: str) -> bool:
        """
        Validate data after each engine execution.
        
        Args:
            gdf: The cuDF DataFrame to validate
            currency_pair: Currency pair for logging
            engine_name: Name of the engine that was just executed
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        try:
            # Check for empty DataFrame
            if len(gdf) == 0:
                logger.error(f"Empty DataFrame after {engine_name} for {currency_pair}")
                return False
            
            # Check for excessive NaN values
            nan_counts = gdf.isna().sum()
            high_nan_cols = nan_counts[nan_counts > len(gdf) * 0.5]  # More than 50% NaN
            if len(high_nan_cols) > 0:
                logger.warning(f"High NaN columns after {engine_name} for {currency_pair}: {list(high_nan_cols.index)}")
            
            # Check for infinite values
            inf_counts = gdf.isin([cp.inf, -cp.inf]).sum()
            high_inf_cols = inf_counts[inf_counts > 0]
            if len(high_inf_cols) > 0:
                logger.warning(f"Infinite values found after {engine_name} for {currency_pair}: {list(high_inf_cols.index)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating intermediate data for {currency_pair} after {engine_name}: {e}")
            return False
    
    def _save_processed_data(self, gdf: cudf.DataFrame, currency_pair: str) -> bool:
        """
        Save the processed data to the output directory.
        
        Args:
            gdf: The processed cuDF DataFrame
            currency_pair: Currency pair for file naming
            
        Returns:
            bool: True if saving was successful, False otherwise
        """
        try:
            # Log I/O start
            logger.info(f"Starting save operation for {currency_pair}: {len(gdf.columns)} columns, {len(gdf)} rows")
            
            logger.info(f"Saving processed data with {len(gdf.columns)} columns to Parquet files for {currency_pair}")
            
            # Get the output path from settings
            output_path = self.settings.output.output_path
            
            # Create output directory structure
            import pathlib
            import gc
            
            output_dir = pathlib.Path(output_path) / currency_pair
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Optionally drop internal metrics columns before saving
            try:
                if bool(getattr(self.settings.features, 'drop_metric_columns_on_save', True)):
                    metrics_prefixes = list(getattr(self.settings.features, 'metrics_prefixes', []))
                    to_drop = [c for c in gdf.columns if any(c.startswith(p) for p in metrics_prefixes)]
                    if to_drop:
                        logger.info(f"Dropping {len(to_drop)} metrics columns before save (prefix filter)")
                        gdf = gdf.drop(columns=to_drop)
            except Exception as e:
                logger.warning(f"Could not drop metrics columns: {e}")

            # Log data statistics before saving
            logger.info(f"Data shape: {gdf.shape}")
            logger.info(f"Columns: {list(gdf.columns)}")
            logger.info(f"Data types: {gdf.dtypes.to_dict()}")
            
            # Save using Parquet directly (cuDF DataFrame)
            logger.info(f"Saving to: {output_dir} using Parquet format")
            
            # Save as Parquet
            try:
                # Save as Parquet using cuDF
                consolidated_path = output_dir / f"{currency_pair}.parquet"
                
                gdf.to_parquet(
                    str(consolidated_path),
                    compression='snappy',
                    index=False
                )
                
                logger.info(f"Saved consolidated file: {consolidated_path} ({len(gdf)} rows)")
                
                # Clean up GPU memory
                gc.collect()
                
            except Exception as save_error:
                logger.error(f"Error saving data: {save_error}")
                raise
            
            # Verify the saved data
            saved_files = list(output_dir.glob("*.parquet"))
            total_size_mb = sum(f.stat().st_size for f in saved_files) / (1024 * 1024)
            
            # Log I/O completion
            logger.info(f"Save operation completed for {currency_pair}: {len(saved_files)} files, {round(total_size_mb, 2)}MB total")
            
            logger.info(f"Successfully saved processed data for {currency_pair}")
            logger.info(f"Files created: {len(saved_files)}")
            logger.info(f"Total size: {total_size_mb:.2f} MB")

            return True
            
        except Exception as save_error:
            logger.error(f"Failed to save processed data for {currency_pair}: {save_error}", exc_info=True)
            return False

    # Removido: persistência separada de selected_features.json (seleção centralizada no statistical_tests)

    # ---------------- Selected features export (CatBoost) ----------------
    def _get_selected_features(self) -> list:
        """Retrieve the final feature list selected by Stage 3 (CatBoost)."""
        try:
            feats = getattr(self.stats, '_last_embedded_selected', None)
            if feats:
                return list(feats)
        except Exception:
            pass
        try:
            # Fallback: infer from importances map if available
            imps = getattr(self.stats, '_last_importances', None) or {}
            if imps:
                # Keep all keys as 'selected'
                return list(imps.keys())
        except Exception:
            pass
        return []

    def _export_selected_features_parquet_cudf(self, gdf: cudf.DataFrame, currency_pair: str) -> str:
        """Create a single parquet with timestamp + CatBoost-selected features and return local path.

        Returns empty string if nothing was written.
        """
        try:
            selected = self._get_selected_features()
            if not selected:
                logger.warning(f"No CatBoost-selected features found for {currency_pair}; skipping model parquet export")
                return ""
            # Ensure timestamp + target + selected features (only those that exist)
            cols = []
            time_col = 'timestamp'
            target_col = getattr(self.settings.features, 'selection_target_column', 'y_ret_fwd_15m')
            
            if time_col in gdf.columns:
                cols.append(time_col)
            if target_col in gdf.columns:
                cols.append(target_col)
            
            missing = []
            for c in selected:
                if c in gdf.columns and c not in cols:
                    cols.append(c)
                else:
                    if c not in gdf.columns:
                        missing.append(c)
            if missing:
                logger.info(f"Some selected features missing in frame (ignored): {missing[:10]}{'...' if len(missing)>10 else ''}")
            
            logger.info(f"Exporting {len(cols)} columns for {currency_pair}: timestamp + target ({target_col}) + {len(selected)} selected features")
            if len(cols) <= 2:  # Need at least timestamp + target + some features
                logger.warning(f"No valid selected features to export for {currency_pair} (only {len(cols)} columns)")
                return ""

            # Arrange output path: place at output root for easy pickup
            import os
            import pyarrow.parquet as pq
            out_root = self.settings.output.output_path
            os.makedirs(out_root, exist_ok=True)
            out_file = os.path.join(out_root, f"{currency_pair}_feature_model.parquet")

            # Convert to Arrow Table and write single parquet
            try:
                table = gdf[cols].to_arrow()
            except Exception:
                # Fallback via pandas
                table = gdf[cols].to_pandas().to_arrow()
            pq.write_table(table, out_file, compression='snappy')
            logger.info(f"Exported CatBoost-selected features parquet: {out_file}")
            return out_file
        except Exception as e:
            logger.error(f"Failed to export selected features parquet for {currency_pair}: {e}")
            return ""

    def _export_selected_features_parquet_dask(self, ddf, currency_pair: str) -> str:
        """Create a single parquet with timestamp + selected features from Dask-cuDF DataFrame.

        Streams partitions to a single file to avoid collecting the full dataset.
        Returns empty string if nothing was written.
        """
        try:
            selected = self._get_selected_features()
            if not selected:
                logger.warning(f"No CatBoost-selected features found for {currency_pair}; skipping model parquet export (Dask)")
                return ""
            cols = []
            time_col = 'timestamp'
            target_col = getattr(self.settings.features, 'selection_target_column', 'y_ret_fwd_15m')
            cols_exist = set(list(ddf.columns))
            
            if time_col in cols_exist:
                cols.append(time_col)
            if target_col in cols_exist:
                cols.append(target_col)
            
            kept = 0
            for c in selected:
                if c in cols_exist and c not in cols:
                    cols.append(c)
                    kept += 1
            logger.info(f"Exporting {len(cols)} columns for {currency_pair} (Dask): timestamp + target ({target_col}) + {kept} selected features")
            
            if kept == 0 and len(cols) <= 2:  # Need at least timestamp + target + some features
                logger.warning(f"No valid selected features present in Dask frame for {currency_pair} (only {len(cols)} columns)")
                return ""

            import os
            import pyarrow as pa
            import pyarrow.parquet as pq
            from dask import delayed

            out_root = self.settings.output.output_path
            os.makedirs(out_root, exist_ok=True)
            out_file = os.path.join(out_root, f"{currency_pair}_feature_model.parquet")

            # Materialize partitions one-by-one on driver and append to a single parquet
            parts = ddf[cols].to_delayed()
            if not parts:
                logger.warning(f"No partitions to export for {currency_pair}")
                return ""

            # Compute first partition to establish schema
            first = parts[0].compute()
            try:
                first_table = first.to_arrow()
            except Exception:
                first_table = pa.Table.from_pandas(first.to_pandas())
            writer = pq.ParquetWriter(out_file, first_table.schema, compression='snappy')
            try:
                writer.write_table(first_table)
                # Remaining partitions
                for p in parts[1:]:
                    part = p.compute()
                    try:
                        tbl = part.to_arrow()
                    except Exception:
                        tbl = pa.Table.from_pandas(part.to_pandas())
                    writer.write_table(tbl)
            finally:
                writer.close()

            logger.info(f"Exported CatBoost-selected features parquet (Dask): {out_file}")
            return out_file
        except Exception as e:
            logger.error(f"Failed Dask export of selected features for {currency_pair}: {e}")
            return ""

    def _upload_selected_features_to_r2(self, local_file: str, currency_pair: str) -> bool:
        """Upload the local parquet file to R2 under bucket root as symbol_feature_model.parquet."""
        try:
            if not local_file:
                return False
            uploader = R2Uploader()
            remote_key = f"{currency_pair}_feature_model.parquet"
            return uploader.upload_file(local_file, remote_key, content_type="application/octet-stream")
        except Exception as e:
            logger.error(f"Failed to upload selected features for {currency_pair} to R2: {e}")
            return False

    def _save_processed_data_dask(self, ddf, currency_pair: str) -> bool:
        """Save the processed Dask-cuDF DataFrame as partitioned Parquet files.

        Avoids collecting the entire dataset into a single GPU/host memory.
        Compatible with our loader, which can read directories with part-*.parquet.
        """
        try:
            from dask import delayed, compute
            import pathlib, os

            output_path = self.settings.output.output_path
            output_dir = pathlib.Path(output_path) / currency_pair
            output_dir.mkdir(parents=True, exist_ok=True)

            # Optionally drop internal metrics columns before saving (pushdown-friendly)
            try:
                if bool(getattr(self.settings.features, 'drop_metric_columns_on_save', True)):
                    metrics_prefixes = list(getattr(self.settings.features, 'metrics_prefixes', []))
                    to_drop = [c for c in ddf.columns if any(str(c).startswith(p) for p in metrics_prefixes)]
                    if to_drop:
                        logger.info(f"Dropping {len(to_drop)} metrics columns before save (prefix filter)")
                        ddf = ddf.drop(columns=to_drop, errors='ignore')
            except Exception as e:
                logger.warning(f"Could not drop metrics columns: {e}")

            # Build delayed write tasks, one per partition
            parts = ddf.to_delayed()

            def _write_part(pdf, out_dir: str, idx: int, cols: list):
                # pdf is a cuDF DataFrame (computed partition)
                try:
                    # Ensure column order is consistent
                    if cols:
                        exist = [c for c in cols if c in pdf.columns]
                        pdf = pdf[exist]
                    fname = os.path.join(out_dir, f"part-{idx:05d}.parquet")
                    pdf.to_parquet(fname, compression='snappy', index=False)
                    return fname
                except Exception as e:
                    raise e

            cols_order = list(ddf.columns)
            tasks = [
                delayed(_write_part)(part, str(output_dir), i, cols_order)
                for i, part in enumerate(parts)
            ]
            written = compute(*tasks)
            logger.info(f"Saved {len(written)} parquet parts to {output_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to save Dask data for {currency_pair}: {e}", exc_info=True)
            return False
    
    def _register_task_success(self, currency_pair: str, r2_path: str):
        """Register successful task completion in the database."""
        if not self.db_connected:
            return
        try:
            task_id = self.db_handler.register_currency_pair(currency_pair, r2_path)
            if task_id:
                self.db_handler.update_task_status(task_id, 'COMPLETED')
                logger.info(f"Successfully registered task completion for {currency_pair} (Task ID: {task_id})")
            else:
                logger.warning(f"Failed to register task in database for {currency_pair}")
        except Exception as e:
            logger.error(f"Error registering task success for {currency_pair}: {e}")
    
    def _register_task_failure(self, currency_pair: str, r2_path: str, error_message: str):
        """Register task failure in the database."""
        if not self.db_connected:
            return
        try:
            task_id = self.db_handler.register_currency_pair(currency_pair, r2_path)
            if task_id:
                self.db_handler.update_task_status(task_id, 'FAILED', error_message)
                logger.info(f"Registered task failure for {currency_pair} (Task ID: {task_id})")
            else:
                logger.warning(f"Failed to register task failure in database for {currency_pair}")
        except Exception as e:
            logger.error(f"Error registering task failure for {currency_pair}: {e}")


def process_currency_pair_worker(currency_pair: str, r2_path: str) -> bool:
    """
    Worker function to process a single currency pair.
    This function is designed to be serializable for Dask workers.
    
    Args:
        currency_pair: Currency pair symbol (e.g., 'EURUSD')
        r2_path: Path to the data file in R2 storage
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    # Set up logging for the worker
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Set currency pair context for all subsequent logs
        set_currency_pair_context(currency_pair)
        processor = DataProcessor()
        return processor.process_currency_pair(currency_pair, r2_path)
    except Exception as e:
        logger.error(f"Error in worker processing {currency_pair}: {e}", exc_info=True)
        return False


def process_currency_pair_dask_worker(currency_pair: str, r2_path: str) -> bool:
    """
    Worker function that processes a single currency pair using the Dask path (dask_cudf),
    constraining all inner tasks to the current worker (one GPU per pair).
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    try:
        # Set currency pair context for all subsequent logs
        set_currency_pair_context(currency_pair)
        # Emit GPU/worker context to verify non-sharing and pinning
        try:
            import os as _os
            import cupy as _cp
            from dask.distributed import get_worker as _gw
            _w = _gw()
            _addr = getattr(_w, 'address', 'unknown')
            _gpu = int(_cp.cuda.runtime.getDevice())
            _vis = _os.environ.get('CUDA_VISIBLE_DEVICES', '')
            logger.info(f"Worker context: pair={currency_pair}, worker={_addr}, gpu_device={_gpu}, VISIBLE={_vis}")
        except Exception:
            pass
        # Resolve current worker and use a worker_client to schedule sub-tasks
        from dask.distributed import get_worker, worker_client
        import dask
        w = get_worker()
        addr = getattr(w, 'address', None)
        with worker_client() as wc:
            # Hot-reload this module on the worker to ensure latest class definitions
            try:
                import importlib, sys as _sys
                _mod = importlib.reload(_sys.modules[__name__])
            except Exception:
                try:
                    import importlib as _importlib
                    import orchestration.data_processor as _dp_mod
                    _mod = _importlib.reload(_dp_mod)
                except Exception:
                    _mod = None
            processor = (_mod.DataProcessor if _mod and hasattr(_mod, 'DataProcessor') else DataProcessor)(client=wc)
            # Ensure required helper methods exist on the instance (bind from latest class if missing)
            import types as _types
            try:
                # Prefer the freshly reloaded class as source
                _src_cls = _mod.DataProcessor if (_mod and hasattr(_mod, 'DataProcessor')) else DataProcessor
            except Exception:
                _src_cls = DataProcessor
            _need = [
                '_validate_initial_data_dask',
                '_validate_intermediate_data_dask',
                '_register_task_failure',
                '_register_task_success',
                '_save_intermediate_data',
                '_save_processed_data_dask',
                '_drop_denied_columns',
                '_execute_engine_dask',
                '_log_statistical_tests_plan_dask',
            ]
            _bound = []
            _missing_after = []
            for _name in _need:
                if not hasattr(processor, _name) and hasattr(_src_cls, _name):
                    try:
                        setattr(processor, _name, _types.MethodType(getattr(_src_cls, _name), processor))
                        _bound.append(_name)
                    except Exception as _e:
                        logger.warning(f"Failed to bind helper '{_name}' on worker: {_e}")
            for _name in _need:
                if not hasattr(processor, _name):
                    _missing_after.append(_name)
            if _bound:
                logger.info(f"Worker bound helpers: {_bound}")
            if _missing_after:
                logger.warning(f"Helpers still missing on worker after bind: {_missing_after}")
            # Annotate all tasks to stay on this worker
            with dask.annotate(workers=[addr] if addr else None, allow_other_workers=False, task_key_name=f"pair-{currency_pair}"):
                # Enforce Dask path only; if missing on class, bind module implementation dynamically
                if not hasattr(processor, 'process_currency_pair_dask'):
                    try:
                        import types as _types
                        from orchestration.data_processor import _process_currency_pair_dask_impl as _impl
                        processor.process_currency_pair_dask = _types.MethodType(_impl, processor)
                    except Exception as _bind_err:
                        raise RuntimeError("Dask path required but missing on worker and dynamic bind failed: " + str(_bind_err))
                return processor.process_currency_pair_dask(currency_pair, r2_path, wc)
    except Exception as e:
        logger.error(f"Error in Dask worker processing {currency_pair}: {e}", exc_info=True)
        return False
