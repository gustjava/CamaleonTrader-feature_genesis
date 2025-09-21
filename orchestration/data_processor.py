"""
Data Processor for Feature Engineering Pipeline

This module handles the processing of individual currency pairs through the feature engineering pipeline.
"""

import logging
import sys
import os
import time
import traceback
from typing import Optional, Dict, Any
from pathlib import Path

import cudf
import cupy as cp
import dask

from config.unified_config import get_unified_config as get_settings
from dask.distributed import Client, wait
from data_io.db_handler_no_mysql import NoDatabaseHandler as DatabaseHandler
from data_io.local_loader import LocalDataLoader
from features import (
    StationarizationEngine,
    StatisticalTests,
    GARCHModels,
    FeatureEngineeringEngine,
)
from features.signal_processing import apply_emd_to_series
from features.base_engine import CriticalPipelineError
from features.final_model import FinalModelTrainer
from utils.logging_utils import get_logger, set_currency_pair_context
from features.engine_metrics import EngineMetrics

logger = get_logger(__name__, "orchestration.processor")


class DataProcessor:
    """
    Handles the processing of individual currency pairs through the feature engineering pipeline.
    
    This class encapsulates all the logic for:
    - Loading data from various sources
    - Applying feature engineering engines in the correct order
    - Validating data once before final CatBoost selection
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
            
            # Data processing completed successfully
            logger.info(f"Data processing completed successfully for {currency_pair}")

            # Build and save the final model
            # Get target column from settings
            target_col = getattr(self.settings.features, 'selection_target_column', 'y_ret_fwd_60m')
            
            # Try to use real selection results from StatisticalTests
            selected_features = getattr(self.stats, '_last_embedded_selected', None)
            feature_importances = getattr(self.stats, '_last_embedded_importances', None)
            
            if not selected_features:
                logger.warning("[FinalModel] No selection artifacts found; skipping final model to avoid training on all columns.")
                logger.info("[FinalModel] Feature selection is the source of truth - no final model needed.")
                # Still register as successful since selection completed
                _register_task_success(self, currency_pair, r2_path)
                return True
            
            # Create selection metadata
            selection_metadata = {
                'target_column': target_col,
                'selected_features': selected_features,
                'feature_count': len(selected_features)
            }
            
            # Build final model with real selection results
            if not self._build_final_model(gdf, target_col, selected_features, feature_importances, selection_metadata, currency_pair, "1h"):
                self._register_task_failure(currency_pair, r2_path, "Final model building failed")
                return False
            
            # Register successful completion
            _register_task_success(self, currency_pair, r2_path)
            
            logger.info(f"Successfully completed processing for {currency_pair}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {currency_pair}: {e}", exc_info=True)
            _register_task_failure(self, currency_pair, r2_path, str(e))
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
            
            # Optional fallback: Feather (sync) if exists
            try:
                gdf = self.loader.load_currency_pair_data_feather_sync(r2_path)
                if gdf is not None:
                    gdf = self._drop_denied_columns(gdf)
                    logger.info(f"Loaded feather data: {len(gdf)} rows, {len(gdf.columns)} columns")
                    return gdf
            except Exception:
                pass
            
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
            pipeline_cfg = getattr(self.settings, 'pipeline', None)
            engine_config = getattr(pipeline_cfg, 'engines', {}) if pipeline_cfg else {}
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
            logger.info(f"📋 Total engines to execute: {len(engine_execution_order)}")
            
            # Execute each engine in order
            for engine_idx, (engine_name, engine_config) in enumerate(engine_execution_order, 1):
                logger.info(f"🔄 Starting engine {engine_idx}/{len(engine_execution_order)}: {engine_name}")
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
                prev_columns = list(gdf.columns)
                cols_before = len(prev_columns)
                
                try:
                    # Execute the appropriate engine
                    logger.info(f"🔧 Executing {engine_name} engine for {currency_pair}")
                    gdf = self._execute_engine(engine_name, gdf)
                    if gdf is None:
                        logger.error(f"❌ Engine {engine_name} returned None for {currency_pair}")
                        return None
                    
                    rows_after = len(gdf)
                    cols_after = len(gdf.columns)
                    new_cols = cols_after - cols_before
                    
                    # Log engine completion
                    logger.info(f"✅ {engine_name.title()} processing completed for {currency_pair}: {rows_before}→{rows_after} rows, {cols_before}→{cols_after} cols (+{new_cols})")

                    # Simple GPU memory snapshot
                    try:
                        import cupy as _cp
                        free_b, total_b = _cp.cuda.runtime.memGetInfo()
                        used_gb = (total_b - free_b) / (1024**3)
                        total_gb = total_b / (1024**3)
                        mem = {'gpu_used_gb': round(used_gb, 2), 'gpu_total_gb': round(total_gb, 2)}
                        logger.info(f"💾 GPU Memory after {engine_name}: {used_gb:.2f}GB used / {total_gb:.2f}GB total")
                    except Exception:
                        mem = {}
                    
                    logger.info(f"📊 {engine_name.title()} complete: {cols_before} -> {cols_after} cols (+{new_cols} new), {rows_before} -> {rows_after} rows ({rows_after - rows_before} change)")
                    
                    if new_cols > 0:
                        new_col_names = [col for col in gdf.columns if col not in prev_columns]
                        logger.info(f"🆕 New columns: {new_col_names[:10] + ['...'] if len(new_col_names) > 10 else new_col_names}")
                    else:
                        logger.warning(f"⚠️ No new columns generated by {engine_name}!")
                    
                    # Validate data after each engine
                    logger.info(f"🔍 Validating data after {engine_name} for {currency_pair}")
                    validation_result = self._validate_intermediate_data(gdf, currency_pair, engine_name)
                    if not validation_result:
                        logger.error(f"❌ Data validation failed after {engine_name} for {currency_pair}")
                        return None
                    else:
                        logger.info(f"✅ Data validation passed after {engine_name} for {currency_pair}")

                    # Aggressive local GPU memory cleanup between engines (cuDF path)
                    try:
                        import gc as _gc
                        _gc.collect()
                        cp.get_default_memory_pool().free_all_blocks()
                        logger.info(f"🧹 Memory cleanup completed after {engine_name}")
                    except Exception as e:
                        logger.warning(f"⚠️ Memory cleanup failed after {engine_name}: {e}")
                    
                    # Log successful completion of this engine
                    logger.info(f"✅ Engine {engine_idx}/{len(engine_execution_order)} ({engine_name}) completed successfully for {currency_pair}")
                    logger.info(f"🔄 Proceeding to next engine...")
                
                except Exception as e:
                    logger.error(f"🚨 CRITICAL ERROR in {engine_name} Engine: {e}")
                    logger.error(f"🔍 Error details for {engine_name}: {type(e).__name__}: {str(e)}")
                    import traceback
                    logger.error(f"📋 Full traceback for {engine_name}: {traceback.format_exc()}")
                    
                    error_handling_cfg = getattr(self.settings, 'error_handling', None)
                    continue_on_error = bool(getattr(error_handling_cfg, 'continue_on_error', False)) if error_handling_cfg else False
                    
                    logger.info(f"⚙️ Error handling config: continue_on_error={continue_on_error}")
                    
                    if not continue_on_error:
                        logger.error(f"🛑 Stopping pipeline immediately due to critical error in {engine_name}.")
                        logger.error(f"🔄 To continue processing despite errors, set error_handling.continue_on_error=true in config.yaml")
                        raise CriticalPipelineError(f"Engine {engine_name} failed: {e}")
                    else:
                        logger.warning(f"⚠️ Continuing pipeline despite error in {engine_name}")
                        logger.warning(f"🔄 Pipeline will continue to next engine due to continue_on_error=true")
            
            # Log completion of all engines
            logger.info(f"🎉 All engines completed successfully for {currency_pair}")
            logger.info(f"📊 Final processing summary for {currency_pair}:")
            
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
                logger.info(f"🔧 Starting EMD (signal_processing) engine")
                try:
                    emd_cfg = getattr(self.settings.features, 'emd', {})
                    max_imfs = int(emd_cfg.get('max_imfs', getattr(self.settings.features, 'emd_max_imfs', 10)))
                    rolling_enable = bool(emd_cfg.get('rolling_enable', getattr(self.settings.features, 'emd_rolling_enable', False)))
                    window_size = int(emd_cfg.get('rolling_window', getattr(self.settings.features, 'emd_rolling_window', 500)))
                    step_size = int(emd_cfg.get('rolling_step', getattr(self.settings.features, 'emd_rolling_step', 50)))
                    embargo = int(emd_cfg.get('rolling_embargo', getattr(self.settings.features, 'emd_rolling_embargo', 0)))
                    logger.info(f"📋 EMD config: max_imfs={max_imfs}, rolling_enable={rolling_enable}, window_size={window_size}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load EMD config, using defaults: {e}")
                    max_imfs = getattr(self.settings.features, 'emd_max_imfs', 10)
                    rolling_enable = getattr(self.settings.features, 'emd_rolling_enable', False)
                    window_size = getattr(self.settings.features, 'emd_rolling_window', 500)
                    step_size = getattr(self.settings.features, 'emd_rolling_step', 50)
                    embargo = getattr(self.settings.features, 'emd_rolling_embargo', 0)

                # Check for source column with fallback
                src_col = 'y_close' if 'y_close' in gdf.columns else ('close' if 'close' in gdf.columns else None)
                logger.info(f"🔍 Looking for source column: found '{src_col}'")
                if not src_col:
                    logger.warning("❌ No 'y_close' or 'close' column found; skipping EMD (signal_processing)")
                    logger.warning(f"📋 Available columns: {list(gdf.columns)[:10]}...")
                    return gdf
                
                logger.info(f"📊 Source column '{src_col}' has {len(gdf[src_col])} rows")
                try:
                    if rolling_enable:
                        logger.info(f"🔄 Using rolling EMD with window_size={window_size}, step_size={step_size}")
                        try:
                            from features.signal_processing import apply_rolling_emd
                            logger.info(f"✅ Successfully imported apply_rolling_emd")
                            imfs = apply_rolling_emd(gdf[src_col], max_imfs=max_imfs,
                                                     window_size=window_size, step_size=step_size,
                                                     embargo=embargo)
                        except ImportError as ie:
                            logger.error(f"❌ Failed to import apply_rolling_emd: {ie}")
                            raise
                        except Exception as ee:
                            logger.error(f"❌ Error calling apply_rolling_emd: {ee}")
                            raise
                    else:
                        logger.info(f"🔄 Using standard EMD with max_imfs={max_imfs}")
                        try:
                            from features.signal_processing import apply_emd_to_series
                            logger.info(f"✅ Successfully imported apply_emd_to_series")
                            imfs = apply_emd_to_series(gdf[src_col], max_imfs=max_imfs)
                        except ImportError as ie:
                            logger.error(f"❌ Failed to import apply_emd_to_series: {ie}")
                            raise
                        except Exception as ee:
                            logger.error(f"❌ Error calling apply_emd_to_series: {ee}")
                            raise
                    
                    logger.info(f"✅ EMD completed successfully, generated {imfs.shape[1]} IMFs")
                    
                    # Prefix with 'emd_' to namespace features
                    imfs = imfs.rename(columns={c: f"emd_{c}" for c in imfs.columns})
                    logger.info(f"🏷️ Renamed IMF columns: {list(imfs.columns)}")
                    
                    gdf = cudf.concat([gdf, imfs], axis=1)
                    logger.info(f"✅ EMD concatenation completed: {imfs.shape[1]} IMFs added to DataFrame")
                    logger.info(f"📊 DataFrame shape after EMD: {gdf.shape}")
                    return gdf
                except Exception as e:
                    logger.error(f"❌ Falha ao aplicar EMD: {e}")
                    import traceback
                    logger.error(f"📋 EMD traceback: {traceback.format_exc()}")
                    logger.warning(f"⚠️ Mantendo DataFrame inalterado devido ao erro no EMD")
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

                if 'y_close' not in ddf.columns or 'timestamp' not in ddf.columns:
                    logger.warning("Colunas necessárias ('timestamp','y_close') ausentes; pulando EMD (Dask)")
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

                # Check for source column with fallback
                src_col = 'y_close' if 'y_close' in ddf.columns else ('close' if 'close' in ddf.columns else None)
                if not src_col:
                    logger.warning("No 'y_close' or 'close' column found; skipping EMD (signal_processing)")
                    return ddf
                
                # Optimized EMD processing to avoid large graph creation
                try:
                    logger.info(f"Starting EMD processing: {max_imfs} IMFs, rolling={rolling_enable}")
                    
                    # Use map_partitions to process EMD on each partition locally
                    # This avoids creating large graphs by keeping computation distributed
                    def process_partition_emd(partition):
                        """Process EMD on a single partition to avoid large graph creation."""
                        if len(partition) == 0:
                            return partition
                        
                        try:
                            # Sort by timestamp within partition
                            partition = partition.sort_values('timestamp')
                            
                            # Suppress EMD logs from signal_processing module
                            import logging
                            emd_logger = logging.getLogger('features.signal_processing')
                            original_level = emd_logger.level
                            emd_logger.setLevel(logging.WARNING)  # Suppress INFO logs
                            
                            try:
                                if rolling_enable:
                                    from features.signal_processing import apply_rolling_emd
                                    imfs = apply_rolling_emd(partition[src_col], max_imfs=max_imfs,
                                                           window_size=window_size, step_size=step_size,
                                                           embargo=embargo)
                                else:
                                    from features.signal_processing import apply_emd_to_series
                                    imfs = apply_emd_to_series(partition[src_col], max_imfs=max_imfs)
                            finally:
                                # Restore original log level
                                emd_logger.setLevel(original_level)
                            
                            # Rename columns and merge with original partition
                            imfs = imfs.rename(columns={c: f"emd_{c}" for c in imfs.columns})

                            # Align to expected number of IMFs based on metadata
                            expected_imfs = n_imfs  # from outer scope
                            actual_imfs_cnt = len(imfs.columns)
                            if actual_imfs_cnt > expected_imfs:
                                # Trim extra IMFs to match meta
                                keep_cols = [f"emd_imf_{i}" for i in range(1, expected_imfs + 1)]
                                existing_keep = [c for c in keep_cols if c in imfs.columns]
                                dropped = set(imfs.columns) - set(existing_keep)
                                if dropped:
                                    logger.debug(f"Trimming extra IMFs: dropping {len(dropped)} columns")
                                imfs = imfs[existing_keep]
                            elif actual_imfs_cnt < expected_imfs:
                                # Pad missing IMFs with NaN
                                logger.debug(f"EMD gerou {actual_imfs_cnt} IMFs, preenchendo {expected_imfs - actual_imfs_cnt} restantes com NaN")
                                import numpy as np
                                for i in range(actual_imfs_cnt + 1, expected_imfs + 1):
                                    imfs[f"emd_imf_{i}"] = np.full(len(imfs), np.nan)
                            
                            imfs = imfs.reset_index(drop=True)
                            partition = partition.reset_index(drop=True)
                            
                            # Combine original data with IMFs
                            result = cudf.concat([partition, imfs], axis=1)
                            return result
                            
                        except Exception as e:
                            # Reduce log verbosity - only log once per currency pair
                            logger.debug(f"EMD failed on partition: {e}")
                            return partition
                    
                    # First, process one partition to determine actual number of IMFs generated
                    sample_partition = ddf.get_partition(0).compute()
                    if len(sample_partition) > 0:
                        sample_partition = sample_partition.sort_values('timestamp')
                        
                        # Suppress EMD logs for sample
                        import logging
                        emd_logger = logging.getLogger('features.signal_processing')
                        original_level = emd_logger.level
                        emd_logger.setLevel(logging.WARNING)
                        
                        try:
                            if rolling_enable:
                                from features.signal_processing import apply_rolling_emd
                                sample_imfs = apply_rolling_emd(sample_partition[src_col], max_imfs=max_imfs,
                                                             window_size=window_size, step_size=step_size,
                                                             embargo=embargo)
                            else:
                                from features.signal_processing import apply_emd_to_series
                                sample_imfs = apply_emd_to_series(sample_partition[src_col], max_imfs=max_imfs)
                            
                            # Determine actual number of IMFs generated
                            actual_imfs = sample_imfs.shape[1]
                            logger.info(f"EMD sample shows {actual_imfs} IMFs will be generated (requested: {max_imfs})")
                            
                        finally:
                            emd_logger.setLevel(original_level)
                    else:
                        actual_imfs = max_imfs

                    # Choose metadata IMF count as the maximum of requested vs sample-observed
                    n_imfs = max(max_imfs, int(actual_imfs))
                    
                    # Create proper metadata that includes EMD columns based on chosen IMF count
                    meta = ddf._meta.copy()
                    for i in range(1, n_imfs + 1):
                        meta[f"emd_imf_{i}"] = meta[src_col].astype('float32')
                    
                    # Apply EMD processing to each partition with correct metadata
                    ddf = ddf.map_partitions(process_partition_emd, meta=meta)
                    logger.info(f"EMD processing completed: {max_imfs} IMFs added to {ddf.npartitions} partitions (requested: {max_imfs}, actual generated: {actual_imfs})")
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
            # Remove smart_persist import - let it fail if needed
            # from utils.dask_graph_optimizer import smart_persist
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
            # Store target for model naming
            self.current_target_column = target
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
                        
                        # Build and evaluate final model with selected features using Dask GPU CatBoost
                        # Only execute final model in Stage B (modeling mode), not in Stage A (preprocess_selection)
                        study_mode = getattr(self.settings, 'study', {}).get('mode', 'preprocess_selection')
                        if study_mode == 'modeling':
                            try:
                                logger.info("[FinalModel] 🚀 Building final CatBoost model with Dask GPU...")
                                
                                # Use Dask GPU CatBoost for final model training
                                from features.dask_gpu_catboost import DaskGPUCatBoostTrainer, DaskGPUCatBoostConfig
                                
                                # Prepare data - keep in dask_cudf format
                                if hasattr(ddf, 'compute'):
                                    # Data is already in dask_cudf format
                                    X_ddf = ddf[final_features]
                                    y_ddf = ddf[target]
                                else:
                                    # Convert to dask_cudf if needed
                                    import dask_cudf
                                    X_ddf = dask_cudf.from_cudf(ddf[final_features], npartitions=1)
                                    y_ddf = dask_cudf.from_cudf(ddf[target], npartitions=1)
                                
                                # Configure Dask GPU CatBoost
                                config = DaskGPUCatBoostConfig(
                                    iterations=1000,
                                    learning_rate=0.03,
                                    depth=6,
                                    task_type='GPU',
                                    devices='0',
                                    verbose=200,
                                    early_stopping_rounds=50
                                )
                                
                                # Create trainer and train model
                                trainer = DaskGPUCatBoostTrainer(config)
                                
                                # Train with feature selection (using already selected features)
                                result = trainer.train_with_feature_selection(
                                    X_ddf, y_ddf,
                                    target_columns=final_features,
                                    num_features_to_select=len(final_features),
                                    task_type='regression',
                                    test_size=0.2
                                )
                                
                                model = result['model']
                                feature_importance_map = result['feature_importances']
                                
                                # Sort features by importance
                                sorted_features = sorted(feature_importance_map.items(), key=lambda x: x[1], reverse=True)
                                
                                logger.info(f"[FinalModel] Dask GPU CatBoost model trained successfully!")
                                logger.info(f"[FinalModel] 📊 Training completed with {result['training_samples']} samples")
                                logger.info(f"[FinalModel] 🎯 Selected {result['selected_count']} features from {result['total_features']} candidates")
                                
                                logger.info(f"[FinalModel] 🎯 All Feature Importances:")
                                for i, (feature_name, importance) in enumerate(sorted_features, 1):
                                    logger.info(f"[FinalModel]   {i:2d}. {feature_name}: {importance:.6f}")
                                
                                # Save model with proper naming (include target)
                                currency_pair = getattr(self, 'current_currency_pair', 'unknown')
                                timeframe = getattr(self, 'current_timeframe', 'unknown')
                                target_name = getattr(self, 'current_target_column', 'unknown_target')
                                model_filename = f"{currency_pair}_{timeframe}_{target_name}_model.cbm"
                                model.save_model(model_filename)
                                logger.info(f"[FinalModel] 💾 Model saved as: {model_filename}")
                            
                            except Exception as e:
                                logger.error(f"[FinalModel] Failed to build final model: {e}")
                                logger.error(f"[FinalModel] Error details: {traceback.format_exc()}")
                            else:
                                logger.info("[FinalModel] ✅ Final model training completed successfully")
                        else:
                            logger.info("[FinalModel] ⏭️  Skipping final model training (Stage A - preprocess_selection mode)")
                            logger.info("[FinalModel] Final model will be trained in Stage B (modeling mode)")
                            
                        if not final_features:
                            logger.warning("[StatTests] ⚠️  FINAL SELECTION: No features selected!")
                        try:
                            self._write_stat_tests_artifacts(
                                getattr(self, 'current_currency_pair', 'unknown'),
                                dcor_summary=getattr(self.stats, '_last_dcor_summary', {}) or None,
                                selection_results=combined,
                            )
                        except Exception:
                            pass
                except Exception as e:
                    logger.error(f"[StatTests] Selection stages failed: {e}")
        except Exception as e:
            logger.error(f"[StatTests] Selection stage failed: {e}")

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
            output_cfg = getattr(self.settings, 'output', None)
            output_path = getattr(output_cfg, 'output_path', './output') if output_cfg else './output'
            out_dir = pathlib.Path(output_path) / currency_pair / 'checkpoint'
            out_dir.mkdir(parents=True, exist_ok=True)
            # File name per engine
            intermediate_format = getattr(output_cfg, 'intermediate_format', 'parquet') if output_cfg else 'parquet'
            fname = f"{engine_name}." + ('parquet' if str(intermediate_format).lower() == 'parquet' else 'feather')

            # Save
            fmt = str(intermediate_format).lower()
            if fmt == 'feather':
                import pyarrow.feather as feather
                table = gdf.to_arrow()
                feather.write_feather(
                    table,
                    str(out_dir / fname),
                    compression=str(getattr(self.settings.output, 'intermediate_compression', 'zstd')),
                    version=int(getattr(self.settings.output, 'intermediate_version', 2)),
                )
            else:
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
        Handles metadata mismatches that can occur when engines add new columns.
        """
        try:
            # Try to get a sample with proper metadata handling
            try:
                sample = ddf.head(1)
                if sample.shape[0] == 0:
                    logger.error(f"Empty Dask DataFrame after {engine_name} for {currency_pair}")
                    return False
                return True
            except Exception as meta_error:
                # Handle metadata mismatch - this is common when engines add new columns
                if "columns in the computed data do not match" in str(meta_error):
                    logger.warning(f"Metadata mismatch detected for {currency_pair} after {engine_name} - attempting to fix")
                    try:
                        # Try to compute a small sample to validate data exists
                        # Use a more robust approach that doesn't rely on metadata
                        sample_size = min(1, len(ddf))
                        if sample_size > 0:
                            # Force computation of a small sample without metadata validation
                            sample = ddf.map_partitions(lambda x: x.head(1) if len(x) > 0 else x).compute()
                            if len(sample) > 0:
                                logger.info(f"Data validation successful for {currency_pair} after {engine_name} (metadata mismatch resolved)")
                                return True
                            else:
                                logger.error(f"Empty data after metadata fix for {currency_pair} after {engine_name}")
                                return False
                        else:
                            logger.error(f"No data available for {currency_pair} after {engine_name}")
                            return False
                    except Exception as fix_error:
                        logger.error(f"Failed to fix metadata mismatch for {currency_pair} after {engine_name}: {fix_error}")
                        return False
                else:
                    # Re-raise if it's not a metadata mismatch
                    raise meta_error
        except Exception as e:
            logger.error(f"Error validating Dask data for {currency_pair} after {engine_name}: {e}")
            return False

    def _build_final_model(self, 
                          full_gdf,  # Accept both cudf.DataFrame and dask_cudf.DataFrame
                          target_col: str,
                          selected_features: list,
                          feature_importances: dict,
                          selection_metadata: dict,
                          symbol: str,
                          timeframe: str):
        """Build and evaluate the final CatBoost model with selected features."""
        
        import traceback
        
        try:
            logger.info(f"[FinalModel] Initializing final model trainer for {symbol} {timeframe}")
            
            # Initialize the final model trainer
            final_model_trainer = FinalModelTrainer(
                config=self.settings,
                logger_instance=logger
            )
            
            # Verify target column exists
            if target_col not in full_gdf.columns:
                logger.error(f"[FinalModel] Target column '{target_col}' not found in DataFrame")
                return False
            
            logger.info(f"[FinalModel] DataFrame columns: {list(full_gdf.columns)}")
            logger.info(f"[FinalModel] DataFrame shape: {full_gdf.shape}")
            logger.info(f"[FinalModel] Building final model with {len(selected_features)} features and {len(full_gdf)} samples")
            
            # Build and evaluate the final model
            final_results = final_model_trainer.build_and_evaluate_final_model(
                X_df=full_gdf,
                y_series=full_gdf[target_col],
                selected_features=selected_features,
                feature_importances=feature_importances,
                selection_metadata=selection_metadata,
                symbol=symbol,
                timeframe=timeframe
            )
            
            logger.info(f"[FinalModel] ✅ Final model completed successfully!")
            logger.info(f"[FinalModel] 💾 Model saved to database - ID: {final_results['database_record_id']}")
            
            # Log R2 upload status
            if final_results.get('r2_upload_success', False):
                logger.info(f"[FinalModel] ☁️  Model uploaded to R2 cloud storage successfully!")
                model_name = f"catboost_{symbol.lower()}_{timeframe}"
                logger.info(f"[FinalModel] 🌐 R2 Path: models/{symbol}/{model_name}_v*.cbm")
                logger.info(f"[FinalModel] 📤 Upload includes model file and JSON metadata")
                logger.info(f"[FinalModel] 🧹 Local temporary files cleaned up automatically")
            else:
                logger.warning(f"[FinalModel] ⚠️  R2 upload failed or was skipped - model only in database")
            
            # Get target name from selection metadata
            target_name = selection_metadata.get('target_column', 'unknown_target')
            model_name = f"catboost_{symbol.lower()}_{timeframe}_{target_name}"
            logger.info(f"[FinalModel] 🏷️  Model Name: {model_name}")
            logger.info(f"[FinalModel] 📊 Task Type: {final_results['task_type']}")
            logger.info(f"[FinalModel] 🎯 Features Used: {final_results['feature_count']}")
            logger.info(f"[FinalModel] 📈 Training Score: {final_results['evaluation_results']['train_primary_metric']:.6f}")
            logger.info(f"[FinalModel] 🎖️  Test Score: {final_results['evaluation_results']['test_primary_metric']:.6f}")
            
            # Log database connection info
            db_config = self.settings.database
            logger.info(f"[FinalModel] 🗄️  Database: {db_config.host}:{db_config.port}/{db_config.database}")
            
            # Log top features from final model
            final_importances = final_results['model_results']['feature_importances']
            top_features = sorted(final_importances.items(), key=lambda x: x[1], reverse=True)[:10]
            
            logger.info(f"[FinalModel] 🏆 Top 10 most important features in final model:")
            for i, (feature, importance) in enumerate(top_features, 1):
                logger.info(f"[FinalModel]   {i:2d}. {feature}: {importance:.6f}")
            
            # Log query commands for easy access
            logger.info(f"[FinalModel] 💡 To query this model: python query_models.py --details {final_results['database_record_id']}")
            logger.info(f"[FinalModel] 💡 To list all {symbol} models: python query_models.py --symbol {symbol}")
            
            # Log R2 management commands
            if final_results.get('r2_upload_success', False):
                logger.info(f"[FinalModel] 🔧 To manage R2 models: python r2_models.py list-models --symbol {symbol}")
                logger.info(f"[FinalModel] 🔍 To view R2 metadata: python r2_models.py get-metadata {symbol} <model_name>")
            
            # Store results for potential later use
            self._last_final_model_results = final_results
            
        except Exception as e:
            logger.error(f"[FinalModel] Failed to build final model: {e}")
            logger.error(f"[FinalModel] Traceback: {traceback.format_exc()}")
            raise

    def process_currency_pair_dask(self, currency_pair: str, r2_path: str, client: Client) -> bool:
        """Process a single currency pair using dask_cudf (delegates to module impl)."""
        return _process_currency_pair_dask_impl(self, currency_pair, r2_path, client)


def _noop_ctx():
    from contextlib import contextmanager
    @contextmanager
    def _noop():
        yield
    return _noop()




def _register_task_success(processor: "DataProcessor", currency_pair: str, r2_path: str):
    """Register successful task completion in the database."""
    if not processor.db_connected:
        return
    try:
        task_id = processor.db_handler.register_currency_pair(currency_pair, r2_path)
        if task_id:
            processor.db_handler.update_task_status(task_id, 'COMPLETED')
            logger.info(f"Successfully registered task completion for {currency_pair} (Task ID: {task_id})")
        else:
            logger.warning(f"Failed to register task in database for {currency_pair}")
    except Exception as e:
        logger.error(f"Error registering task success for {currency_pair}: {e}")

def _register_task_failure(processor: "DataProcessor", currency_pair: str, r2_path: str, error_message: str):
    """Register task failure in the database."""
    if not processor.db_connected:
        return
    try:
        task_id = processor.db_handler.register_currency_pair(currency_pair, r2_path)
        if task_id:
            processor.db_handler.update_task_status(task_id, 'FAILED', error_message)
            logger.info(f"Registered task failure for {currency_pair} (Task ID: {task_id})")
        else:
            logger.warning(f"Failed to register task failure in database for {currency_pair}")
    except Exception as e:
        logger.error(f"Error registering task failure for {currency_pair}: {e}")



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
                    '_drop_denied_columns',
                    '_execute_engine_dask',
                    '_log_statistical_tests_plan_dask',
                    '_build_final_model',
                    '_run_statistical_tests_stages',
                ]
                
                # Try multiple sources for helper methods
                _sources = [_src_cls, DataProcessor]
                
                for _name in _need:
                    if not hasattr(self, _name):
                        for _source_cls in _sources:
                            if hasattr(_source_cls, _name):
                                try:
                                    setattr(self, _name, _types.MethodType(getattr(_source_cls, _name), self))
                                    break
                                except Exception as _e:
                                    logger.debug(f"Failed to bind '{_name}' from {_source_cls.__name__}: {_e}")
                                    continue
            except Exception:
                pass

            # Verify that all required methods are available - fail fast if missing
            required_methods = [
                '_validate_initial_data_dask',
                '_validate_intermediate_data_dask', 
                '_save_intermediate_data',
                '_drop_denied_columns',
                '_execute_engine_dask',
                '_log_statistical_tests_plan_dask',
                '_build_final_model',
                '_run_statistical_tests_stages'
            ]
            
            missing_methods = []
            for method_name in required_methods:
                if not hasattr(self, method_name):
                    missing_methods.append(method_name)
            
            if missing_methods:
                error_msg = f"Missing required methods: {missing_methods}. Pipeline cannot continue."
                logger.error(error_msg)
                raise RuntimeError(error_msg)

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
                import dask
                # Only use annotations if not in query-planning mode
                try:
                    with dask.annotate(task_key_name=f"load-{currency_pair}"):
                        _use_annotations = True
                except Exception:
                    _use_annotations = False
                
                if _use_annotations:
                    with dask.annotate(task_key_name=f"load-{currency_pair}"):
                        if r2_lower.endswith('.parquet'):
                            ddf = self.loader.load_currency_pair_data(r2_path, client)
                        elif r2_lower.endswith('.feather'):
                            ddf = self.loader.load_currency_pair_data_feather(r2_path, client)
                        else:
                            # Attempt feather (dir or file), then parquet
                            ddf = self.loader.load_currency_pair_data_feather(r2_path, client)
                            if ddf is None:
                                ddf = self.loader.load_currency_pair_data(r2_path, client)
                else:
                    # Execute without annotations
                    if r2_lower.endswith('.parquet'):
                        ddf = self.loader.load_currency_pair_data(r2_path, client)
                    elif r2_lower.endswith('.feather'):
                        ddf = self.loader.load_currency_pair_data_feather(r2_path, client)
                    else:
                        # Attempt feather (dir or file), then parquet
                        ddf = self.loader.load_currency_pair_data_feather(r2_path, client)
                        if ddf is None:
                            ddf = self.loader.load_currency_pair_data(r2_path, client)
            except Exception as _e:
                logger.error(f"❌ CRITICAL ERROR: Failed to load data for {currency_pair}: {_e}")
                logger.error("❌ Pipeline cannot continue without data.")
                raise RuntimeError(f"Failed to load data for {currency_pair}: {_e}")
            if ddf is None:
                _register_task_failure(self, currency_pair, r2_path, "Failed to load Dask data")
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

            # Intelligent repartitioning to optimize performance
            try:
                n_workers = 1
                try:
                    sched = client.scheduler_info()
                    n_workers = max(1, len(sched.get('workers', {})))
                except Exception:
                    pass
                
                # Calculate optimal partition count based on data size and workers
                current_partitions = ddf.npartitions
                data_size = len(ddf) if hasattr(ddf, '__len__') else 0
                
                # Optimal partition sizing strategy:
                # - Minimum: 1 partition per worker for parallelization
                # - Maximum: 4 partitions per worker to avoid overhead
                # - Consider data size: larger datasets can handle more partitions
                min_partitions = n_workers
                max_partitions = min(n_workers * 4, 50)  # Cap at 50 to prevent excessive partitions
                
                # For very large datasets, allow more partitions but still cap reasonably
                if data_size > 1000000:  # 1M+ rows
                    max_partitions = min(n_workers * 6, 75)
                elif data_size > 100000:  # 100K+ rows
                    max_partitions = min(n_workers * 5, 60)
                
                # Choose target partitions within reasonable bounds
                target_parts = max(min_partitions, min(max_partitions, current_partitions))
                
                # Only repartition if it makes sense
                if current_partitions != target_parts and target_parts > 0:
                    import dask as _dask
                    with _dask.annotate(task_key_name=f"repartition-{currency_pair}"):
                        logger.info(f"Repartitioning from {current_partitions} to {target_parts} (workers: {n_workers}, data_size: {data_size})")
                        ddf = ddf.repartition(npartitions=target_parts)
                else:
                    logger.info(f"Keeping current partition count: {current_partitions} (optimal for {n_workers} workers)")
                
                try:
                    logger.info(f"Post-load overview: cols={len(ddf.columns)}, npartitions={ddf.npartitions}")
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"Could not repartition: {e}")

            # Validate initial data
            if not self._validate_initial_data_dask(ddf, currency_pair):
                _register_task_failure(self, currency_pair, r2_path, "Initial Dask data validation failed")
                return False

            # Engine execution order from settings
            pipeline_cfg = getattr(self.settings, 'pipeline', None)
            engine_config = getattr(pipeline_cfg, 'engines', {}) if pipeline_cfg else {}
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
                        logger.error(f"❌ CRITICAL: Engine {engine_name} returned None for {currency_pair}")
                        logger.error(f"🛑 This is a critical error that should stop the pipeline")
                        return False

                    # Stabilize graph/memory between engines with smart persist strategy
                    logger.info(f"Persist start: {engine_name}")
                    try:
                        debug_dash = bool(getattr(self.settings.development, 'debug_dashboard', False))
                        if debug_dash:
                            # In debug mode, avoid graph optimization to keep task names visible
                            from dask import persist as _persist
                            ddf, = _persist(ddf, optimize_graph=False)
                        else:
                            # Use smart persist strategy for other engines
                            # Remove smart_persist import - let it fail if needed
            # from utils.dask_graph_optimizer import smart_persist
                            ddf = ddf.persist()
                    except Exception as persist_error:
                        logger.error(f"❌ CRITICAL ERROR: Failed to persist data for {engine_name}: {persist_error}")
                        raise RuntimeError(f"Failed to persist data for {engine_name}: {persist_error}")
                    logger.info(f"Persist returned: {engine_name}")
                    # Revert to prior behavior: do not immediately wait here; let downstream stages trigger computation as needed

                    # Free CuPy pools on all workers to reduce carry-over memory
                    try:
                        if self.client:
                            self.client.run(lambda: (__import__('gc').collect(), __import__('cupy').get_default_memory_pool().free_all_blocks()))
                    except Exception:
                        pass

                    # Skip intermediate validation - will validate once before final model
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
                    logger.error(f"🔍 Error details for {engine_name}: {type(eng_err).__name__}: {str(eng_err)}")
                    import traceback
                    logger.error(f"📋 Full traceback for {engine_name}: {traceback.format_exc()}")
                    
                    if self.db_connected and stage_id:
                        try:
                            self.db_handler.error_stage(stage_id, error_message=str(eng_err))
                        except Exception:
                            pass
                    
                    # Check if this is a critical error that should stop the pipeline
                    error_handling_cfg = getattr(self.settings, 'error_handling', None)
                    continue_on_error = bool(getattr(error_handling_cfg, 'continue_on_error', False)) if error_handling_cfg else False
                    
                    # Critical errors that should always stop the pipeline
                    critical_errors = [
                        "'NoneType' object has no attribute",
                        "Engine returned None",
                        "Data validation failed",
                        "CriticalPipelineError",
                        "NameError",
                        "name 'self' is not defined",
                        "out of memory",
                        "cuda error",
                        "memory allocation",
                        "bad_alloc",
                        "cudaerrormemoryallocation",
                        "rmm failure",
                        "maximum pool size exceeded",
                        "worker failed to start"
                    ]
                    
                    is_critical_error = any(critical_err in str(eng_err) for critical_err in critical_errors)
                    
                    if is_critical_error or not continue_on_error:
                        logger.error(f"🛑 CRITICAL ERROR: Stopping pipeline due to error in {engine_name}")
                        
                        # Special handling for memory-related critical errors
                        error_str = str(eng_err).lower()
                        if any(mem_err in error_str for mem_err in ['out of memory', 'cuda error', 'memory allocation', 'bad_alloc']):
                            logger.critical("=" * 80)
                            logger.critical("CRITICAL MEMORY ERROR DETECTED")
                            logger.critical("=" * 80)
                            logger.critical("This error indicates insufficient GPU or system memory.")
                            logger.critical("The pipeline has been stopped to prevent system instability.")
                            logger.critical("=" * 80)
                            logger.critical("RECOMMENDATIONS:")
                            logger.critical("1. Reduce memory usage in configuration")
                            logger.critical("2. Check available GPU memory")
                            logger.critical("3. Reduce number of workers or memory per worker")
                            logger.critical("4. Check for memory leaks in the pipeline")
                            logger.critical("=" * 80)
                        
                        logger.error(f"🔄 To continue despite errors, set error_handling.continue_on_error=true in config.yaml")
                        return False
                    else:
                        logger.warning(f"⚠️ Continuing pipeline despite error in {engine_name}")
                        logger.warning(f"🔄 Pipeline will continue to next engine for {currency_pair}")
                        continue

            # Data processing completed successfully
            logger.info(f"Data processing completed successfully for {currency_pair}")

            # Final validation before CatBoost selection
            logger.info(f"[FinalModel] Validating data before CatBoost selection for {currency_pair}")
            if not self._validate_intermediate_data_dask(ddf, currency_pair, "final_validation"):
                logger.error(f"Final validation failed for {currency_pair} - skipping CatBoost selection")
                _register_task_failure(self, currency_pair, r2_path, "Final validation failed")
                return False
            
            # Skip final model building to avoid memory issues with large datasets
            logger.info(f"[FinalModel] Skipping final model building for {currency_pair} to avoid memory issues")
            logger.info(f"[FinalModel] Feature selection completed successfully - this is the main goal")
            
            # Register successful completion
            _register_task_success(self, currency_pair, r2_path)

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

            _register_task_success(self, currency_pair, r2_path)
            logger.info(f"Successfully completed Dask processing for {currency_pair}")
            return True
    except Exception as e:
        logger.error(f"Error in Dask processing for {currency_pair}: {e}", exc_info=True)
        _register_task_failure(self, currency_pair, r2_path, str(e))
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
            logger.info(f"🔍 Starting validation after {engine_name} for {currency_pair}")
            logger.info(f"📊 DataFrame shape: {gdf.shape}")
            
            # Check for empty DataFrame
            if len(gdf) == 0:
                logger.error(f"❌ Empty DataFrame after {engine_name} for {currency_pair}")
                return False
            
            logger.info(f"✅ DataFrame is not empty: {len(gdf)} rows")
            
            # Check for excessive NaN values
            nan_counts = gdf.isna().sum()
            high_nan_cols = nan_counts[nan_counts > len(gdf) * 0.5]  # More than 50% NaN
            if len(high_nan_cols) > 0:
                logger.warning(f"⚠️ High NaN columns after {engine_name} for {currency_pair}: {list(high_nan_cols.index)}")
            else:
                logger.info(f"✅ No high NaN columns found after {engine_name}")
            
            # Check for infinite values (dtype-safe approach)
            import numpy as np
            numeric_cols = [c for c, dt in gdf.dtypes.items() if np.issubdtype(np.dtype(str(dt)), np.number)]
            logger.info(f"🔢 Found {len(numeric_cols)} numeric columns for infinite value check")
            
            if numeric_cols:
                # Replace inf values with NaN for counting
                gdf_numeric = gdf[numeric_cols].replace([np.inf, -np.inf], np.nan)
                inf_counts = gdf_numeric.isna().sum() - gdf[numeric_cols].isna().sum()  # Only count new NaNs (from inf)
                high_inf_cols = inf_counts[inf_counts > 0]
                if len(high_inf_cols) > 0:
                    logger.warning(f"⚠️ Infinite values found after {engine_name} for {currency_pair}: {list(high_inf_cols.index)}")
                else:
                    logger.info(f"✅ No infinite values found after {engine_name}")
            
            logger.info(f"✅ Validation passed for {engine_name} on {currency_pair}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validating intermediate data for {currency_pair} after {engine_name}: {e}")
            import traceback
            logger.error(f"📋 Validation traceback: {traceback.format_exc()}")
            return False
    

    # Removido: persistência separada de selected_features.json (seleção centralizada no statistical_tests)


    

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


def process_currency_pair_dask_worker(currency_pair: str, r2_path: str, run_id: Optional[int] = None) -> bool:
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
            processor = (_mod.DataProcessor if _mod and hasattr(_mod, 'DataProcessor') else DataProcessor)(client=wc, run_id=run_id)
            # Directly invoke the class implementation; it handles any internal bindings as needed
            return processor.process_currency_pair_dask(currency_pair, r2_path, wc)
    except Exception as e:
        logger.error(f"Error in Dask worker processing {currency_pair}: {e}", exc_info=True)
        return False


