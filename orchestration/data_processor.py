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
from features import StationarizationEngine, StatisticalTests, GARCHModels, FeatureEngineeringEngine
from features.base_engine import CriticalPipelineError
from utils.logging_utils import get_logger
from features.engine_metrics import EngineMetrics

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
            logger.info(f"Starting processing for {currency_pair}")
            
            # Connect to database for task tracking (non-fatal)
            if not self.db_connected:
                self.db_connected = self.db_handler.connect()
                if not self.db_connected:
                    logger.warning("Database unavailable; proceeding without task tracking")
            
            # Load data
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
            # Try loading as cuDF first
            gdf = self.loader.load_currency_pair_data_feather_sync(r2_path)
            if gdf is not None:
                # Drop denied columns early (never enter any stage)
                gdf = self._drop_denied_columns(gdf)
                logger.info(f"Loaded data as cuDF: {len(gdf)} rows, {len(gdf.columns)} columns")
                return gdf
            
            # Fallback to regular loading
            gdf = self.loader.load_currency_pair_data_sync(r2_path)
            if gdf is not None:
                # Drop denied columns early
                gdf = self._drop_denied_columns(gdf)
                logger.info(f"Loaded data with fallback: {len(gdf)} rows, {len(gdf.columns)} columns")
                return gdf
            
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
                logger.info("Signal processing engine removed; pass-through (cuDF)")
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
                return self.stats.process(ddf)
            elif engine_name == 'signal_processing':
                logger.info("Signal processing engine removed; pass-through (Dask)")
                return ddf
            elif engine_name == 'garch_models':
                return self.garch.process(ddf)
            else:
                logger.warning(f"⚠️ Unknown engine: {engine_name}")
                return ddf
        except Exception as e:
            logger.error(f"Error executing Dask engine {engine_name}: {e}", exc_info=True)
            return None

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
            # File name per engine
            fname = f"{engine_name}." + ('parquet' if str(getattr(self.settings.output, 'intermediate_format', 'parquet')).lower() == 'parquet' else 'feather')

            # Save
            fmt = str(getattr(self.settings.output, 'intermediate_format', 'parquet')).lower()
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
        """
        Process a single currency pair using dask_cudf + multi-GPU engines on the driver.
        """
        try:
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
                    logger.info(f"Waiting for {engine_name} computation to complete (timeout: 60s)...")
                    try:
                        wait(ddf, timeout=60)  # 1 minute timeout
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

            # Save directly from Dask without materializing everything on one GPU
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
            
            logger.info(f"Saving processed data with {len(gdf.columns)} columns to Feather v2 files for {currency_pair}")
            
            # Get the output path from settings
            output_path = self.settings.output.output_path
            compression = "lz4"  # Fast compression for Feather v2
            
            # Create output directory structure
            import pathlib
            import gc
            import pyarrow.feather as feather
            
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
            
            # Save using Feather v2 directly (cuDF DataFrame)
            logger.info(f"Saving to: {output_dir} using Feather v2 (Arrow IPC)")
            
            # Convert cuDF DataFrame to Arrow and save
            try:
                # Convert to Arrow (CPU)
                table = gdf.to_arrow()
                
                # Save as Feather v2
                consolidated_path = output_dir / f"{currency_pair}.feather"
                
                feather.write_feather(
                    table,
                    str(consolidated_path),
                    compression=compression,
                    version=2  # Ensure Feather v2
                )
                
                logger.info(f"Saved consolidated file: {consolidated_path} ({len(gdf)} rows)")
                
                # Clean up GPU memory
                del table
                gc.collect()
                
            except Exception as save_error:
                logger.error(f"Error saving data: {save_error}")
                raise
            
            # Verify the saved data
            saved_files = list(output_dir.glob("*.feather"))
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

    def _save_processed_data_dask(self, ddf, currency_pair: str) -> bool:
        """Save the processed Dask-cuDF DataFrame as partitioned Feather files.

        Avoids collecting the entire dataset into a single GPU/host memory.
        Compatible with our loader, which can read directories with part-*.feather.
        """
        try:
            from dask import delayed, compute
            import pathlib, os
            import pyarrow.feather as feather

            output_path = self.settings.output.output_path
            compression = "lz4"
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

            def _write_part(pdf, out_dir: str, idx: int, comp: str, cols: list):
                # pdf is a cuDF DataFrame (computed partition)
                try:
                    # Ensure column order is consistent
                    if cols:
                        exist = [c for c in cols if c in pdf.columns]
                        pdf = pdf[exist]
                    table = pdf.to_arrow()
                    fname = os.path.join(out_dir, f"part-{idx:05d}.feather")
                    feather.write_feather(table, fname, compression=comp, version=2)
                    return fname
                except Exception as e:
                    raise e

            cols_order = list(ddf.columns)
            tasks = [
                delayed(_write_part)(part, str(output_dir), i, compression, cols_order)
                for i, part in enumerate(parts)
            ]
            written = compute(*tasks)
            logger.info(f"Saved {len(written)} feather parts to {output_dir}")
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
        processor = DataProcessor()
        return processor.process_currency_pair(currency_pair, r2_path)
    except Exception as e:
        logger.error(f"Error in worker processing {currency_pair}: {e}", exc_info=True)
        return False
