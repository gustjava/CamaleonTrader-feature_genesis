"""
Base class for all feature engineering engines in the pipeline.

This module provides a robust foundation for feature engineering engines with:
- Comprehensive input validation
- Enhanced error handling and logging
- Memory management utilities
- Data quality checks
- Performance monitoring
"""

import logging
import sys
import os
import time
import gc
from abc import ABC, abstractmethod
from typing import Dict, Any, Mapping, Optional, List, Union
from dataclasses import dataclass
from contextlib import contextmanager

import cudf
import cupy as cp
import dask_cudf
from dask.distributed import Client

from config.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class CriticalPipelineError(Exception):
    """Special exception raised when a critical error occurs that should stop the entire pipeline."""
    pass


class DataValidationError(Exception):
    """Exception raised when data validation fails."""
    pass


class BaseFeatureEngine(ABC):
    """
    A robust base class for feature engineering modules with comprehensive validation and error handling.
    
    This class provides:
    - Input validation for DataFrames
    - Memory management and monitoring
    - Enhanced error handling and logging
    - Performance tracking
    - Data quality checks
    """
    
    def __init__(self, config: UnifiedConfig, client: Optional[Client] = None):
        """
        Initialize the engine with configuration and optional Dask client.

        Args:
            config: The unified configuration object
            client: Optional Dask distributed client for distributed processing
        """
        self.config = config
        # Backwards-compat alias for engines that expect `self.settings`
        try:
            self.settings = config
        except Exception:
            self.settings = config
        self.client = client
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Performance tracking
        self.processing_times = {}
        self.memory_usage = {}

        # Optional task/run context for DB-backed metrics
        self._run_id: Optional[int] = None
        self._task_id: Optional[int] = None
        self._db_handler = None
        
        # Validation settings
        self.required_columns = self._get_required_columns()
        self.optional_columns = self._get_optional_columns()
        self.expected_dtypes = self._get_expected_dtypes()

    # ---- Optional task context & metrics helpers ----
    def set_task_context(self, run_id: Optional[int], task_id: Optional[int], db_handler: Optional[object] = None):
        """Attach runtime context so engines can persist metrics/artifacts.

        Args:
            run_id: Current pipeline run id
            task_id: Current processing task id
            db_handler: Instance exposing add_metrics/add_artifact methods
        """
        self._run_id = run_id
        self._task_id = task_id
        self._db_handler = db_handler

    def _record_metrics(self, stage: str, metrics: Dict[str, Any]):
        try:
            if not self._db_handler or not hasattr(self._db_handler, 'add_metrics'):
                return
            # Guard by monitoring flag if available on settings
            if hasattr(self.settings, 'monitoring') and not getattr(self.settings.monitoring, 'metrics_enabled', True):
                return
            self._db_handler.add_metrics(self._run_id, self._task_id, stage, metrics)
        except Exception:
            # Non-fatal
            pass

    def _record_artifact(self, stage: str, path: str, kind: str = "file", meta: Optional[Dict[str, Any]] = None):
        try:
            if not self._db_handler or not hasattr(self._db_handler, 'add_artifact'):
                return
            if hasattr(self.settings, 'monitoring') and not getattr(self.settings.monitoring, 'metrics_enabled', True):
                return
            self._db_handler.add_artifact(self._run_id, self._task_id, stage, path, kind, meta)
        except Exception:
            pass
    
    @abstractmethod
    def process(self, df: Union[cudf.DataFrame, dask_cudf.DataFrame]) -> Union[cudf.DataFrame, dask_cudf.DataFrame]:
        """
        The main processing method to be implemented by each engine.

        Args:
            df: The input DataFrame (cuDF or dask-cuDF)

        Returns:
            The DataFrame with new features added
        """
        raise NotImplementedError("Each feature engine must implement the 'process' method.")
    
    def process_with_validation(self, df: Union[cudf.DataFrame, dask_cudf.DataFrame]) -> Union[cudf.DataFrame, dask_cudf.DataFrame]:
        """
        Process data with comprehensive validation and error handling.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Processed DataFrame
            
        Raises:
            DataValidationError: If input validation fails
            CriticalPipelineError: If processing fails critically
        """
        start_time = time.time()
        engine_name = self.__class__.__name__
        
        try:
            self.logger.info(f"Starting {engine_name} processing")
            
            # Pre-processing validation
            validation_result = self._validate_input_data(df)
            if not validation_result.is_valid:
                error_msg = f"Input validation failed for {engine_name}: {validation_result.errors}"
                self.logger.error(error_msg)
                raise DataValidationError(error_msg)
            
            if validation_result.warnings:
                for warning in validation_result.warnings:
                    self.logger.warning(f"{engine_name} input warning: {warning}")
            
            # Memory check before processing
            self._check_memory_usage("before_processing")
            
            # Perform the actual processing
            result_df = self.process(df)
            
            # Post-processing validation
            if result_df is None:
                raise CriticalPipelineError(f"{engine_name} returned None result")
            
            # Validate output
            output_validation = self._validate_output_data(result_df)
            if not output_validation.is_valid:
                error_msg = f"Output validation failed for {engine_name}: {output_validation.errors}"
                self.logger.error(error_msg)
                raise DataValidationError(error_msg)
            
            # Memory check after processing
            self._check_memory_usage("after_processing")
            
            # Record processing time
            processing_time = time.time() - start_time
            self.processing_times[engine_name] = processing_time
            
            self.logger.info(f"{engine_name} completed successfully in {processing_time:.2f}s")
            
            return result_df
            
        except (DataValidationError, CriticalPipelineError):
            # Re-raise these specific exceptions
            raise
        except Exception as e:
            # Log and raise as critical error
            error_msg = f"Unexpected error in {engine_name}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self._critical_error(error_msg)
    
    def _get_required_columns(self) -> List[str]:
        """
        Get the list of required columns for this engine.
        
        Override in subclasses to specify required columns.
        
        Returns:
            List of required column names
        """
        return []
    
    def _get_optional_columns(self) -> List[str]:
        """
        Get the list of optional columns for this engine.
        
        Override in subclasses to specify optional columns.
        
        Returns:
            List of optional column names
        """
        return []
    
    def _get_expected_dtypes(self) -> Dict[str, str]:
        """
        Get the expected data types for columns.
        
        Override in subclasses to specify expected data types.
        
        Returns:
            Dictionary mapping column names to expected data types
        """
        return {}
    
    def _validate_input_data(self, df: Union[cudf.DataFrame, dask_cudf.DataFrame]) -> ValidationResult:
        """
        Comprehensive validation of input data.
        
        Args:
            df: Input DataFrame to validate
            
        Returns:
            ValidationResult with validation status and issues
        """
        result = ValidationResult(is_valid=True)
        
        try:
            # Check if DataFrame is None or empty
            if df is None:
                result.is_valid = False
                result.errors.append("Input DataFrame is None")
                return result
            
            # Handle Dask/cuDF differences for emptiness and metrics
            is_dask = isinstance(df, dask_cudf.DataFrame)
            if is_dask:
                try:
                    # Use head(1) to check emptiness without full compute
                    if df.head(1).shape[0] == 0:
                        result.is_valid = False
                        result.errors.append("Input DataFrame is empty")
                        return result
                except Exception:
                    # If we cannot determine, assume non-empty and continue
                    pass
            else:
                if len(df) == 0:
                    result.is_valid = False
                    result.errors.append("Input DataFrame is empty")
                    return result
            
            # Check for required columns
            missing_required = [col for col in self.required_columns if col not in df.columns]
            if missing_required:
                result.is_valid = False
                result.errors.append(f"Missing required columns: {missing_required}")
            
            # Check data types
            for col, expected_dtype in self.expected_dtypes.items():
                if col in df.columns:
                    actual_dtype = str(df[col].dtype)
                    if actual_dtype != expected_dtype:
                        result.warnings.append(f"Column {col} has dtype {actual_dtype}, expected {expected_dtype}")
            
            # Check for NaN values in critical columns (best-effort for Dask)
            if self.required_columns:
                for col in self.required_columns:
                    if col in df.columns:
                        try:
                            if is_dask:
                                # Sample-based check to avoid full compute
                                sample = df[[col]].head(1000)
                                nan_count = int(sample[col].isna().sum())
                                sample_len = len(sample)
                                if sample_len > 0 and nan_count > 0:
                                    nan_percentage = (nan_count / sample_len) * 100
                                    result.warnings.append(f"Column {col} has NaNs (sample {nan_percentage:.1f}%)")
                            else:
                                nan_count = int(df[col].isna().sum())
                                if nan_count > 0:
                                    nan_percentage = (nan_count / len(df)) * 100
                                    if nan_percentage > 50:
                                        result.errors.append(f"Column {col} has {nan_percentage:.1f}% NaN values")
                                    else:
                                        result.warnings.append(f"Column {col} has {nan_count} NaN values ({nan_percentage:.1f}%)")
                        except Exception:
                            # Non-fatal validation issue
                            pass
            
            # Check for infinite values (best-effort)
            try:
                if is_dask:
                    sample = df.head(1000)
                    inf_counts = sample.isin([cp.inf, -cp.inf]).sum()
                    high_inf_cols = inf_counts[inf_counts > 0]
                    if len(high_inf_cols) > 0:
                        result.warnings.append(f"Infinite values found (sample) in columns: {list(high_inf_cols.index)}")
                else:
                    inf_counts = df.isin([cp.inf, -cp.inf]).sum()
                    high_inf_cols = inf_counts[inf_counts > 0]
                    if len(high_inf_cols) > 0:
                        result.warnings.append(f"Infinite values found in columns: {list(high_inf_cols.index)}")
            except Exception:
                pass
            
            # Check for duplicate column names
            if len(df.columns) != len(set(df.columns)):
                result.errors.append("Duplicate column names found")
            
            # Memory usage check (skip heavy compute for Dask)
            if hasattr(df, 'memory_usage') and not is_dask:
                try:
                    memory_usage = df.memory_usage(deep=True).sum()
                    memory_gb = memory_usage / (1024**3)
                    if memory_gb > 8:  # 8GB threshold
                        result.warnings.append(f"Large DataFrame detected: {memory_gb:.2f} GB")
                except Exception:
                    pass
            
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Validation error: {str(e)}")
        
        return result
    
    def _validate_output_data(self, df: Union[cudf.DataFrame, dask_cudf.DataFrame]) -> ValidationResult:
        """
        Validate the output data after processing.
        
        Args:
            df: Output DataFrame to validate
            
        Returns:
            ValidationResult with validation status and issues
        """
        result = ValidationResult(is_valid=True)
        
        try:
            # Basic checks
            if df is None:
                result.is_valid = False
                result.errors.append("Output DataFrame is None")
                return result
            
            is_dask = isinstance(df, dask_cudf.DataFrame)
            if is_dask:
                try:
                    if df.head(1).shape[0] == 0:
                        result.is_valid = False
                        result.errors.append("Output DataFrame is empty")
                        return result
                except Exception:
                    pass
            else:
                if len(df) == 0:
                    result.is_valid = False
                    result.errors.append("Output DataFrame is empty")
                    return result
            
            # Check for new columns (should have added features)
            if hasattr(self, '_input_columns'):
                new_columns = [col for col in df.columns if col not in self._input_columns]
                if not new_columns:
                    result.warnings.append("No new columns were added during processing")
            
            # Check for excessive NaN values in new columns
            if hasattr(self, '_input_columns'):
                new_columns = [col for col in df.columns if col not in self._input_columns]
                for col in new_columns:
                    try:
                        if is_dask:
                            sample = df[[col]].head(1000)
                            nan_count = int(sample[col].isna().sum())
                            sample_len = len(sample)
                            if sample_len > 0:
                                nan_percentage = (nan_count / sample_len) * 100
                                if nan_percentage > 80:
                                    result.warnings.append(f"New column {col} has high NaNs in sample ({nan_percentage:.1f}%)")
                        else:
                            nan_count = int(df[col].isna().sum())
                            nan_percentage = (nan_count / len(df)) * 100
                            if nan_percentage > 80:
                                result.warnings.append(f"New column {col} has {nan_percentage:.1f}% NaN values")
                    except Exception:
                        pass
            
            # Check for infinite values in new columns
            if hasattr(self, '_input_columns'):
                new_columns = [col for col in df.columns if col not in self._input_columns]
                for col in new_columns:
                    try:
                        if is_dask:
                            sample = df[[col]].head(1000)
                            inf_count = int(sample[col].isin([cp.inf, -cp.inf]).sum())
                            if inf_count > 0:
                                result.warnings.append(f"New column {col} has infinite values (sample)")
                        else:
                            inf_count = int(df[col].isin([cp.inf, -cp.inf]).sum())
                            if inf_count > 0:
                                result.warnings.append(f"New column {col} has {inf_count} infinite values")
                    except Exception:
                        pass
            
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Output validation error: {str(e)}")
        
        return result
    
    def _check_memory_usage(self, stage: str):
        """
        Check and log memory usage at different stages.
        
        Args:
            stage: Stage identifier for logging (kept for DB compatibility)
        """
        try:
            import psutil
            
            # Convert stage to operation name for logging
            operation_name = stage.replace('_', ' ').title()
            
            # System memory
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_gb = memory.used / (1024**3)
            
            self.memory_usage[stage] = {
                'system_percent': memory_percent,
                'system_gb': memory_gb
            }
            
            if memory_percent > 85:
                self.logger.warning(f"High system memory usage during {operation_name}: {memory_percent:.1f}% ({memory_gb:.2f} GB)")
            
            # GPU memory if available
            try:
                gpu_memory = cp.cuda.runtime.memGetInfo()
                gpu_used_gb = (gpu_memory[1] - gpu_memory[0]) / (1024**3)
                gpu_total_gb = gpu_memory[1] / (1024**3)
                gpu_percent = (gpu_used_gb / gpu_total_gb) * 100
                
                self.memory_usage[stage]['gpu_percent'] = gpu_percent
                self.memory_usage[stage]['gpu_gb'] = gpu_used_gb
                
                if gpu_percent > 85:
                    self.logger.warning(f"High GPU memory usage during {operation_name}: {gpu_percent:.1f}% ({gpu_used_gb:.2f} GB)")
                    
            except Exception as e:
                self.logger.debug(f"Could not get GPU memory info: {e}")
                
        except ImportError:
            self.logger.debug("psutil not available for memory monitoring")
        except Exception as e:
            self.logger.debug(f"Error checking memory usage: {e}")
    
    @contextmanager
    def _memory_monitoring(self, operation_name: str):
        """
        Context manager for monitoring memory during operations.
        
        Args:
            operation_name: Name of the operation for logging
        """
        self._check_memory_usage(f"before_{operation_name}")
        start_time = time.time()
        
        try:
            yield
        finally:
            processing_time = time.time() - start_time
            self._check_memory_usage(f"after_{operation_name}")
            self.logger.debug(f"{operation_name} completed in {processing_time:.2f}s")
    
    def _log_info(self, msg: str, **fields):
        """Log info message with optional fields."""
        if fields:
            self.logger.info(f"{msg} | {fields}")
        else:
            self.logger.info(msg)

    def _log_warn(self, msg: str, **fields):
        """Log warning message with optional fields."""
        if fields:
            self.logger.warning(f"{msg} | {fields}")
        else:
            self.logger.warning(msg)

    def _log_error(self, msg: str, **fields):
        """Log error message with optional fields."""
        if fields:
            self.logger.error(f"{msg} | {fields}")
        else:
            self.logger.error(msg)

    def _critical_error(self, msg: str, **fields):
        """
        Log critical error and raise a special exception that will be caught by the main process.
        This method should be called when any error occurs that should stop the pipeline.
        """
        if fields:
            error_msg = f"{msg} | {fields}"
        else:
            error_msg = msg
        
        self.logger.error(f"🚨 CRITICAL ERROR: {error_msg}")
        self.logger.error("🛑 Stopping pipeline immediately due to critical error.")
        
        # Check memory usage before raising exception
        self._check_memory_usage("critical_error")
        
        # Raise a special exception that will be caught by the main process
        raise CriticalPipelineError(error_msg)
    
    def _ensure_sorted(self, df: dask_cudf.DataFrame, by: str) -> dask_cudf.DataFrame:
        """
        Ensure global sorting by column (deterministic shuffle in Dask).
        
        Args:
            df: Input DataFrame
            by: Column to sort by
            
        Returns:
            Sorted DataFrame
        """
        return df.set_index(by, shuffle="tasks")

    def _broadcast_scalars(self, df: dask_cudf.DataFrame, mapping: Mapping[str, Any]) -> dask_cudf.DataFrame:
        """
        Add scalar columns to DataFrame in a lazy and cheap way.
        
        Args:
            df: Input DataFrame
            mapping: Dictionary of column_name -> scalar_value
            
        Returns:
            DataFrame with new scalar columns
        """
        # Avoid broadcasting large string scalars across Dask DataFrames (can explode GPU memory)
        try:
            is_dask = hasattr(df, 'map_partitions')
        except Exception:
            is_dask = False

        if not is_dask:
            # cuDF path: usually smaller; keep behavior
            return df.assign(**mapping)

        safe: Dict[str, Any] = {}
        skipped: Dict[str, int] = {}
        for k, v in mapping.items():
            try:
                if isinstance(v, (int, float, bool)):
                    safe[k] = v
                elif isinstance(v, str):
                    # Only broadcast short strings to avoid large allocations on all rows
                    if len(v) <= 128:
                        safe[k] = v
                    else:
                        skipped[k] = len(v)
                else:
                    # Skip complex types/lists/dicts
                    skipped[k] = -1
            except Exception:
                skipped[k] = -1

        if skipped:
            try:
                self.logger.info(
                    "Skipping broadcast of large/complex scalars on Dask",
                    extra={
                        'skipped_keys': list(skipped.keys())[:10],
                        'counts': {k: skipped[k] for k in list(skipped.keys())[:10]},
                    }
                )
            except Exception:
                pass

        return df.assign(**safe) if safe else df

    def _single_partition(self, df: dask_cudf.DataFrame, cols: Optional[list] = None) -> dask_cudf.DataFrame:
        """
        Force DataFrame to single partition (entire series in one GPU worker).
        
        Args:
            df: Input DataFrame
            cols: Columns to select (optional)
            
        Returns:
            Single partition DataFrame
        """
        sub = df[cols] if cols else df
        
        # Check if DataFrame has repartition method (Dask DataFrame)
        if hasattr(sub, 'repartition'):
            return sub.repartition(npartitions=1)
        else:
            # If it's a regular cuDF DataFrame, return as is
            return sub

    def _log_progress(self, message: str, **kwargs):
        """
        Helper method for consistent logging (legacy compatibility).
        """
        self._log_info(message, **kwargs)
    
    def _cleanup_memory(self):
        """Clean up memory and force garbage collection."""
        try:
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            self.logger.debug("Memory cleanup completed")
        except Exception as e:
            self.logger.debug(f"Error during memory cleanup: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for this engine.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            'processing_times': self.processing_times,
            'memory_usage': self.memory_usage,
            'engine_name': self.__class__.__name__
        }
