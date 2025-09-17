"""
Logging utilities for structured logging with context enrichment.

This module provides:
- LoggerAdapter for automatic context injection
- Event-based logging helpers
- LogRecordFactory for default field handling
- Component name derivation utilities
"""

import logging
import logging.handlers
import sys
import time
from typing import Any, Dict, Optional, Union
from datetime import datetime

from .log_context import get_context, set_task


class GPUMemoryWarningFilter(logging.Filter):
    """
    Filter to suppress specific GPU memory warning messages from CatBoost and other libraries.
    """
    
    def filter(self, record):
        # Check if this is the specific GPU memory warning we want to suppress
        if hasattr(record, 'getMessage'):
            message = record.getMessage()
        else:
            message = record.msg % record.args if hasattr(record, 'args') else str(record.msg)
        
        # Suppress the specific GPU memory warning (with variations)
        warning_patterns = [
            "less than 75% GPU memory available for training",
            "Warning: less than 75% GPU memory available for training",
            "WARNING: less than 75% GPU memory available for training",
            "75% GPU memory available for training",
            "GPU memory available for training",
            # Dask warnings about insufficient elements
            "Insufficient elements for `head`",
            "elements requested, only",
            "elements available. Try passing larger `npartitions` to `head`",
            # Dask large graph warnings
            "Sending large graph of size",
            "This may cause some slowdown",
            "Consider loading the data with Dask directly",
            "or using futures or delayed objects to embed the data into the graph without repetition"
        ]
        
        for pattern in warning_patterns:
            if pattern in message:
                return False
        
        # Allow all other messages
        return True


class GPUMemoryWarningSuppressor:
    """
    Alternative approach to suppress GPU memory warnings by intercepting stdout/stderr.
    """
    
    def __init__(self):
        self.original_stdout = None
        self.original_stderr = None
        self.suppressed_count = 0
    
    def _should_suppress(self, text):
        """Check if the text should be suppressed."""
        warning_patterns = [
            "less than 75% GPU memory available for training",
            "Warning: less than 75% GPU memory available for training",
            "WARNING: less than 75% GPU memory available for training",
            "75% GPU memory available for training",
            "GPU memory available for training",
            # Dask warnings about insufficient elements
            "Insufficient elements for `head`",
            "elements requested, only",
            "elements available. Try passing larger `npartitions` to `head`",
            # Dask large graph warnings
            "Sending large graph of size",
            "This may cause some slowdown",
            "Consider loading the data with Dask directly",
            "or using futures or delayed objects to embed the data into the graph without repetition"
        ]
        
        for pattern in warning_patterns:
            if pattern in text:
                return True
        return False
    
    def _filtered_write(self, original_write, text):
        """Write function that filters out GPU memory warnings."""
        if self._should_suppress(text):
            self.suppressed_count += 1
            return  # Suppress the message
        return original_write(text)
    
    def start_suppression(self):
        """Start suppressing GPU memory warnings."""
        import sys
        import io
        
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Create filtered stdout
        class FilteredStdout:
            def __init__(self, original, suppressor):
                self.original = original
                self.suppressor = suppressor
            
            def write(self, text):
                return self.suppressor._filtered_write(self.original.write, text)
            
            def flush(self):
                return self.original.flush()
            
            def __getattr__(self, name):
                return getattr(self.original, name)
        
        sys.stdout = FilteredStdout(self.original_stdout, self)
        sys.stderr = FilteredStdout(self.original_stderr, self)
    
    def stop_suppression(self):
        """Stop suppressing GPU memory warnings."""
        import sys
        if self.original_stdout:
            sys.stdout = self.original_stdout
        if self.original_stderr:
            sys.stderr = self.original_stderr
    
    def get_suppressed_count(self):
        """Get the number of suppressed messages."""
        return self.suppressed_count


# Global suppressor instance
_gpu_warning_suppressor = None

def apply_gpu_memory_warning_filter():
    """
    Apply the GPU memory warning filter to suppress specific CatBoost warnings.
    This should be called early in the application startup.
    """
    global _gpu_warning_suppressor
    
    # Use the stdout/stderr interception approach
    _gpu_warning_suppressor = GPUMemoryWarningSuppressor()
    _gpu_warning_suppressor.start_suppression()
    
    # Also apply the logging filter as backup
    gpu_filter = GPUMemoryWarningFilter()
    
    # Apply to root logger to catch all warnings
    root_logger = logging.getLogger()
    root_logger.addFilter(gpu_filter)
    
    # Also apply to all existing handlers
    for handler in root_logger.handlers:
        handler.addFilter(gpu_filter)
    
    # Also apply to common library loggers that might generate this warning
    library_loggers = [
        'catboost',
        'catboost.core',
        'catboost.utils',
        'cudf',
        'cupy',
        'rapids',
        'dask',
        'distributed'
    ]
    
    for logger_name in library_loggers:
        logger = logging.getLogger(logger_name)
        logger.addFilter(gpu_filter)
        # Also apply to handlers of these loggers
        for handler in logger.handlers:
            handler.addFilter(gpu_filter)


def get_suppressed_gpu_warnings_count():
    """
    Get the number of GPU memory warnings that have been suppressed.
    """
    global _gpu_warning_suppressor
    if _gpu_warning_suppressor:
        return _gpu_warning_suppressor.get_suppressed_count()
    return 0


class ContextualLoggerAdapter(logging.LoggerAdapter):
    """
    LoggerAdapter that automatically injects context variables and component information.
    
    This adapter ensures that all log records include contextual information
    from the current execution context, making logs more structured and useful.
    """
    
    def __init__(self, logger: logging.Logger, component: Optional[str] = None):
        """
        Initialize the contextual logger adapter.
        
        Args:
            logger: The base logger to wrap
            component: Component name (defaults to logger name)
        """
        super().__init__(logger, {})
        self.component = component or self._derive_component_name(logger.name)
    
    def _derive_component_name(self, logger_name: str) -> str:
        """
        Derive a clean component name from the logger name.
        
        Args:
            logger_name: The logger's name
            
        Returns:
            Clean component name
        """
        # Convert module paths to component names
        if logger_name.startswith('orchestration.'):
            return logger_name.replace('orchestration.', 'orchestration.')
        elif logger_name.startswith('features.'):
            return logger_name.replace('features.', 'features.')
        elif logger_name.startswith('data_io.'):
            return logger_name.replace('data_io.', 'data_io.')
        elif logger_name.startswith('utils.'):
            return logger_name.replace('utils.', 'utils.')
        else:
            return logger_name
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """
        Process the log message and inject context.
        
        Args:
            msg: The log message
            kwargs: Additional keyword arguments
            
        Returns:
            Tuple of (processed_message, processed_kwargs)
        """
        # Get current context
        context = get_context()
        
        # Merge context with extra fields (avoid overwriting existing fields)
        extra = kwargs.get('extra', {})
        # Only add context fields that don't already exist in extra
        for key, value in context.items():
            if key not in extra:
                extra[key] = value
        extra['component'] = self.component
        
        # Ensure currency_pair is always available (alias for pair)
        if 'pair' in context and context['pair'] is not None:
            extra['currency_pair'] = context['pair']
        
        # Add timestamp if not present
        if 'timestamp' not in extra:
            extra['timestamp'] = datetime.utcnow().isoformat() + 'Z'
        
        kwargs['extra'] = extra
        return msg, kwargs


class DetailedPipelineLogger:
    """
    High-level structured logging facade for pipeline transparency.

    This wraps a standard logger/adapter and provides semantic methods
    aligned with the pipeline documentation goals.
    """

    def __init__(self, logger: Union[logging.Logger, ContextualLoggerAdapter]):
        self._logger = logger

    # ---- Stage lifecycle ----
    def log_stage_start(self, stage: str, input_shape: Optional[tuple] = None, config: Optional[dict] = None):
        info_event(self._logger, Events.ENGINE_START, f"Stage start: {stage}", engine=stage, input_shape=input_shape, config=config)

    def log_stage_end(self, stage: str, output_shape: Optional[tuple] = None, metrics: Optional[dict] = None):
        info_event(self._logger, Events.ENGINE_END, f"Stage end: {stage}", engine=stage, output_shape=output_shape, **(metrics or {}))

    # ---- Transformations ----
    def log_transformation(self, engine: str, operation: str,
                           input_cols: Optional[list] = None, output_cols: Optional[list] = None,
                           metrics: Optional[dict] = None, duration: Optional[float] = None):
        fields: Dict[str, Any] = {
            'engine': engine,
            'operation': operation,
            'input_cols': input_cols,
            'output_cols': output_cols,
            'duration_ms': int(duration * 1000) if isinstance(duration, (int, float)) else None,
        }
        if metrics:
            fields.update(metrics)
        info_event(self._logger, "engine.transformation", f"{engine}: {operation}", **fields)

    # ---- Feature evolution ----
    def log_feature_evolution(self, before: int, after: int,
                              added: Optional[list] = None, removed: Optional[list] = None, modified: Optional[list] = None):
        fields = {
            'cols_before': before,
            'cols_after': after,
            'new_cols': after - before if (before is not None and after is not None) else None,
            'added': (added or [])[:50],  # cap to avoid huge logs
            'removed': (removed or [])[:50],
            'modified': (modified or [])[:50],
        }
        info_event(self._logger, "engine.feature_evolution", "Feature evolution", **fields)

    # ---- Config impact ----
    def log_config_impact(self, param: str, value: Any, effect: str, metrics: Optional[dict] = None):
        fields: Dict[str, Any] = {'param': param, 'value': value, 'effect': effect}
        if metrics:
            fields.update(metrics)
        info_event(self._logger, "pipeline.config.impact", f"Config impact: {param}", **fields)


class LogRecordFactory:
    """
    Factory for creating LogRecord instances with default field values.
    
    This ensures that formatters don't fail when expected fields are missing.
    """
    
    @staticmethod
    def create_record(name: str, level: int, fn: str, lno: int, msg: str, 
                     args: tuple, exc_info: Optional[Any], 
                     func: Optional[str] = None, extra: Optional[Dict[str, Any]] = None,
                     sinfo: Optional[str] = None) -> logging.LogRecord:
        """
        Create a LogRecord with default field values.
        
        Args:
            name: Logger name
            level: Log level
            fn: Filename
            lno: Line number
            msg: Log message
            args: Message arguments
            exc_info: Exception info
            func: Function name
            extra: Extra fields
            sinfo: Stack info
            
        Returns:
            LogRecord with default fields
        """
        # Create the record
        record = logging.LogRecord(name, level, fn, lno, msg, args, exc_info, func, sinfo)
        
        # Add default fields if not present
        defaults = {
            'component': name,
            'run_id': None,
            'task_id': None,
            'pair': None,
            'currency_pair': None,  # Alternative field name for currency pair
            'engine': None,
            'hostname': None,
            'gpu_count': None,
            'workers': None,
            'dashboard_url': None,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'duration_ms': None,
            'rows_before': None,
            'rows_after': None,
            'cols_before': None,
            'cols_after': None,
            'new_cols': None,
        }
        
        # Apply defaults for missing fields only
        for key, default_value in defaults.items():
            if not hasattr(record, key):
                setattr(record, key, default_value)
        
        # Apply extra fields if provided (only if they don't already exist)
        if extra:
            for key, value in extra.items():
                if not hasattr(record, key):
                    setattr(record, key, value)
                else:
                    # If field already exists, only update if it's None
                    current_value = getattr(record, key)
                    if current_value is None and value is not None:
                        setattr(record, key, value)
        
        # Set default event if not provided
        if not hasattr(record, 'event'):
            setattr(record, 'event', 'unknown')
        
        return record


def setup_logging_factory():
    """
    Set up the custom LogRecordFactory to handle default fields.
    """
    logging.setLogRecordFactory(LogRecordFactory.create_record)


def get_logger(name: str, component: Optional[str] = None) -> logging.Logger:
    """
    Get a simple logger for the given name.
    
    Args:
        name: Logger name (usually __name__)
        component: Optional component name override (ignored for simple logging)
        
    Returns:
        Standard logging.Logger instance
    """
    return logging.getLogger(name)


def set_currency_pair_context(currency_pair: str) -> None:
    """
    Set the currency pair in the logging context for all subsequent logs.
    
    Args:
        currency_pair: The currency pair symbol (e.g., 'EURUSD')
    """
    set_task(pair=currency_pair)


# Event-based logging helpers
def info_event(logger: Union[logging.Logger, ContextualLoggerAdapter], 
               event: str, message: str, **fields: Any) -> None:
    """
    Log an info-level event with structured fields.
    
    Args:
        logger: Logger instance
        event: Event name (e.g., 'pipeline.start', 'engine.end')
        message: Log message
        **fields: Additional structured fields
    """
    if isinstance(logger, ContextualLoggerAdapter):
        # ContextualLoggerAdapter already has event field, don't overwrite
        logger.info(message, extra=fields)
    else:
        logger.info(message, extra={'event': event, **fields})


def warn_event(logger: Union[logging.Logger, ContextualLoggerAdapter], 
               event: str, message: str, **fields: Any) -> None:
    """
    Log a warning-level event with structured fields.
    
    Args:
        logger: Logger instance
        event: Event name
        message: Log message
        **fields: Additional structured fields
    """
    if isinstance(logger, ContextualLoggerAdapter):
        # ContextualLoggerAdapter already has event field, don't overwrite
        logger.warning(message, extra=fields)
    else:
        logger.warning(message, extra={'event': event, **fields})


def error_event(logger: Union[logging.Logger, ContextualLoggerAdapter], 
                event: str, message: str, **fields: Any) -> None:
    """
    Log an error-level event with structured fields.
    
    Args:
        logger: Logger instance
        event: Event name
        message: Log message
        **fields: Additional structured fields
    """
    if isinstance(logger, ContextualLoggerAdapter):
        # ContextualLoggerAdapter already has event field, don't overwrite
        logger.error(message, extra=fields)
    else:
        logger.error(message, extra={'event': event, **fields})


def critical_event(logger: Union[logging.Logger, ContextualLoggerAdapter], 
                   event: str, message: str, **fields: Any) -> None:
    """
    Log a critical-level event with structured fields.
    
    Args:
        logger: Logger instance
        event: Event name
        message: Log message
        **fields: Additional structured fields
    """
    if isinstance(logger, ContextualLoggerAdapter):
        # ContextualLoggerAdapter already has event field, don't overwrite
        logger.critical(message, extra=fields)
    else:
        logger.critical(message, extra={'event': event, **fields})


def debug_event(logger: Union[logging.Logger, ContextualLoggerAdapter], 
                event: str, message: str, **fields: Any) -> None:
    """
    Log a debug-level event with structured fields.
    
    Args:
        logger: Logger instance
        event: Event name
        message: Log message
        **fields: Additional structured fields
    """
    if isinstance(logger, ContextualLoggerAdapter):
        # ContextualLoggerAdapter already has event field, don't overwrite
        logger.debug(message, extra=fields)
    else:
        logger.debug(message, extra={'event': event, **fields})


class TimingContext:
    """
    Context manager for timing operations and logging duration.
    """
    
    def __init__(self, logger: Union[logging.Logger, ContextualLoggerAdapter], 
                 event: str, message: str, **fields: Any):
        """
        Initialize timing context.
        
        Args:
            logger: Logger instance
            event: Event name for the timing log
            message: Log message
            **fields: Additional structured fields
        """
        self.logger = logger
        self.event = event
        self.message = message
        self.fields = fields
        self.start_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        info_event(self.logger, f"{self.event}.start", self.message, **self.fields)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log duration."""
        if self.start_time:
            duration_ms = int((time.time() - self.start_time) * 1000)
            end_event = f"{self.event}.end"
            end_message = f"{self.message} completed in {duration_ms}ms"
            
            if exc_type is None:
                info_event(self.logger, end_event, end_message, 
                          duration_ms=duration_ms, **self.fields)
            else:
                error_event(self.logger, f"{self.event}.error", 
                           f"{self.message} failed after {duration_ms}ms", 
                           duration_ms=duration_ms, **self.fields)


def time_event(logger: Union[logging.Logger, ContextualLoggerAdapter], 
               event: str, message: str, **fields: Any) -> TimingContext:
    """
    Create a timing context for an event.
    
    Args:
        logger: Logger instance
        event: Event name
        message: Log message
        **fields: Additional structured fields
        
    Returns:
        TimingContext instance
        
    Example:
        with time_event(logger, "engine.processing", "Processing features"):
            # Do work here
            pass
    """
    return TimingContext(logger, event, message, **fields)


# Predefined event types for consistency
class Events:
    """Predefined event types for consistent logging."""
    
    # Pipeline events
    PIPELINE_START = "pipeline.start"
    PIPELINE_END = "pipeline.end"
    PIPELINE_SUMMARY = "pipeline.summary"
    PIPELINE_ABORT = "pipeline.abort"
    PIPELINE_ERROR = "pipeline.error"
    
    # Cluster events
    CLUSTER_START = "cluster.start"
    CLUSTER_CONFIG_RMM = "cluster.config.rmm"
    CLUSTER_CLIENT_CREATED = "cluster.client.created"
    CLUSTER_READY = "cluster.ready"
    CLUSTER_SHUTDOWN = "cluster.shutdown"
    
    # Task discovery and execution
    TASK_DISCOVERY_START = "task.discovery.start"
    TASK_DISCOVERY_FOUND = "task.discovery.found"
    TASK_QUEUE_ADDED = "task.queue.added"
    TASK_START = "task.start"
    TASK_SUCCESS = "task.success"
    TASK_FAILURE = "task.failure"
    TASK_EXECUTION_START = "task.execution.start"
    
    # Engine events
    ENGINE_START = "engine.start"
    ENGINE_END = "engine.end"
    ENGINE_VALIDATE_BEFORE = "engine.validate.before"
    ENGINE_VALIDATE_AFTER = "engine.validate.after"
    ENGINE_SAMPLING = "engine.sampling"
    ENGINE_WRAPPER_FIT = "engine.wrapper_fit"
    ENGINE_FEATURE_SELECTION = "engine.feature_selection"
    
    # I/O events
    IO_LOAD_START = "io.load.start"
    IO_LOAD_END = "io.load.end"
    IO_SAVE_START = "io.save.start"
    IO_SAVE_END = "io.save.end"
    
    # Monitoring events
    MEMORY_ALERT = "memory.alert"
    GPU_ALERT = "gpu.alert"


class ConsoleFilter(logging.Filter):
    """
    Filter for console output to clean up None values and make logs more readable.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter and clean log records for console output.
        
        Args:
            record: Log record to filter
            
        Returns:
            True if record should be logged
        """
        # Replace None values with empty strings for cleaner console output
        for attr in ['run_id', 'task_id', 'pair', 'currency_pair', 'engine', 'hostname', 'gpu_count', 
                     'workers', 'dashboard_url', 'duration_ms', 'rows_before', 
                     'rows_after', 'cols_before', 'cols_after', 'new_cols']:
            if hasattr(record, attr) and getattr(record, attr) is None:
                setattr(record, attr, '')
        
        # Ensure event field is present
        if not hasattr(record, 'event') or record.event is None:
            record.event = 'unknown'
        
        # Ensure component field is present
        if not hasattr(record, 'component') or record.component is None:
            record.component = record.name
        
        return True


class JsonFilter(logging.Filter):
    """
    Filter for JSON output to ensure all required fields are present.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter and enrich log records for JSON output.
        
        Args:
            record: Log record to filter
            
        Returns:
            True if record should be logged
        """
        # Ensure all required fields are present with defaults
        defaults = {
            'event': 'unknown',
            'component': record.name,
            'run_id': None,
            'task_id': None,
            'pair': None,
            'currency_pair': None,  # Alternative field name for currency pair
            'engine': None,
            'hostname': None,
            'gpu_count': None,
            'workers': None,
            'dashboard_url': None,
            'duration_ms': None,
            'rows_before': None,
            'rows_after': None,
            'cols_before': None,
            'cols_after': None,
            'new_cols': None,
        }
        
        for key, default_value in defaults.items():
            if not hasattr(record, key):
                setattr(record, key, default_value)
        
        return True


# Initialize the logging factory when module is imported
setup_logging_factory()
