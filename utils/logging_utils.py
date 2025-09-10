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

from .log_context import get_context


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
            
            # Merge context with extra fields
            extra = kwargs.get('extra', {})
            extra.update(context)
            extra['component'] = self.component
            
            # Add timestamp if not present
            if 'timestamp' not in extra:
                extra['timestamp'] = datetime.utcnow().isoformat() + 'Z'
            
            kwargs['extra'] = extra
            return msg, kwargs


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
        
        # Apply defaults for missing fields
        for key, default_value in defaults.items():
            if not hasattr(record, key):
                setattr(record, key, default_value)
        
        # Apply extra fields if provided
        if extra:
            for key, value in extra.items():
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


def get_logger(name: str, component: Optional[str] = None) -> ContextualLoggerAdapter:
    """
    Get a contextual logger adapter for the given name.
    
    Args:
        name: Logger name (usually __name__)
        component: Optional component name override
        
    Returns:
        ContextualLoggerAdapter instance
    """
    base_logger = logging.getLogger(name)
    return ContextualLoggerAdapter(base_logger, component)


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
        logger.info(message, extra={'event': event, **fields})
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
        logger.warning(message, extra={'event': event, **fields})
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
        logger.error(message, extra={'event': event, **fields})
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
        logger.critical(message, extra={'event': event, **fields})
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
        logger.debug(message, extra={'event': event, **fields})
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
        for attr in ['run_id', 'task_id', 'pair', 'engine', 'hostname', 'gpu_count', 
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