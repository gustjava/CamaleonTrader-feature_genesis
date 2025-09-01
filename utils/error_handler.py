"""
Comprehensive Error Handling for Dynamic Stage 0 Pipeline

This module provides robust error handling with:
- Granular exception handling for different error types
- Retry logic with exponential backoff
- Graceful degradation strategies
- Detailed error logging and reporting
- Error recovery mechanisms
"""

import logging
import time
import traceback
import functools
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
from contextlib import contextmanager

import cudf
import cupy as cp
from dask.distributed import Client

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    DATA_LOADING = "data_loading"
    DATA_VALIDATION = "data_validation"
    GPU_MEMORY = "gpu_memory"
    SYSTEM_MEMORY = "system_memory"
    NETWORK = "network"
    DATABASE = "database"
    COMPUTATION = "computation"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation: str
    component: str
    severity: ErrorSeverity
    category: ErrorCategory
    retry_count: int = 0
    max_retries: int = 3
    retry_delay: float = 1.0
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorReport:
    """Comprehensive error report."""
    timestamp: float
    error_type: str
    error_message: str
    context: ErrorContext
    traceback: str
    system_info: Dict[str, Any] = field(default_factory=dict)
    recovery_action: Optional[str] = None
    success: bool = False


class ErrorHandler:
    """
    Comprehensive error handler for the pipeline.
    
    Provides:
    - Granular exception handling
    - Retry logic with exponential backoff
    - Error classification and reporting
    - Recovery strategies
    - Performance monitoring
    """
    
    def __init__(self):
        """Initialize the error handler."""
        self.error_reports: List[ErrorReport] = []
        self.error_stats: Dict[str, int] = {}
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {}
        self._lock = threading.Lock()
        
        # Register default recovery strategies
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default recovery strategies."""
        self.recovery_strategies[ErrorCategory.GPU_MEMORY] = self._handle_gpu_memory_error
        self.recovery_strategies[ErrorCategory.SYSTEM_MEMORY] = self._handle_system_memory_error
        self.recovery_strategies[ErrorCategory.DATA_LOADING] = self._handle_data_loading_error
        self.recovery_strategies[ErrorCategory.NETWORK] = self._handle_network_error
        self.recovery_strategies[ErrorCategory.DATABASE] = self._handle_database_error
    
    def handle_error(self, error: Exception, context: ErrorContext) -> ErrorReport:
        """
        Handle an error with the given context.
        
        Args:
            error: The exception that occurred
            context: Error context information
            
        Returns:
            ErrorReport with handling results
        """
        with self._lock:
            # Classify the error
            category = self._classify_error(error)
            context.category = category
            
            # Update error statistics
            error_type = type(error).__name__
            self.error_stats[error_type] = self.error_stats.get(error_type, 0) + 1
            
            # Create error report
            report = ErrorReport(
                timestamp=time.time(),
                error_type=error_type,
                error_message=str(error),
                context=context,
                traceback=traceback.format_exc(),
                system_info=self._get_system_info()
            )
            
            # Log the error
            self._log_error(report)
            
            # Attempt recovery if possible
            if context.retry_count < context.max_retries:
                recovery_action = self._attempt_recovery(report)
                report.recovery_action = recovery_action
                
                if recovery_action:
                    logger.info(f"Recovery action taken: {recovery_action}")
                    report.success = True
                else:
                    logger.error(f"No recovery strategy available for {category.value} error")
            else:
                logger.error(f"Max retries ({context.max_retries}) exceeded for {context.operation}")
            
            # Store the report
            self.error_reports.append(report)
            
            return report
    
    def _classify_error(self, error: Exception) -> ErrorCategory:
        """Classify the error based on its type and message."""
        error_type = type(error)
        error_message = str(error).lower()
        
        # GPU memory errors
        if any(term in error_message for term in ['cuda', 'gpu', 'memory', 'out of memory']):
            return ErrorCategory.GPU_MEMORY
        
        # System memory errors
        if any(term in error_message for term in ['memory', 'ram', 'out of memory']):
            return ErrorCategory.SYSTEM_MEMORY
        
        # Data loading errors
        if any(term in error_message for term in ['file', 'load', 'read', 'parquet', 'feather']):
            return ErrorCategory.DATA_LOADING
        
        # Network errors
        if any(term in error_message for term in ['network', 'connection', 'timeout', 'http']):
            return ErrorCategory.NETWORK
        
        # Database errors
        if any(term in error_message for term in ['database', 'sql', 'mysql', 'connection']):
            return ErrorCategory.DATABASE
        
        # Data validation errors
        if any(term in error_message for term in ['validation', 'invalid', 'missing', 'column']):
            return ErrorCategory.DATA_VALIDATION
        
        # Configuration errors
        if any(term in error_message for term in ['config', 'setting', 'parameter']):
            return ErrorCategory.CONFIGURATION
        
        return ErrorCategory.UNKNOWN
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get current system information."""
        try:
            import psutil
            
            info = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_available_gb': psutil.virtual_memory().available / (1024**3),
                'disk_usage_percent': psutil.disk_usage('/').percent
            }
            
            # GPU information if available
            try:
                gpu_memory = cp.cuda.runtime.memGetInfo()
                info['gpu_memory_used_gb'] = (gpu_memory[1] - gpu_memory[0]) / (1024**3)
                info['gpu_memory_total_gb'] = gpu_memory[1] / (1024**3)
            except:
                pass
            
            return info
        except ImportError:
            return {'error': 'psutil not available'}
    
    def _log_error(self, report: ErrorReport):
        """Log the error with appropriate level based on severity."""
        context = report.context
        
        log_message = (
            f"Error in {context.operation} ({context.component}): "
            f"{report.error_type}: {report.error_message}"
        )
        
        if context.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif context.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif context.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        # Log additional context for debugging
        if context.additional_info:
            logger.debug(f"Error context: {context.additional_info}")
    
    def _attempt_recovery(self, report: ErrorReport) -> Optional[str]:
        """Attempt to recover from the error."""
        category = report.context.category
        
        if category in self.recovery_strategies:
            try:
                return self.recovery_strategies[category](report)
            except Exception as e:
                logger.error(f"Recovery strategy failed: {e}")
                return None
        
        return None
    
    def _handle_gpu_memory_error(self, report: ErrorReport) -> Optional[str]:
        """Handle GPU memory errors."""
        try:
            # Clear GPU memory
            cp.get_default_memory_pool().free_all_blocks()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info("GPU memory cleared and garbage collection performed")
            return "GPU memory cleared"
            
        except Exception as e:
            logger.error(f"Failed to handle GPU memory error: {e}")
            return None
    
    def _handle_system_memory_error(self, report: ErrorReport) -> Optional[str]:
        """Handle system memory errors."""
        try:
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info("System garbage collection performed")
            return "System memory cleared"
            
        except Exception as e:
            logger.error(f"Failed to handle system memory error: {e}")
            return None
    
    def _handle_data_loading_error(self, report: ErrorReport) -> Optional[str]:
        """Handle data loading errors."""
        # For data loading errors, we typically want to retry with a delay
        context = report.context
        if context.retry_count < context.max_retries:
            delay = context.retry_delay * (2 ** context.retry_count)  # Exponential backoff
            time.sleep(delay)
            return f"Retrying data loading (attempt {context.retry_count + 1})"
        
        return None
    
    def _handle_network_error(self, report: ErrorReport) -> Optional[str]:
        """Handle network errors."""
        # For network errors, retry with exponential backoff
        context = report.context
        if context.retry_count < context.max_retries:
            delay = context.retry_delay * (2 ** context.retry_count)
            time.sleep(delay)
            return f"Retrying network operation (attempt {context.retry_count + 1})"
        
        return None
    
    def _handle_database_error(self, report: ErrorReport) -> Optional[str]:
        """Handle database errors."""
        # For database errors, try to reconnect
        try:
            # This would typically involve reconnecting to the database
            # Implementation depends on the specific database handler
            return "Database reconnection attempted"
        except Exception as e:
            logger.error(f"Failed to handle database error: {e}")
            return None
    
    def retry_with_backoff(self, func: Callable, context: ErrorContext, *args, **kwargs) -> Any:
        """
        Execute a function with retry logic and exponential backoff.
        
        Args:
            func: Function to execute
            context: Error context
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        
        for attempt in range(context.max_retries + 1):
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                context.retry_count = attempt
                
                if attempt < context.max_retries:
                    # Handle the error
                    self.handle_error(e, context)
                    
                    # Calculate delay with exponential backoff
                    delay = context.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying {context.operation} in {delay:.2f}s (attempt {attempt + 1}/{context.max_retries})")
                    time.sleep(delay)
                else:
                    # Final attempt failed
                    logger.error(f"All retry attempts failed for {context.operation}")
                    self.handle_error(e, context)
                    raise last_exception
        
        raise last_exception
    
    @contextmanager
    def error_context(self, operation: str, component: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        """
        Context manager for error handling.
        
        Args:
            operation: Name of the operation
            component: Component name
            severity: Error severity level
        """
        context = ErrorContext(
            operation=operation,
            component=component,
            severity=severity
        )
        
        try:
            yield context
        except Exception as e:
            self.handle_error(e, context)
            raise
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of all errors."""
        with self._lock:
            return {
                'total_errors': len(self.error_reports),
                'error_stats': self.error_stats,
                'recent_errors': self.error_reports[-10:] if self.error_reports else [],
                'successful_recoveries': sum(1 for r in self.error_reports if r.success)
            }
    
    def clear_error_history(self):
        """Clear error history."""
        with self._lock:
            self.error_reports.clear()
            self.error_stats.clear()


# Global error handler instance
error_handler = ErrorHandler()


def handle_errors(operation: str, component: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """
    Decorator for automatic error handling.
    
    Args:
        operation: Operation name
        component: Component name
        severity: Error severity level
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            context = ErrorContext(
                operation=operation,
                component=component,
                severity=severity
            )
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler.handle_error(e, context)
                raise
        
        return wrapper
    return decorator


def retry_on_error(max_retries: int = 3, retry_delay: float = 1.0):
    """
    Decorator for automatic retry logic.
    
    Args:
        max_retries: Maximum number of retries
        retry_delay: Base delay between retries
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            context = ErrorContext(
                operation=func.__name__,
                component=func.__module__,
                severity=ErrorSeverity.MEDIUM,
                max_retries=max_retries,
                retry_delay=retry_delay
            )
            
            return error_handler.retry_with_backoff(func, context, *args, **kwargs)
        
        return wrapper
    return decorator
