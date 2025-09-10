from contextvars import ContextVar, copy_context
from typing import Optional, Dict, Any, Callable, TypeVar
import functools
import threading

# Context variables for logging enrichment
_run_id: ContextVar[Optional[int]] = ContextVar("run_id", default=None)
_task_id: ContextVar[Optional[int]] = ContextVar("task_id", default=None)
_pair: ContextVar[Optional[str]] = ContextVar("pair", default=None)
_engine: ContextVar[Optional[str]] = ContextVar("engine", default=None)
_hostname: ContextVar[Optional[str]] = ContextVar("hostname", default=None)
_gpu_count: ContextVar[Optional[int]] = ContextVar("gpu_count", default=None)
_workers: ContextVar[Optional[int]] = ContextVar("workers", default=None)
_dashboard_url: ContextVar[Optional[str]] = ContextVar("dashboard_url", default=None)

# Thread-local storage for Dask worker compatibility
_local = threading.local()

T = TypeVar('T')


def set_run_id(run_id: Optional[int]) -> None:
    """Set the run ID in the current context."""
    _run_id.set(run_id)


def set_task(task_id: Optional[int] = None, pair: Optional[str] = None) -> None:
    """Set task ID and pair in the current context."""
    if task_id is not None:
        _task_id.set(task_id)
    if pair is not None:
        _pair.set(pair)


def set_engine(engine: Optional[str]) -> None:
    """Set the engine name in the current context."""
    _engine.set(engine)


def set_cluster_info(hostname: Optional[str] = None, gpu_count: Optional[int] = None, 
                    workers: Optional[int] = None, dashboard_url: Optional[str] = None) -> None:
    """Set cluster information in the current context."""
    if hostname is not None:
        _hostname.set(hostname)
    if gpu_count is not None:
        _gpu_count.set(gpu_count)
    if workers is not None:
        _workers.set(workers)
    if dashboard_url is not None:
        _dashboard_url.set(dashboard_url)


def clear() -> None:
    """Clear all context variables."""
    _run_id.set(None)
    _task_id.set(None)
    _pair.set(None)
    _engine.set(None)
    _hostname.set(None)
    _gpu_count.set(None)
    _workers.set(None)
    _dashboard_url.set(None)


def get_context() -> Dict[str, Optional[object]]:
    """Get all current context values as a dictionary."""
    return {
        "run_id": _run_id.get(),
        "task_id": _task_id.get(),
        "pair": _pair.get(),
        "engine": _engine.get(),
        "hostname": _hostname.get(),
        "gpu_count": _gpu_count.get(),
        "workers": _workers.get(),
        "dashboard_url": _dashboard_url.get(),
    }


def bind_context(**kwargs: Any) -> Callable[[T], T]:
    """
    Decorator to bind context variables for a function execution.
    Thread-safe and Dask-compatible.
    
    Args:
        **kwargs: Context variables to bind (run_id, task_id, pair, engine, etc.)
    
    Example:
        @bind_context(run_id=42, pair="EURUSD")
        def process_pair():
            # This function will have run_id=42 and pair="EURUSD" in context
            pass
    """
    def decorator(func: T) -> T:
        @functools.wraps(func)
        def wrapper(*args, **func_kwargs):
            # Store current context
            current_context = copy_context()
            
            # Set new context variables
            for key, value in kwargs.items():
                if key == "run_id":
                    _run_id.set(value)
                elif key == "task_id":
                    _task_id.set(value)
                elif key == "pair":
                    _pair.set(value)
                elif key == "engine":
                    _engine.set(value)
                elif key == "hostname":
                    _hostname.set(value)
                elif key == "gpu_count":
                    _gpu_count.set(value)
                elif key == "workers":
                    _workers.set(value)
                elif key == "dashboard_url":
                    _dashboard_url.set(value)
            
            try:
                return func(*args, **func_kwargs)
            finally:
                # Restore original context
                for key in kwargs.keys():
                    if key == "run_id":
                        _run_id.set(current_context.get(_run_id))
                    elif key == "task_id":
                        _task_id.set(current_context.get(_task_id))
                    elif key == "pair":
                        _pair.set(current_context.get(_pair))
                    elif key == "engine":
                        _engine.set(current_context.get(_engine))
                    elif key == "hostname":
                        _hostname.set(current_context.get(_hostname))
                    elif key == "gpu_count":
                        _gpu_count.set(current_context.get(_gpu_count))
                    elif key == "workers":
                        _workers.set(current_context.get(_workers))
                    elif key == "dashboard_url":
                        _dashboard_url.set(current_context.get(_dashboard_url))
        
        return wrapper
    return decorator


def with_context(**kwargs: Any) -> Callable[[T], T]:
    """
    Context manager alternative to bind_context for use in with statements.
    Thread-safe and Dask-compatible.
    
    Args:
        **kwargs: Context variables to bind
    
    Example:
        with with_context(run_id=42, pair="EURUSD"):
            # Code here will have the context bound
            pass
    """
    class ContextManager:
        def __init__(self, **context_kwargs):
            self.context_kwargs = context_kwargs
            self.original_context = None
            
        def __enter__(self):
            # Store current context
            self.original_context = copy_context()
            
            # Set new context variables
            for key, value in self.context_kwargs.items():
                if key == "run_id":
                    _run_id.set(value)
                elif key == "task_id":
                    _task_id.set(value)
                elif key == "pair":
                    _pair.set(value)
                elif key == "engine":
                    _engine.set(value)
                elif key == "hostname":
                    _hostname.set(value)
                elif key == "gpu_count":
                    _gpu_count.set(value)
                elif key == "workers":
                    _workers.set(value)
                elif key == "dashboard_url":
                    _dashboard_url.set(value)
            
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            # Restore original context
            if self.original_context:
                for key in self.context_kwargs.keys():
                    if key == "run_id":
                        _run_id.set(self.original_context.get(_run_id))
                    elif key == "task_id":
                        _task_id.set(self.original_context.get(_task_id))
                    elif key == "pair":
                        _pair.set(self.original_context.get(_pair))
                    elif key == "engine":
                        _engine.set(self.original_context.get(_engine))
                    elif key == "hostname":
                        _hostname.set(self.original_context.get(_hostname))
                    elif key == "gpu_count":
                        _gpu_count.set(self.original_context.get(_gpu_count))
                    elif key == "workers":
                        _workers.set(self.original_context.get(_workers))
                    elif key == "dashboard_url":
                        _dashboard_url.set(self.original_context.get(_dashboard_url))
    
    return ContextManager(**kwargs)

