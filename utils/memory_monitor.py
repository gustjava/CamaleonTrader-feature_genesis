"""
Memory Monitoring Utility for Dynamic Stage 0 Pipeline

Provides real-time monitoring of GPU memory usage and automatic
memory management to prevent OOM errors.
"""

import logging
import time
import threading
from typing import Optional, Dict, Any, Callable, List
import cupy as cp
import psutil

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitors GPU and system memory usage in real-time."""
    
    def __init__(self, alert_threshold: float = 0.9, check_interval: int = 10):
        """
        Initialize the memory monitor.
        
        Args:
            alert_threshold: Memory usage threshold for alerts (0.0-1.0)
            check_interval: Check interval in seconds
        """
        self.alert_threshold = alert_threshold
        self.check_interval = check_interval
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.alert_callbacks: List[Callable] = []
        
    def start_monitoring(self):
        """Start the memory monitoring thread."""
        if self.is_monitoring:
            logger.warning("Memory monitoring is already running")
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop the memory monitoring thread."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Memory monitoring stopped")
    
    def add_alert_callback(self, callback: Callable):
        """Add a callback function to be called when memory alert is triggered."""
        self.alert_callbacks.append(callback)
    
    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """Get current GPU memory information."""
        try:
            # Get GPU memory info
            free_memory, total_memory = cp.cuda.runtime.memGetInfo()
            used_memory = total_memory - free_memory
            usage_percent = used_memory / total_memory
            
            return {
                'free_memory_gb': free_memory / (1024**3),
                'total_memory_gb': total_memory / (1024**3),
                'used_memory_gb': used_memory / (1024**3),
                'usage_percent': usage_percent,
                'device_count': cp.cuda.runtime.getDeviceCount()
            }
        except Exception as e:
            logger.error(f"Error getting GPU memory info: {e}")
            return {}
    
    def get_system_memory_info(self) -> Dict[str, Any]:
        """Get current system memory information."""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'usage_percent': memory.percent / 100.0
            }
        except Exception as e:
            logger.error(f"Error getting system memory info: {e}")
            return {}
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Get memory information
                gpu_info = self.get_gpu_memory_info()
                system_info = self.get_system_memory_info()
                
                # Check for high memory usage
                if gpu_info and gpu_info.get('usage_percent', 0) > self.alert_threshold:
                    logger.warning(f"High GPU memory usage: {gpu_info['usage_percent']:.2%}")
                    self._trigger_alerts(gpu_info, system_info)
                
                if system_info and system_info.get('usage_percent', 0) > self.alert_threshold:
                    logger.warning(f"High system memory usage: {system_info['usage_percent']:.2%}")
                    self._trigger_alerts(gpu_info, system_info)
                
                # Log memory status periodically
                if gpu_info:
                    logger.debug(f"GPU Memory: {gpu_info['used_memory_gb']:.2f}GB / {gpu_info['total_memory_gb']:.2f}GB ({gpu_info['usage_percent']:.2%})")
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def _trigger_alerts(self, gpu_info: Dict[str, Any], system_info: Dict[str, Any]):
        """Trigger memory alert callbacks."""
        alert_info = {
            'gpu_info': gpu_info,
            'system_info': system_info,
            'timestamp': time.time()
        }
        
        for callback in self.alert_callbacks:
            try:
                callback(alert_info)
            except Exception as e:
                logger.error(f"Error in memory alert callback: {e}")
    
    def force_memory_cleanup(self):
        """Force cleanup of GPU memory."""
        try:
            # Free all CuPy memory blocks
            cp.get_default_memory_pool().free_all_blocks()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info("Forced memory cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of current memory usage."""
        gpu_info = self.get_gpu_memory_info()
        system_info = self.get_system_memory_info()
        
        return {
            'gpu': gpu_info,
            'system': system_info,
            'timestamp': time.time(),
            'is_monitoring': self.is_monitoring
        }


class MemoryOptimizer:
    """Provides memory optimization utilities."""
    
    @staticmethod
    def optimize_chunk_size(available_memory_gb: float, data_size_gb: float, 
                          min_chunk_size: int = 1000, max_chunk_size: int = 50000) -> int:
        """
        Calculate optimal chunk size based on available memory.
        
        Args:
            available_memory_gb: Available GPU memory in GB
            data_size_gb: Size of data to process in GB
            min_chunk_size: Minimum chunk size
            max_chunk_size: Maximum chunk size
            
        Returns:
            Optimal chunk size
        """
        # Reserve 20% of memory for overhead
        usable_memory_gb = available_memory_gb * 0.8
        
        # Calculate chunk size based on memory
        if data_size_gb > 0:
            optimal_chunk_size = int(usable_memory_gb / data_size_gb * 10000)
        else:
            optimal_chunk_size = 10000
        
        # Clamp to reasonable bounds
        optimal_chunk_size = max(min_chunk_size, min(optimal_chunk_size, max_chunk_size))
        
        return optimal_chunk_size
    
    @staticmethod
    def estimate_memory_usage(data_shape: tuple, dtype_size: int = 8) -> float:
        """
        Estimate memory usage for data processing.
        
        Args:
            data_shape: Shape of the data
            dtype_size: Size of data type in bytes
            
        Returns:
            Estimated memory usage in GB
        """
        total_elements = 1
        for dim in data_shape:
            total_elements *= dim
        
        memory_bytes = total_elements * dtype_size
        memory_gb = memory_bytes / (1024**3)
        
        return memory_gb
    
    @staticmethod
    def should_use_chunking(data_size_gb: float, available_memory_gb: float, 
                          threshold: float = 0.5) -> bool:
        """
        Determine if chunked processing should be used.
        
        Args:
            data_size_gb: Size of data in GB
            available_memory_gb: Available GPU memory in GB
            threshold: Memory usage threshold for chunking
            
        Returns:
            True if chunking should be used
        """
        return data_size_gb > (available_memory_gb * threshold)
