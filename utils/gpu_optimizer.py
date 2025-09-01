"""
GPU Optimization Utilities for Dynamic Stage 0 Pipeline

This module provides utilities for:
- Maximizing GPU performance
- Efficient memory management
- Avoiding unnecessary CPU-GPU transfers
- GPU kernel optimization
- Memory pool management
"""

import logging
import time
import gc
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager
import functools

import cupy as cp
import cudf
import numpy as np
from numba import cuda, jit
from numba.cuda import jit as cuda_jit

logger = logging.getLogger(__name__)


class GPUOptimizer:
    """
    GPU optimization utilities for maximizing performance and memory efficiency.
    
    This class provides:
    - Memory pool management
    - Kernel optimization
    - Transfer optimization
    - Performance monitoring
    - Memory cleanup utilities
    """
    
    def __init__(self):
        """Initialize the GPU optimizer."""
        self.memory_pool = cp.get_default_memory_pool()
        self.pinned_memory_pool = cp.get_default_pinned_memory_pool()
        self.performance_stats = {}
        
        # Configure memory pools for optimal performance
        self._configure_memory_pools()
    
    def _configure_memory_pools(self):
        """Configure memory pools for optimal performance."""
        try:
            # Set memory pool size (8GB)
            pool_size = 8 * 1024**3
            self.memory_pool.set_limit(size=pool_size)
            
            # Enable memory spilling
            self.memory_pool.set_limit(size=pool_size, fraction=0.9)
            
            logger.info(f"GPU memory pool configured with {pool_size / (1024**3):.1f} GB limit")
            
        except Exception as e:
            logger.warning(f"Could not configure memory pools: {e}")
    
    @contextmanager
    def memory_monitoring(self, operation_name: str):
        """
        Context manager for monitoring GPU memory usage during operations.
        
        Args:
            operation_name: Name of the operation for logging
        """
        start_memory = self._get_gpu_memory_usage()
        start_time = time.time()
        
        try:
            yield
        finally:
            end_memory = self._get_gpu_memory_usage()
            end_time = time.time()
            
            memory_diff = end_memory['used'] - start_memory['used']
            time_diff = end_time - start_time
            
            self.performance_stats[operation_name] = {
                'memory_used_mb': memory_diff / (1024**2),
                'duration_s': time_diff,
                'memory_peak_mb': end_memory['used'] / (1024**2)
            }
            
            logger.debug(f"{operation_name}: {memory_diff / (1024**2):.1f} MB, {time_diff:.2f}s")
    
    def _get_gpu_memory_usage(self) -> Dict[str, int]:
        """Get current GPU memory usage."""
        try:
            free, total = cp.cuda.runtime.memGetInfo()
            used = total - free
            return {
                'free': free,
                'total': total,
                'used': used,
                'percent': (used / total) * 100
            }
        except Exception as e:
            logger.warning(f"Could not get GPU memory info: {e}")
            return {'free': 0, 'total': 0, 'used': 0, 'percent': 0}
    
    def optimize_cudf_operations(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """
        Optimize cuDF DataFrame for better GPU performance.
        
        Args:
            df: Input cuDF DataFrame
            
        Returns:
            Optimized cuDF DataFrame
        """
        with self.memory_monitoring("cudf_optimization"):
            # Ensure data types are optimal for GPU
            optimized_df = self._optimize_dtypes(df)
            
            # Sort columns for better memory access patterns
            optimized_df = self._optimize_column_order(optimized_df)
            
            return optimized_df
    
    def _optimize_dtypes(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """Optimize data types for GPU performance."""
        optimized_df = df.copy()
        
        for col in df.columns:
            dtype = df[col].dtype
            
            # Use float32 instead of float64 for better GPU performance
            if dtype == 'float64':
                optimized_df[col] = optimized_df[col].astype('float32')
            
            # Use int32 instead of int64 where possible
            elif dtype == 'int64':
                col_min = df[col].min()
                col_max = df[col].max()
                if col_min >= -2**31 and col_max <= 2**31 - 1:
                    optimized_df[col] = optimized_df[col].astype('int32')
        
        return optimized_df
    
    def _optimize_column_order(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """Optimize column order for better memory access patterns."""
        # Reorder columns to group similar data types together
        float_cols = [col for col in df.columns if df[col].dtype in ['float32', 'float64']]
        int_cols = [col for col in df.columns if df[col].dtype in ['int32', 'int64']]
        other_cols = [col for col in df.columns if col not in float_cols + int_cols]
        
        reordered_cols = float_cols + int_cols + other_cols
        return df[reordered_cols]
    
    def avoid_cpu_transfer(self, func: Callable) -> Callable:
        """
        Decorator to ensure functions stay on GPU and avoid CPU transfers.
        
        Args:
            func: Function to optimize
            
        Returns:
            Optimized function
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if any arguments are on CPU and transfer to GPU if needed
            gpu_args = []
            for arg in args:
                if isinstance(arg, np.ndarray):
                    gpu_args.append(cp.asarray(arg))
                elif isinstance(arg, cudf.DataFrame):
                    gpu_args.append(cudf.DataFrame.from_pandas(arg))
                else:
                    gpu_args.append(arg)
            
            # Execute function on GPU
            result = func(*gpu_args, **kwargs)
            
            # Ensure result stays on GPU
            if isinstance(result, np.ndarray):
                return cp.asarray(result)
            elif isinstance(result, cudf.DataFrame):
                return cudf.DataFrame.from_pandas(result)
            
            return result
        
        return wrapper
    
    def create_gpu_kernel(self, kernel_func: Callable, signature: str) -> Callable:
        """
        Create an optimized GPU kernel using Numba CUDA.
        
        Args:
            kernel_func: Function to compile as GPU kernel
            signature: CUDA kernel signature (e.g., 'void(float32[:], float32[:], int32)')
            
        Returns:
            Compiled GPU kernel
        """
        try:
            # Compile the kernel
            compiled_kernel = cuda_jit(signature)(kernel_func)
            
            logger.info(f"GPU kernel compiled successfully: {kernel_func.__name__}")
            return compiled_kernel
            
        except Exception as e:
            logger.error(f"Failed to compile GPU kernel {kernel_func.__name__}: {e}")
            raise
    
    def optimize_rolling_operations(self, df: cudf.DataFrame, window: int, operation: str) -> cudf.Series:
        """
        Optimize rolling window operations for GPU.
        
        Args:
            df: Input cuDF DataFrame
            window: Rolling window size
            operation: Operation to perform ('mean', 'std', 'sum', etc.)
            
        Returns:
            Result as cuDF Series
        """
        with self.memory_monitoring(f"rolling_{operation}"):
            # Use cuDF's optimized rolling operations
            if operation == 'mean':
                return df.rolling(window=window).mean()
            elif operation == 'std':
                return df.rolling(window=window).std()
            elif operation == 'sum':
                return df.rolling(window=window).sum()
            elif operation == 'min':
                return df.rolling(window=window).min()
            elif operation == 'max':
                return df.rolling(window=window).max()
            else:
                # Fallback to custom implementation
                return self._custom_rolling_operation(df, window, operation)
    
    def _custom_rolling_operation(self, df: cudf.DataFrame, window: int, operation: str) -> cudf.Series:
        """Custom rolling operation implementation for GPU."""
        # This would implement custom rolling operations optimized for GPU
        # For now, return a simple implementation
        return df.rolling(window=window).apply(lambda x: getattr(x, operation)())
    
    def batch_process(self, data: List[cudf.DataFrame], batch_size: int, 
                     process_func: Callable) -> List[Any]:
        """
        Process data in batches to optimize GPU memory usage.
        
        Args:
            data: List of cuDF DataFrames to process
            batch_size: Number of DataFrames to process at once
            process_func: Function to apply to each batch
            
        Returns:
            List of processed results
        """
        results = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            
            with self.memory_monitoring(f"batch_{i//batch_size}"):
                # Process batch
                batch_result = process_func(batch)
                results.append(batch_result)
                
                # Clean up memory after each batch
                self.cleanup_memory()
        
        return results
    
    def cleanup_memory(self):
        """Clean up GPU memory and force garbage collection."""
        try:
            # Clear memory pools
            self.memory_pool.free_all_blocks()
            self.pinned_memory_pool.free_all_blocks()
            
            # Force garbage collection
            gc.collect()
            
            logger.debug("GPU memory cleanup completed")
            
        except Exception as e:
            logger.warning(f"Error during GPU memory cleanup: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get GPU performance statistics."""
        memory_usage = self._get_gpu_memory_usage()
        
        return {
            'memory_usage': memory_usage,
            'performance_stats': self.performance_stats,
            'memory_pool_stats': {
                'pool_size': self.memory_pool.get_limit(),
                'used_blocks': self.memory_pool.used_bytes(),
                'free_blocks': self.memory_pool.free_bytes()
            }
        }
    
    def optimize_convolve(self, data: cp.ndarray, kernel: cp.ndarray, mode: str = 'same') -> cp.ndarray:
        """
        Optimized convolution operation for GPU.
        
        Args:
            data: Input data array
            kernel: Convolution kernel
            mode: Convolution mode ('same', 'valid', 'full')
            
        Returns:
            Convolved result
        """
        with self.memory_monitoring("convolution"):
            # Use CuPy's optimized convolution
            if kernel.size <= 129:
                # Use direct convolution for small kernels
                return cp.convolve(data, kernel, mode=mode)
            else:
                # Use FFT convolution for large kernels
                try:
                    from cusignal import fftconvolve
                    return fftconvolve(data, kernel, mode=mode)
                except ImportError:
                    # Fallback to CuPy FFT
                    return cp.convolve(data, kernel, mode=mode)
    
    def optimize_correlation(self, x: cp.ndarray, y: cp.ndarray) -> float:
        """
        Optimized correlation calculation for GPU.
        
        Args:
            x: First array
            y: Second array
            
        Returns:
            Correlation coefficient
        """
        with self.memory_monitoring("correlation"):
            # Ensure arrays are on GPU
            x_gpu = cp.asarray(x)
            y_gpu = cp.asarray(y)
            
            # Calculate correlation using GPU-optimized operations
            x_mean = cp.mean(x_gpu)
            y_mean = cp.mean(y_gpu)
            
            numerator = cp.sum((x_gpu - x_mean) * (y_gpu - y_mean))
            denominator = cp.sqrt(cp.sum((x_gpu - x_mean)**2) * cp.sum((y_gpu - y_mean)**2))
            
            if denominator == 0:
                return 0.0
            
            return float(numerator / denominator)
    
    def optimize_distance_correlation(self, x: cp.ndarray, y: cp.ndarray, 
                                    max_samples: int = 10000) -> float:
        """
        Optimized distance correlation calculation for GPU.
        
        Args:
            x: First array
            y: Second array
            max_samples: Maximum number of samples to use
            
        Returns:
            Distance correlation coefficient
        """
        with self.memory_monitoring("distance_correlation"):
            # Sample data if too large
            if len(x) > max_samples:
                indices = cp.random.choice(len(x), max_samples, replace=False)
                x = x[indices]
                y = y[indices]
            
            # Calculate distance matrices on GPU
            x_dist = self._calculate_distance_matrix(x)
            y_dist = self._calculate_distance_matrix(y)
            
            # Calculate distance correlation
            x_mean = cp.mean(x_dist)
            y_mean = cp.mean(y_dist)
            
            numerator = cp.sum((x_dist - x_mean) * (y_dist - y_mean))
            denominator = cp.sqrt(cp.sum((x_dist - x_mean)**2) * cp.sum((y_dist - y_mean)**2))
            
            if denominator == 0:
                return 0.0
            
            return float(numerator / denominator)
    
    def _calculate_distance_matrix(self, data: cp.ndarray) -> cp.ndarray:
        """Calculate distance matrix efficiently on GPU."""
        # Use broadcasting for efficient distance calculation
        diff = data[:, cp.newaxis] - data[cp.newaxis, :]
        return cp.abs(diff)
    
    def optimize_garch_likelihood(self, returns: cp.ndarray, params: cp.ndarray) -> float:
        """
        Optimized GARCH log-likelihood calculation for GPU.
        
        Args:
            returns: Return series
            params: GARCH parameters [omega, alpha, beta]
            
        Returns:
            Log-likelihood value
        """
        with self.memory_monitoring("garch_likelihood"):
            omega, alpha, beta = params
            
            # Initialize variance
            n = len(returns)
            variance = cp.zeros(n, dtype=cp.float32)
            variance[0] = cp.var(returns)
            
            # Calculate conditional variance
            for i in range(1, n):
                variance[i] = omega + alpha * returns[i-1]**2 + beta * variance[i-1]
            
            # Calculate log-likelihood
            log_likelihood = -0.5 * cp.sum(cp.log(2 * cp.pi * variance) + returns**2 / variance)
            
            return float(log_likelihood)


# Global GPU optimizer instance
gpu_optimizer = GPUOptimizer()


# Utility functions for easy access
def optimize_cudf(df: cudf.DataFrame) -> cudf.DataFrame:
    """Optimize cuDF DataFrame for GPU performance."""
    return gpu_optimizer.optimize_cudf_operations(df)


def avoid_cpu_transfer(func: Callable) -> Callable:
    """Decorator to avoid CPU-GPU transfers."""
    return gpu_optimizer.avoid_cpu_transfer(func)


def gpu_kernel(signature: str):
    """Decorator to create GPU kernels."""
    def decorator(func: Callable) -> Callable:
        return gpu_optimizer.create_gpu_kernel(func, signature)
    return decorator


def cleanup_gpu_memory():
    """Clean up GPU memory."""
    gpu_optimizer.cleanup_memory()


def get_gpu_stats() -> Dict[str, Any]:
    """Get GPU performance statistics."""
    return gpu_optimizer.get_performance_stats()
