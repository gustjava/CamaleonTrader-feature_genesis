"""
Utility functions for statistical tests module.

This module contains helper functions for GPU memory management, adaptive tiling,
and other utility operations used across the statistical tests components.
"""

import logging
import numpy as np
import cupy as cp
import cudf
from typing import Dict, Any
from utils.logging_utils import get_logger

logger = get_logger(__name__, "features.statistical_tests.utils")


def _free_gpu_memory_worker():
    """Free CuPy default memory pool on a Dask worker (best-effort)."""
    try:
        import cupy as _cp  # Import CuPy locally to avoid tokenization issues
        _cp.get_default_memory_pool().free_all_blocks()  # Free all GPU memory blocks
    except Exception as e:
        logger.error(f"Failed to free GPU memory: {e}")
        pass  # Ignore errors if GPU memory cleanup fails


def _adaptive_tile(requested: int) -> int:
    """Adapt tile size to GPU memory to keep per-block usage within a safe fraction.

    Approximation: per block we hold ~4 tile^2 float64 matrices (dx, dy, A, B):
    mem_block ≈ 32 * tile^2 bytes. Target ~8% of total device memory.
    """
    try:
        frac = 0.08
        free_b, total_b = cp.cuda.runtime.memGetInfo()
        import math as _m
        tile_dyn = int(_m.sqrt(max(1.0, (frac * float(total_b)) / 32.0)))
        # Round down to multiple of 256 for better alignment
        tile_dyn = max(256, (tile_dyn // 256) * 256)
        return max(256, min(int(requested), tile_dyn))
    except Exception:
        return int(requested)


def _hermitian_pinv_gpu(R: cp.ndarray, eps: float = 1e-6) -> cp.ndarray:
    """Pseudo-inverse for symmetric/Hermitian positive semi-definite matrices on GPU.

    Uses eigen-decomposition with eigenvalue clipping to eps for stability.
    Falls back to SVD-based pinv if eigh is unavailable.
    """
    try:
        # promote to float64 for numerical stability
        Rw = R.astype(cp.float64, copy=False)
        # ensure symmetry by averaging with transpose
        Rw = 0.5 * (Rw + Rw.T)
        w, V = cp.linalg.eigh(Rw)  # Eigenvalue decomposition
        w = cp.where(w > eps, w, eps)  # Clip small eigenvalues for stability
        inv_w = 1.0 / w  # Compute inverse eigenvalues
        # scale columns of V by inv_w and multiply by V^T
        Rinv = (V * inv_w) @ V.T
        # symmetrize result to ensure numerical stability
        Rinv = 0.5 * (Rinv + Rinv.T)
        return Rinv.astype(R.dtype, copy=False)  # Convert back to original dtype
    except Exception:
        # Fallback to SVD-based pseudoinverse if eigen decomposition fails
        return cp.linalg.pinv(R)


def _tail_k(pdf, k: int):
    """Return tail(k) of a partition (preserve backend: cuDF or pandas).

    Also emits lightweight worker events so we can see GPU activity
    in the scheduler logs even if the dashboard metrics are noisy.
    """
    try:
        from distributed import get_worker as _gw
        _w = _gw()
        _addr = getattr(_w, 'address', 'unknown')
    except Exception:
        _w, _addr = None, 'unknown'
    try:
        import cupy as _cp
        _gpu = int(_cp.cuda.runtime.getDevice())
    except Exception:
        _gpu = -1
    try:
        if _w is not None:
            _w.log_event("stage1", {"event": "tail_start", "worker": _addr, "gpu": _gpu, "in_rows": int(len(pdf)), "k": int(k)})
    except Exception as e:
        logger.error(f"Failed to log tail start event: {e}")
        pass
    out = pdf.tail(int(k))
    try:
        if _w is not None:
            _w.log_event("stage1", {"event": "tail_done", "worker": _addr, "gpu": _gpu, "out_rows": int(len(out))})
    except Exception as e:
        logger.error(f"Failed to log tail done event: {e}")
        pass
    return out


def _tail_k_to_pandas(pdf, k: int):
    """Tail(k) and convert to pandas on worker to avoid GPU-object gather.

    Useful when running with TCP protocol or when the client cannot safely
    receive GPU-backed cuDF objects (reduces device-serialization overhead).
    """
    try:
        out = pdf.tail(int(k))
    except Exception:
        try:
            n = int(k)
        except Exception:
            n = 100000
        out = pdf.iloc[-n:]
    return out.to_pandas() if hasattr(out, 'to_pandas') else out


# ADF functions removed from project


# JB removed from project: helpers deleted


def _compute_forward_log_return_partition(pdf: cudf.DataFrame, price_col: str, horizon: int, out_col: str) -> cudf.DataFrame:
    """Compute forward log returns for a partition."""
    try:
        if price_col not in pdf.columns:
            return cudf.DataFrame({out_col: cudf.Series([], dtype='f4')})
        
        prices = pdf[price_col].values
        n = len(prices)
        log_returns = cp.full(n, cp.nan, dtype=cp.float32)
        
        for i in range(n - horizon):
            if cp.isfinite(prices[i]) and cp.isfinite(prices[i + horizon]):
                log_returns[i] = cp.log(prices[i + horizon] / prices[i])
        
        return cudf.DataFrame({out_col: cudf.Series(log_returns, index=pdf.index)})
    except Exception:
        return cudf.DataFrame({out_col: cudf.Series([], dtype='f4')})
