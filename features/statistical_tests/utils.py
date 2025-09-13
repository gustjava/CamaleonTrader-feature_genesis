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

logger = logging.getLogger(__name__)


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


def _adf_tstat_window_host(vals: np.ndarray) -> float:
    """Compute ADF t-statistic for a time series window on CPU."""
    # Drop NaN/inf inside the window to avoid poisoning the statistic
    try:
        x = np.asarray(vals, dtype=np.float64)
    except Exception:
        return np.nan
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:  # Need at least 3 points for ADF test
        return np.nan
    prev = float(x[0])  # Previous value for differencing
    sum_z = sum_y = sum_zz = sum_yy = sum_zy = 0.0  # Initialize sums for regression
    m = n - 1  # Number of differences
    for i in range(1, n):
        z = prev  # Lagged value (x_t-1)
        y = float(x[i]) - prev  # First difference (Δx_t)
        prev = float(x[i])  # Update previous value
        sum_z += z  # Sum of lagged values
        sum_y += y  # Sum of differences
        sum_zz += z * z  # Sum of squared lagged values
        sum_yy += y * y  # Sum of squared differences
        sum_zy += z * y  # Sum of cross products
    mz = sum_z / m  # Mean of lagged values
    my = sum_y / m  # Mean of differences
    Sxx = sum_zz - m * mz * mz  # Sum of squares for x (lagged values)
    Sxy = sum_zy - m * mz * my  # Sum of cross products
    if Sxx <= 0.0 or m <= 2:  # Check for valid regression
        return np.nan
    beta = Sxy / Sxx  # Regression coefficient (AR(1) coefficient)
    alpha = my - beta * mz  # Intercept
    # Sum of squared errors calculation
    SSE = (sum_yy + m * alpha * alpha + beta * beta * sum_zz - 2.0 * alpha * sum_y - 2.0 * beta * sum_zy + 2.0 * alpha * beta * sum_z)
    dof = m - 2  # Degrees of freedom
    if dof <= 0:
        return np.nan
    sigma2 = SSE / dof  # Error variance
    if sigma2 <= 0.0:
        return np.nan
    se_beta = (sigma2 / Sxx) ** 0.5  # Standard error of beta
    if se_beta == 0.0:
        return np.nan
    return beta / se_beta  # t-statistic for unit root test


def _adf_rolling_partition(series: cudf.Series, window: int, min_periods: int) -> cudf.Series:
    """Apply ADF test to rolling windows of a time series."""
    try:
        # Host rolling apply with host function
        return series.rolling(window=window, min_periods=min_periods).apply(lambda x: _adf_tstat_window_host(np.asarray(x)))
    except Exception:
        return cudf.Series(cp.full(len(series), cp.nan))  # Return NaN series if computation fails


def _jb_pvalue_window_host(vals: np.ndarray) -> float:
    """Jarque–Bera p-value for a window (CPU, causal).

    JB = n/6 * (skew^2 + (kurt-3)^2 / 4)
    For chi2 with df=2, p = exp(-JB/2)
    """
    try:
        x = np.asarray(vals, dtype=np.float64)
        x = x[np.isfinite(x)]
        n = len(x)
        if n < 8:
            return float('nan')
        m = np.mean(x)
        s = np.std(x)
        if s == 0.0:
            return float('nan')
        z = (x - m) / s
        skew = np.mean(z**3)
        kurt = np.mean(z**4)
        jb = (n / 6.0) * (skew**2 + ((kurt - 3.0)**2) / 4.0)
        # For df=2, chi2 survival function simplifies to exp(-x/2)
        p = float(np.exp(-0.5 * jb))
        return p
    except Exception:
        return float('nan')

def _jb_rolling_partition(series: cudf.Series, window: int, min_periods: int) -> cudf.Series:
    """Apply Jarque–Bera p-value to rolling windows of a time series (CPU func on cuDF).

    Uses the existing host implementation `_jb_pvalue_window_host` applied via cuDF rolling.
    """
    try:
        return series.rolling(window=window, min_periods=min_periods).apply(
            lambda x: _jb_pvalue_window_host(np.asarray(x))
        )
    except Exception:
        import cupy as _cp
        return cudf.Series(_cp.full(len(series), _cp.nan))


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
