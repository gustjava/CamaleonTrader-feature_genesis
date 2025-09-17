"""CPU implementations of key feature transforms for backtesting."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

try:  # Optional dependency for Baxter-King filter
    from statsmodels.tsa.filters.bk_filter import bkfilter
except Exception:  # pragma: no cover - statsmodels optional
    bkfilter = None  # type: ignore

try:  # Optional dependency for Empirical Mode Decomposition
    from emd import sift
except Exception:  # pragma: no cover - EMD optional
    sift = None  # type: ignore

try:  # Optional dependency for GARCH fitting
    from arch import arch_model
except Exception:  # pragma: no cover - arch optional
    arch_model = None  # type: ignore


def _numeric_columns(df: pd.DataFrame, candidates: Optional[Sequence[str]] = None) -> Iterable[str]:
    if candidates:
        for col in candidates:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                yield col
    else:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                yield col


def _fracdiff_weights(d: float, size: int, tol: float) -> np.ndarray:
    weights = [1.0]
    for k in range(1, size):
        weight = -weights[-1] * (d - k + 1) / k
        if abs(weight) <= tol:
            break
        weights.append(weight)
    return np.array(weights, dtype=float)


def fractional_difference(series: pd.Series, d: float, tol: float = 1e-5, max_lag: int = 512) -> pd.Series:
    """Compute fractional differencing on CPU."""
    if d is None:
        return pd.Series(np.nan, index=series.index)

    valid = series.astype("float64")
    weights = _fracdiff_weights(float(d), max_lag, float(tol))
    output = np.full(valid.shape[0], np.nan, dtype=float)

    for idx in range(len(valid)):
        start = max(0, idx - len(weights) + 1)
        window = valid.values[start : idx + 1]
        if np.isnan(window).any():
            continue
        w = weights[-len(window) :]
        output[idx] = np.dot(w[::-1], window)

    return pd.Series(output, index=series.index)


# -------- stationarization -----------------------------------------------------

def apply_stationarization_cpu(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]],
    params: Mapping[str, object],
) -> pd.DataFrame:
    frac_cfg = params.get("fracdiff") if isinstance(params, Mapping) else {}
    default_d = float(params.get("d", 0.5)) if isinstance(params, Mapping) else 0.5
    tol = float(params.get("tol", 1e-5)) if isinstance(params, Mapping) else 1e-5
    max_lag = int(params.get("max_lag", 512)) if isinstance(params, Mapping) else 512
    zscore_window = int(params.get("zscore_window", 0)) if isinstance(params, Mapping) else 0

    result = df.copy()
    if not isinstance(frac_cfg, Mapping):
        frac_cfg = {}

    for col in _numeric_columns(df, columns):
        d_val = frac_cfg.get(col, default_d)
        frac_series = fractional_difference(df[col], d_val, tol=tol, max_lag=max_lag)
        result[f"frac_diff_{col}"] = frac_series

        if zscore_window and zscore_window > 1:
            rolling = df[col].rolling(zscore_window)
            zscore = (df[col] - rolling.mean()) / rolling.std(ddof=0)
            result[f"zscore_{col}_{zscore_window}"] = zscore

    return result


# -------- signal processing ----------------------------------------------------

def apply_signal_processing_cpu(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]],
    params: Mapping[str, object],
) -> pd.DataFrame:
    if sift is None:
        return df

    max_imfs = int(params.get("max_imfs", 5)) if isinstance(params, Mapping) else 5
    out = df.copy()

    for col in _numeric_columns(df, columns):
        series = df[col].astype(float).fillna(method="ffill").fillna(method="bfill")
        if series.isna().all():
            continue
        try:
            imfs = sift.sift(series.to_numpy(), max_imfs=max_imfs)
        except Exception:
            continue
        if imfs.ndim != 2:
            continue
        for idx in range(imfs.shape[1]):
            out[f"emd_imf_{idx + 1}"] = imfs[:, idx]
        break  # Mantém compatibilidade: aplica EMD apenas à primeira coluna relevante

    return out


# -------- feature engineering --------------------------------------------------

def apply_feature_engineering_cpu(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]],
    params: Mapping[str, object],
) -> pd.DataFrame:
    k = int(params.get("k", 12)) if isinstance(params, Mapping) else 12
    low = float(params.get("low_freq", 32.0)) if isinstance(params, Mapping) else 32.0
    high = float(params.get("high_freq", 6.0)) if isinstance(params, Mapping) else 6.0

    out = df.copy()

    for col in _numeric_columns(df, columns):
        series = df[col].astype(float)
        if bkfilter is not None:
            try:
                filtered = bkfilter(series.dropna(), low=low, high=high, K=k)
                filtered = pd.Series(filtered, index=series.dropna().index)
                out[f"bk_filter_{col}"] = filtered.reindex(series.index)
                continue
            except Exception:
                pass
        # fallback: band-pass approximation using difference of rolling means
        slow = series.rolling(int(low)).mean()
        fast = series.rolling(int(high)).mean()
        out[f"bk_filter_{col}"] = fast - slow

    return out


# -------- garch features -------------------------------------------------------

def apply_garch_cpu(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]],
    params: Mapping[str, object],
) -> pd.DataFrame:
    window = int(params.get("vol_window", 120)) if isinstance(params, Mapping) else 120
    out = df.copy()

    for col in _numeric_columns(df, columns):
        returns = df[col].astype(float).pct_change().dropna()
        if returns.empty:
            continue

        if arch_model is not None:
            try:
                model = arch_model(returns * 100, vol="Garch", p=1, q=1)
                fitted = model.fit(disp="off")
                cond_vol = fitted.conditional_volatility / 100.0
                out[f"garch_cond_vol_{col}"] = cond_vol.reindex(df.index)
                persistence = float(fitted.params.get("alpha[1]", 0.0) + fitted.params.get("beta[1]", 0.0))
                out[f"garch_persistence_{col}"] = persistence
                out[f"garch_omega_{col}"] = float(fitted.params.get("omega", 0.0))
                continue
            except Exception:
                pass

        cond_vol = returns.rolling(window).std().reindex(df.index)
        out[f"garch_cond_vol_{col}"] = cond_vol
        out[f"garch_persistence_{col}"] = cond_vol.rolling(window).mean()
        out[f"garch_omega_{col}"] = cond_vol.rolling(window).var()

    return out
