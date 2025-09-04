"""
Feature Engineering Engine (Stage 0)

Applies early feature transformations before selection stages.
Initial scope: Generalized Baxter–King (BK) filter over configured source columns.
"""

import logging
from typing import List, Dict, Any, Optional

import dask_cudf
import cudf

from .base_engine import BaseFeatureEngine

# Local BK helpers (decoupled from legacy SignalProcessor)
import numpy as _np
import cupy as _cp
from functools import lru_cache as _lru_cache

try:
    from cusignal import fftconvolve as _fftconv
except Exception:
    try:
        from cusignal.filtering import fftconvolve as _fftconv
    except Exception:
        from scipy.signal import fftconvolve as _scipy_fft
        def _fftconv(x, w, mode="same"):
            return _cp.asarray(_scipy_fft(_cp.asnumpy(x), _cp.asnumpy(w), mode=mode))


@_lru_cache(maxsize=16)
def _bk_weights_cpu(k: int, low_period: float, high_period: float) -> _np.ndarray:
    """CPU Baxter–King weights (float32), cached per worker.

    low_period > high_period (e.g., 32 > 6) ⇒ pass-band between [w_high, w_low].
    """
    w_low = 2 * _np.pi / float(low_period)
    w_high = 2 * _np.pi / float(high_period)
    weights = _np.zeros(2 * k + 1, dtype=_np.float32)
    weights[k] = (w_high - w_low) / _np.pi
    j = _np.arange(1, k + 1, dtype=_np.float32)
    weights[k + 1:] = (_np.sin(w_high * j) - _np.sin(w_low * j)) / (_np.pi * j)
    weights[:k] = weights[k + 1:][::-1]
    wsum = weights.sum(dtype=_np.float64)
    weights[k] -= (wsum - weights[k]).astype(_np.float32)
    return weights


def _apply_bk_filter_gpu_partition(series: cudf.Series, k: int, low_period: float, high_period: float, causal: bool = True) -> cudf.Series:
    """Deterministic partition function: apply BK on a single partition (GPU).

    Handles nulls by filling with 0 for the convolution and restoring NaNs
    at original null positions in the output.
    """
    # Fast-path: if dtype is obviously non-numeric, return all-NaN
    try:
        dt = str(series.dtype).lower()
        if not (dt.startswith('float') or dt.startswith('int') or dt.startswith('uint')):
            return cudf.Series(_cp.full(len(series), _cp.nan, dtype=_cp.float32), index=series.index)
    except Exception:
        pass
    # Ensure float32 and capture null mask before conversion (coerce failure → all-NaN)
    try:
        s = series.astype('f4')
    except Exception:
        return cudf.Series(_cp.full(len(series), _cp.nan, dtype=_cp.float32), index=series.index)
    null_mask = s.isna()
    # Fill nulls to allow cudf->cupy conversion
    s_filled = s.fillna(0.0)
    x = s_filled.to_cupy()
    w = _cp.asarray(_bk_weights_cpu(int(k), float(low_period), float(high_period)))
    n_kernel = 2 * int(k) + 1
    if n_kernel <= 129:
        y = _cp.convolve(x, w, mode="same")
    else:
        y = _fftconv(x, w, mode="same")
    k = int(k)
    # Restore NaNs where input had nulls
    try:
        m = null_mask.to_cupy()
        if m.any():
            y[m] = _cp.nan
    except Exception:
        # Best-effort; if mask conversion fails, proceed without restoration
        pass
    # BK borders are NaN by definition
    y[:k] = _cp.nan
    y[-k:] = _cp.nan

    # Build output series and apply causal shift if requested
    out = cudf.Series(y, index=series.index)
    if causal and k > 0:
        # Shift by +k to ensure each value at t depends only on x[<= t]
        out = out.shift(k)
    return out

logger = logging.getLogger(__name__)


class FeatureEngineeringEngine(BaseFeatureEngine):
    """
    Stage 0 feature engineering engine.

    Currently supports:
    - Baxter–King band‑pass filter over multiple source columns.
    """

    def _bk_params(self) -> Dict[str, Any]:
        """Resolve BK parameters from config with backward compatibility.

        Priority:
        1) features.feature_engineering.baxter_king.{k,low_freq,high_freq,source_columns}
        2) features.baxter_king.{k,low_freq,high_freq}
        3) legacy flat: features.baxter_king_k, _low_freq, _high_freq
        """
        feats = getattr(self.settings, 'features', None)
        k = 12
        low = 32.0
        high = 6.0
        sources: List[str] = []
        apply_all = False
        causal = True
        if feats is not None:
            try:
                fe = getattr(feats, 'feature_engineering', {}) or {}
                bk = fe.get('baxter_king', {}) if isinstance(fe, dict) else {}
                if isinstance(bk, dict):
                    k = int(bk.get('k', k))
                    low = float(bk.get('low_freq', low))
                    high = float(bk.get('high_freq', high))
                    sc = bk.get('source_columns', [])
                    if isinstance(sc, list):
                        sources = [str(c) for c in sc]
                    apply_all = bool(bk.get('apply_to_all', False))
                    causal = bool(bk.get('causal', causal))
            except Exception:
                pass
            # fallback to features.baxter_king dict
            try:
                if not sources:
                    bk2 = getattr(feats, 'baxter_king', {}) or {}
                    if isinstance(bk2, dict):
                        k = int(bk2.get('k', k))
                        low = float(bk2.get('low_freq', low))
                        high = float(bk2.get('high_freq', high))
                        causal = bool(bk2.get('causal', causal))
            except Exception:
                pass
            # legacy flat keys
            try:
                k = int(getattr(feats, 'baxter_king_k', k))
                low = float(getattr(feats, 'baxter_king_low_freq', low))
                high = float(getattr(feats, 'baxter_king_high_freq', high))
            except Exception:
                pass
        return {'k': k, 'low': low, 'high': high, 'source_columns': sources, 'apply_to_all': apply_all, 'causal': causal}

    def _eligible_all_numeric(self, df, exclude: List[str]) -> List[str]:
        """Return all numeric columns suitable for BK, excluding prefixes/names.

        - Uses DataFrame dtypes (cuDF or Dask-cuDF meta) to pick floats only.
        - Excludes targets/deny/metrics prefixes and already-derived bk_filter_*.
        """
        try:
            # Use meta dtypes for Dask-cuDF to avoid compute
            dtypes = getattr(getattr(df, '_meta', None), 'dtypes', None) or getattr(df, 'dtypes', None)
        except Exception:
            dtypes = None
        cols = list(map(str, getattr(df, 'columns', [])))
        numeric = []
        if dtypes is not None:
            try:
                for c in cols:
                    dt = dtypes.get(c, None) if hasattr(dtypes, 'get') else None
                    sdt = str(dt).lower() if dt is not None else ''
                    if sdt.startswith('float') or sdt in ('f4', 'f8'):
                        numeric.append(c)
            except Exception:
                # Fallback: include all, will cast later
                numeric = cols[:]
        else:
            numeric = cols[:]

        # Build exclusion predicates from settings
        feats = getattr(self.settings, 'features', None)
        deny_exact = set((getattr(feats, 'feature_denylist', []) or [])) if feats else set()
        deny_prefixes = list((getattr(feats, 'feature_deny_prefixes', []) or [])) if feats else []
        dataset_target_columns = list((getattr(feats, 'dataset_target_columns', []) or [])) if feats else []
        dataset_target_prefixes = list((getattr(feats, 'dataset_target_prefixes', []) or [])) if feats else []
        metrics_prefixes = list((getattr(feats, 'metrics_prefixes', []) or [])) if feats else []
        sel_target = str(getattr(feats, 'selection_target_column', '')) if feats else ''
        sel_targets = list((getattr(feats, 'selection_target_columns', []) or [])) if feats else []

        def _excluded(name: str) -> bool:
            if name in exclude:
                return True
            if name.startswith('bk_filter_'):
                return True
            if name == sel_target or name in sel_targets:
                return True
            if name in deny_exact or name in dataset_target_columns:
                return True
            if any(name.startswith(p) for p in (deny_prefixes + dataset_target_prefixes + metrics_prefixes)):
                return True
            return False

        return [c for c in numeric if not _excluded(c)]

    def _bk_sources_present(self, df_cols: List[str], configured: List[str]) -> List[str]:
        """Choose source columns: prefer configured; else heuristics.

        Heuristics when not configured:
        - Include 'y_close' if present
        - Include 'log_stabilized_y_close' if present
        - Include first column containing 'close' (to avoid explosion)
        """
        cols = list(map(str, df_cols))
        if configured:
            return [c for c in configured if c in cols]
        picks: List[str] = []
        if 'y_close' in cols:
            picks.append('y_close')
        if 'log_stabilized_y_close' in cols:
            picks.append('log_stabilized_y_close')
        # one generic close if none selected yet
        if not picks:
            for c in cols:
                if 'close' in c.lower():
                    picks.append(c)
                    break
        return picks

    # -------------------- cuDF path --------------------
    def process_cudf(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        self._log_info("Starting FeatureEngineering (cuDF)…")
        params = self._bk_params()
        apply_all = bool(params.get('apply_to_all', False))
        if apply_all:
            src = self._eligible_all_numeric(gdf, exclude=[])
        else:
            src = self._bk_sources_present(list(gdf.columns), params.get('source_columns', []))
        if not src:
            self._log_warn("FeatureEngineering: no BK source columns detected; skipping BK")
            return gdf
        k = int(params['k'])
        low = float(params['low'])
        high = float(params['high'])
        causal = bool(params.get('causal', True))
        new_cols: List[str] = []
        for col in src:
            out = f"bk_filter_{col}"
            try:
                gdf[out] = _apply_bk_filter_gpu_partition(gdf[col], k, low, high, causal)
                new_cols.append(out)
                self._log_info("Applied BK", source=col, out=out, k=k, low=low, high=high, causal=bool(causal))
            except Exception as e:
                self._log_warn("BK application failed", source=col, error=str(e))

        # Record metrics/artifact
        try:
            metrics = {
                'new_columns': new_cols,
                'new_columns_count': len(new_cols),
                'bk_k': k,
                'bk_low_period': low,
                'bk_high_period': high,
            }
            self._record_metrics('feature_engineering', metrics)
            if bool(getattr(self.settings.features, 'debug_write_artifacts', True)):
                from pathlib import Path
                import json as _json
                out_root = Path(getattr(self.settings.output, 'output_path', './output'))
                subdir = str(getattr(self.settings.features, 'artifacts_dir', 'artifacts'))
                out_dir = out_root / subdir / 'signal'
                out_dir.mkdir(parents=True, exist_ok=True)
                summary_path = out_dir / 'summary_stage0_bk.json'
                with open(summary_path, 'w') as f:
                    _json.dump(metrics, f, indent=2)
                self._record_artifact('feature_engineering', str(summary_path), kind='json')
        except Exception:
            pass

        self._log_info("FeatureEngineering complete (cuDF).", new_cols=len(new_cols))
        return gdf

    # -------------------- Dask-cuDF path --------------------
    def process(self, df: dask_cudf.DataFrame) -> dask_cudf.DataFrame:
        self._log_info("Starting FeatureEngineering (Dask)…")
        params = self._bk_params()
        try:
            cols = list(df.columns)
        except Exception:
            cols = []
        apply_all = bool(params.get('apply_to_all', False))
        if apply_all:
            src = self._eligible_all_numeric(df, exclude=[])
        else:
            src = self._bk_sources_present(cols, params.get('source_columns', []))
        if not src:
            self._log_warn("FeatureEngineering: no BK source columns detected (Dask); skipping BK")
            return df
        k = int(params['k'])
        low = float(params['low'])
        high = float(params['high'])
        causal = bool(params.get('causal', True))
        new_cols: List[str] = []
        for col in src:
            out = f"bk_filter_{col}"
            try:
                df[out] = df[col].map_partitions(
                    _apply_bk_filter_gpu_partition, k, low, high, bool(causal), meta=(out, 'f4')
                )
                new_cols.append(out)
                self._log_info("Applied BK (Dask)", source=col, out=out, k=k, low=low, high=high, causal=bool(causal))
            except Exception as e:
                self._log_warn("BK application failed (Dask)", source=col, error=str(e))

        # Record metrics/artifact (Dask)
        try:
            metrics = {
                'new_columns': new_cols,
                'new_columns_count': len(new_cols),
                'bk_k': k,
                'bk_low_period': low,
                'bk_high_period': high,
            }
            self._record_metrics('feature_engineering', metrics)
            if bool(getattr(self.settings.features, 'debug_write_artifacts', True)):
                from pathlib import Path
                import json as _json
                out_root = Path(getattr(self.settings.output, 'output_path', './output'))
                subdir = str(getattr(self.settings.features, 'artifacts_dir', 'artifacts'))
                out_dir = out_root / subdir / 'signal'
                out_dir.mkdir(parents=True, exist_ok=True)
                summary_path = out_dir / 'summary_stage0_bk.json'
                with open(summary_path, 'w') as f:
                    _json.dump(metrics, f, indent=2)
                self._record_artifact('feature_engineering', str(summary_path), kind='json')
        except Exception:
            pass

        self._log_info("FeatureEngineering complete (Dask).", new_cols=len(new_cols))
        return df
