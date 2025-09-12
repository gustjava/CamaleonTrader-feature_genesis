"""
Engine metrics helpers.

Provides convenience functions to summarize input characteristics, transformation
impacts, performance and quality metrics in a GPU-friendly way without heavy computes.
"""

from __future__ import annotations

from typing import Dict, Any
import cupy as cp


class EngineMetrics:
    def record_input_characteristics(self, data) -> Dict[str, Any]:
        try:
            n_rows = int(len(data))
            n_cols = int(len(getattr(data, 'columns', [])))
            return {'rows': n_rows, 'cols': n_cols}
        except Exception:
            return {}

    def record_transformation_impact(self, before, after) -> Dict[str, Any]:
        try:
            b = int(len(getattr(before, 'columns', [])))
            a = int(len(getattr(after, 'columns', [])))
            return {'cols_before': b, 'cols_after': a, 'new_cols': a - b}
        except Exception:
            return {}

    def record_performance_metrics(self, duration: float, memory_usage: Dict[str, Any]) -> Dict[str, Any]:
        try:
            mem = memory_usage or {}
            return {'duration_ms': int(duration * 1000), **mem}
        except Exception:
            return {'duration_ms': int(duration * 1000) if duration is not None else None}

    def record_quality_metrics(self, data) -> Dict[str, Any]:
        # sample-based simple NaN/inf share to avoid full compute
        try:
            sample = data.head(1000) if hasattr(data, 'head') else data
            total = int(len(sample)) if hasattr(sample, '__len__') else 0
            if total == 0:
                return {}
            # approximate: count NaNs in first numeric column
            for c in list(getattr(sample, 'columns', [])):
                try:
                    col = sample[c]
                    # cuDF supports isna(); compute on-device then bring small scalar
                    n_nan = int(col.isna().sum())
                    # For infs (best-effort)
                    n_inf = int(col.isin([cp.inf, -cp.inf]).sum())
                    return {
                        'nan_percent': (n_nan / max(total, 1)) * 100.0,
                        'inf_percent': (n_inf / max(total, 1)) * 100.0,
                    }
                except Exception:
                    continue
        except Exception:
            pass
        return {}

