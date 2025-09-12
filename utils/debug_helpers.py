"""
Debug helpers for pipeline transparency.

Provides utilities to trace feature origins, explain transformations
and diagnose performance issues in a structured, low-overhead manner.
"""

from __future__ import annotations

from typing import Dict, Any, Optional


class PipelineDebugger:
    def __init__(self):
        # Mapping from feature -> origin metadata (filled by engines optionally)
        self._origin: Dict[str, Dict[str, Any]] = {}

    def trace_feature_origin(self, feature_name: str) -> str:
        meta = self._origin.get(feature_name)
        if not meta:
            return f"No origin metadata for feature: {feature_name}"
        parts = [f"Feature: {feature_name}"]
        src = meta.get('source')
        if src:
            parts.append(f"  Source: {src}")
        op = meta.get('operation')
        if op:
            parts.append(f"  Operation: {op}")
        params = meta.get('params')
        if params:
            parts.append(f"  Params: {params}")
        return "\n".join(parts)

    def explain_transformation(self, input_data, output_data) -> str:
        # Simple structural comparison, avoids heavy stats
        try:
            in_cols = len(getattr(input_data, 'columns', []))
            out_cols = len(getattr(output_data, 'columns', []))
            return f"Columns: {in_cols} -> {out_cols} (+{out_cols - in_cols})"
        except Exception:
            return "Could not compare inputs and outputs"

    def diagnose_performance_issues(self, metrics: Dict[str, Any]) -> list:
        suggestions = []
        dur = metrics.get('duration_ms')
        if isinstance(dur, (int, float)) and dur > 60_000:
            suggestions.append('Stage took >60s; consider reducing candidate set or enabling chunking')
        if metrics.get('gpu_usage', 0) > 0.9:
            suggestions.append('GPU memory pressure high; enable spilling or reduce batch size')
        return suggestions

    def suggest_config_optimizations(self, current_config: Any) -> Dict[str, Any]:
        try:
            return {
                'stage1_top_n': getattr(current_config.features, 'stage1_top_n', None),
                'recommendation': 'Tune stage1_top_n to balance transparency and throughput',
            }
        except Exception:
            return {'recommendation': 'Review features.* thresholds and deny lists'}

