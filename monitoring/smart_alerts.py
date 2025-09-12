"""
Smart alerting for pipeline health.

Lightweight checks for anomalies in feature counts, performance degradation,
memory usage patterns, and data quality signals.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional


class SmartAlertSystem:
    def __init__(self):
        self._alerts: List[Dict[str, Any]] = []

    def _add(self, kind: str, message: str, **fields):
        self._alerts.append({'type': kind, 'message': message, **fields})

    def check_feature_count_anomalies(self, current: int, expected: Optional[int]):
        if expected is None:
            return
        # Raise if deviation > 50%
        if expected > 0 and (abs(current - expected) / expected) > 0.5:
            self._add('feature_count', 'Large deviation in feature counts', current=current, expected=expected)

    def check_performance_degradation(self, current_time: float, baseline_time: Optional[float]):
        if baseline_time is None or baseline_time <= 0:
            return
        if current_time > baseline_time * 1.5:
            self._add('performance', 'Stage slower than baseline', current=current_time, baseline=baseline_time)

    def check_memory_usage_patterns(self, usage_history: List[Dict[str, Any]]):
        # naive: flag if last GPU usage > 90%
        try:
            if not usage_history:
                return
            last = usage_history[-1]
            gpu = (last.get('gpu') or {})
            if gpu.get('usage_percent', 0) > 0.90:
                self._add('memory', 'High GPU memory pressure', usage_percent=gpu.get('usage_percent'))
        except Exception:
            pass

    def check_data_quality_issues(self, quality_metrics: Dict[str, Any]):
        # flag high NaN ratio
        try:
            nan_pct = float(quality_metrics.get('nan_percent', 0))
            if nan_pct > 50.0:
                self._add('data_quality', 'High NaN percentage', nan_percent=nan_pct)
        except Exception:
            pass

    def get_alerts(self) -> List[Dict[str, Any]]:
        return list(self._alerts)

