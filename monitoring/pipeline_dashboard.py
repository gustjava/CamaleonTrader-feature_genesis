"""
Text-based pipeline dashboard utilities.

Provides simple, low-overhead functions to present real-time-ish progress
and resource usage in logs or CLI-friendly strings. This avoids GUI deps.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import time

try:
    from utils.memory_monitor import MemoryMonitor
except Exception:
    class MemoryMonitor:  # type: ignore
        def __init__(self, *_, **__):
            pass
        def get_memory_summary(self):
            return {'gpu': {}, 'system': {}, 'timestamp': time.time(), 'is_monitoring': False}


@dataclass
class StageCounter:
    name: str
    before_cols: Optional[int] = None
    after_cols: Optional[int] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    def complete(self):
        self.finished_at = time.time()

    @property
    def duration_sec(self) -> Optional[float]:
        if self.finished_at is None:
            return None
        return self.finished_at - self.started_at


class PipelineDashboard:
    def __init__(self):
        self._mm = MemoryMonitor()
        self._progress: Dict[str, Any] = {
            'total_pairs': 0,
            'completed_pairs': 0,
            'current_pair': None,
            'stages': {},  # stage_name -> StageCounter
        }

    # ---- Public API ----
    def show_real_time_progress(self) -> str:
        p = self._progress
        bars = self._make_bar(p.get('completed_pairs', 0), p.get('total_pairs', 1))
        lines = [
            "PIPELINE FEATURE GENESIS - Progress",
            "=" * 40,
            f"Pairs: {p.get('completed_pairs', 0)}/{p.get('total_pairs', 0)} {bars}",
            f"Current: {p.get('current_pair') or '-'}",
        ]
        for name, sc in p.get('stages', {}).items():
            dur = f"{sc.duration_sec:.1f}s" if sc.duration_sec else "..."
            lines.append(f" - {name}: {sc.before_cols}->{sc.after_cols} ({dur})")
        return "\n".join(lines)

    def display_feature_counts_by_stage(self) -> Dict[str, Any]:
        return {k: {
            'before': v.before_cols,
            'after': v.after_cols,
            'delta': (v.after_cols - v.before_cols) if (v.after_cols is not None and v.before_cols is not None) else None
        } for k, v in self._progress.get('stages', {}).items()}

    def show_memory_usage_evolution(self) -> List[Dict[str, Any]]:
        # On-demand snapshot
        return [self._mm.get_memory_summary()]

    def display_config_impact_summary(self) -> Dict[str, Any]:
        # Placeholder: dashboard consumes visualizer output if provided externally
        return {'summary': 'Attach PipelineVisualizer.generate_config_impact_analysis() here'}

    def show_error_analysis(self) -> Dict[str, Any]:
        # Placeholder for error aggregation
        return {'errors': []}

    # ---- Helpers State API (called by orchestration) ----
    def set_total_pairs(self, n: int):
        self._progress['total_pairs'] = int(n)

    def set_current_pair(self, pair: Optional[str]):
        self._progress['current_pair'] = pair

    def increment_completed(self):
        self._progress['completed_pairs'] = int(self._progress.get('completed_pairs', 0)) + 1

    def stage_start(self, name: str, before_cols: Optional[int] = None):
        sc = StageCounter(name=name, before_cols=before_cols)
        self._progress['stages'][name] = sc

    def stage_end(self, name: str, after_cols: Optional[int] = None):
        sc = self._progress['stages'].get(name)
        if sc:
            sc.after_cols = after_cols
            sc.complete()

    # ---- Private ----
    def _make_bar(self, current: int, total: int, width: int = 30) -> str:
        total = max(1, int(total))
        ratio = max(0.0, min(1.0, float(current) / float(total)))
        fill = int(ratio * width)
        return "[" + ("█" * fill) + ("-" * (width - fill)) + "]"
