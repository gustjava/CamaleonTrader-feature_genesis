"""
Dask Dashboard Plugins for Pipeline Visibility

Adds worker-level custom metrics and parses task names to expose
human-friendly information in the Dask dashboard.
"""

from __future__ import annotations

import time
from typing import Optional

try:
    from dask.distributed import WorkerPlugin
except Exception:  # pragma: no cover - safety for environments without dask.distributed
    class WorkerPlugin:  # type: ignore
        pass


class PipelineWorkerPlugin(WorkerPlugin):
    """Worker plugin that exposes GPU memory and active task name as custom metrics.

    Shows up under each worker page in the Dask dashboard under "Custom Metrics".
    - gpu_mem_used_gb / gpu_mem_total_gb: simple CuPy-based metrics
    - active_task: last task key observed transitioning to executing
    """

    def __init__(self, poll_interval_s: float = 2.0):
        self.poll_interval_s = float(poll_interval_s)
        self._pc = None
        self._cp = None
        self.worker = None

    def setup(self, worker):  # type: ignore[override]
        self.worker = worker
        # Try to import CuPy on workers
        try:
            import cupy as cp  # noqa: WPS433
            self._cp = cp
        except Exception:
            self._cp = None

        # Initialize metrics
        worker.metrics.update({
            'gpu_mem_used_gb': 0.0,
            'gpu_mem_total_gb': 0.0,
            'active_task': None,
            'last_task_update_ts': time.time(),
        })

        # Periodically update GPU memory metrics
        try:
            from tornado.ioloop import PeriodicCallback  # noqa: WPS433

            def _update_metrics():
                try:
                    if self._cp is not None:
                        free_b, total_b = self._cp.cuda.runtime.memGetInfo()
                        used_gb = (total_b - free_b) / (1024 ** 3)
                        total_gb = total_b / (1024 ** 3)
                        worker.metrics['gpu_mem_used_gb'] = round(float(used_gb), 2)
                        worker.metrics['gpu_mem_total_gb'] = round(float(total_gb), 2)
                except Exception:
                    # keep silent to avoid noisy logs
                    pass

            self._pc = PeriodicCallback(_update_metrics, int(self.poll_interval_s * 1000))
            self._pc.start()
        except Exception:
            # Periodic updates not available; keep static metrics
            self._pc = None

    def teardown(self, worker):  # type: ignore[override]
        try:
            if self._pc is not None:
                self._pc.stop()
        except Exception:
            pass
        self._pc = None

    # Called when tasks change state; we parse the key to expose a short label
    def transition(self, key: str, start: str, finish: str, **kwargs):  # type: ignore[override]
        try:
            if finish == 'executing' and self.worker is not None:
                label = str(key)
                # Keep label compact in dashboard
                if len(label) > 120:
                    label = label[:117] + '...'
                self.worker.metrics['active_task'] = label
                self.worker.metrics['last_task_update_ts'] = time.time()
        except Exception:
            pass

