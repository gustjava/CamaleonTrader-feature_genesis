"""
Main Orchestration Script for Feature Engineering Pipeline

This script manages the Dask-CUDA cluster lifecycle and provides the foundation
for the GPU-accelerated feature engineering pipeline using the new modular architecture.

**Pipeline Overview:**

The pipeline is designed to perform feature engineering on a large dataset of currency pair data.
It uses a Dask-CUDA cluster to distribute the workload across multiple GPUs, enabling
efficient processing of large volumes of data.

The main steps of the pipeline are:
1.  **Initialization:**
    - Load the unified configuration from `config.yaml` and environment variables.
    - Set up logging using the configuration from `config/logging.yaml`.
    - Initialize the `PipelineOrchestrator`.
2.  **Task Discovery:**
    - The `PipelineOrchestrator` discovers the currency pairs that need to be processed.
3.  **Dask Cluster Management:**
    - A `DaskClusterManager` is created to manage the Dask-CUDA cluster.
    - The cluster is started, and the Dask client is created.
4.  **Pipeline Execution:**
    - The `PipelineOrchestrator` executes the feature engineering pipeline on the Dask cluster.
    - The `process_currency_pair_worker` function is called for each currency pair.
5.  **Shutdown:**
    - The Dask cluster is shut down gracefully.
    - The pipeline summary is logged.
"""

import logging
import logging.config
import sys
import os
import signal
import time
import threading
import yaml
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client
    import cupy as cp
    import cudf
except ImportError as e:
    print(f"Error importing Dask-CUDA libraries: {e}")
    print("Make sure the GPU environment is properly set up.")
    sys.exit(1)

from config.unified_config import get_unified_config
from orchestration.pipeline_orchestrator import PipelineOrchestrator
from features.base_engine import CriticalPipelineError
from utils.logging_utils import (
    get_logger,
    info_event,
    warn_event,
    error_event,
    critical_event,
)
from utils import log_context

def setup_logging(default_path='config/logging.yaml', default_level=logging.INFO):
    """Set up logging configuration."""
    path = default_path
    if os.path.exists(path):
        with open(path, 'rt') as f:
            try:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
            except Exception as e:
                print(f"Error reading logging configuration: {e}")
                logging.basicConfig(level=default_level)
    else:
        logging.basicConfig(level=default_level)
        print("logging.yaml not found, using basic logging.")

setup_logging()
logger = get_logger(__name__, component="orchestration.main")

# Global flag for emergency shutdown
EMERGENCY_SHUTDOWN = threading.Event()


def emergency_shutdown_handler(signum, frame):
    """Handle emergency shutdown signals."""
    critical_event(logger, "pipeline.abort", "Emergency shutdown signal received.")
    critical_event(logger, "pipeline.abort", "Initiating immediate shutdown of all processes...")
    EMERGENCY_SHUTDOWN.set()
    sys.exit(1)


# Register signal handlers for emergency shutdown
signal.signal(signal.SIGINT, emergency_shutdown_handler)
signal.signal(signal.SIGTERM, emergency_shutdown_handler)


class DaskClusterManager:
    """Manages the Dask-CUDA cluster lifecycle for the feature engineering pipeline."""

    def __init__(self):
        """Initialize the cluster manager with unified configuration."""
        self.config = get_unified_config()
        self.cluster: Optional[LocalCUDACluster] = None
        self.client: Optional[Client] = None
        self.hostname = os.uname().nodename if hasattr(os, 'uname') else 'unknown'

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        info_event(logger, "pipeline.signal", "Received shutdown signal, initiating graceful shutdown", signal=signum)
        self.shutdown()
        sys.exit(0)

    def _get_gpu_count(self) -> int:
        """Get the number of available GPUs."""
        try:
            return cp.cuda.runtime.getDeviceCount()
        except Exception as e:
            warn_event(logger, "cluster.gpu.detect.warn", "Could not detect GPU count", error=str(e))
            return 1

    def _configure_rmm(self):
        """Configure RMM (RAPIDS Memory Manager) for optimal memory management."""
        try:
            from rmm import reinitialize
            
            def parse_size_gb(val: str) -> float:
                v = str(val).strip().upper()
                if v.endswith('GB'):
                    return float(v[:-2])
                if v.endswith('MB'):
                    return float(v[:-2]) / 1024.0
                return float(v)

            try:
                free_b, total_b = cp.cuda.runtime.memGetInfo()
                total_gb = total_b / (1024 ** 3)
            except Exception:
                total_gb = 8.0

            pool_frac = float(getattr(self.config.dask, 'rmm_pool_fraction', 0.0) or 0.0)
            init_frac = float(getattr(self.config.dask, 'rmm_initial_pool_fraction', 0.0) or 0.0)
            max_frac = float(getattr(self.config.dask, 'rmm_maximum_pool_fraction', 0.0) or 0.0)

            if pool_frac > 0.0:
                desired_pool_gb = max(0.25, total_gb * pool_frac)
            else:
                desired_pool_gb = parse_size_gb(self.config.dask.rmm_pool_size)

            if init_frac > 0.0:
                desired_init_gb = max(0.25, total_gb * init_frac)
            else:
                desired_init_gb = parse_size_gb(self.config.dask.rmm_initial_pool_size)

            if max_frac > 0.0:
                cap_gb = max(0.25, total_gb * max_frac)
            else:
                cap_gb = max(0.25, total_gb * 0.60)
            safe_pool_gb = max(0.25, min(desired_pool_gb, cap_gb))
            safe_init_gb = max(0.25, min(desired_init_gb, safe_pool_gb))

            self._safe_rmm_pool_size_str = f"{safe_pool_gb:.2f}GB"
            initial_pool_size = int(safe_init_gb * (1024 ** 3))

            try:
                reinitialize(
                    pool_allocator=True,
                    initial_pool_size=initial_pool_size,
                    managed_memory=False
                )
                info_event(
                    logger,
                    "cluster.rmm.config",
                    "RMM configured (pool)",
                    initial_gb=round(safe_init_gb, 2),
                    pool_gb=round(safe_pool_gb, 2),
                    cap_gb=round(cap_gb, 2),
                    total_gb=round(total_gb, 2),
                )
            except Exception as e_pool:
                warn_event(
                    logger,
                    "cluster.rmm.fallback",
                    "RMM pool init failed; falling back to default CUDA allocator",
                    error=str(e_pool),
                )
                os.environ.setdefault("RMM_ALLOCATOR", "cuda_malloc")
            
        except ImportError:
            warn_event(logger, "cluster.rmm.missing", "RMM not available, using default CUDA management")
            os.environ.setdefault("RMM_ALLOCATOR", "cuda_malloc")
        except Exception as e:
            error_event(logger, "cluster.rmm.error", "Failed to configure RMM", error=str(e))
            os.environ.setdefault("RMM_ALLOCATOR", "cuda_malloc")

    def start_cluster(self) -> bool:
        """
        Start the Dask-CUDA cluster with proper RMM configuration.
        Returns:
            bool: True if cluster started successfully, False otherwise
        """
        try:
            info_event(logger, "cluster.start", "Starting Dask-CUDA cluster...")
            gpu_count = 1  # Force 1 worker for debugging
            info_event(logger, "cluster.gpu.detect", "Detected GPU(s)", gpu_count=gpu_count)

            self._configure_rmm()

            try:
                import dask
                dask.config.set({
                    'distributed.worker.memory.target': float(self.config.dask.memory_target_fraction),
                    'distributed.worker.memory.spill': float(self.config.dask.memory_spill_fraction),
                })
                info_event(
                    logger,
                    "cluster.memory.config",
                    "Dask memory config set",
                    target=float(self.config.dask.memory_target_fraction),
                    spill=float(self.config.dask.memory_spill_fraction),
                )
            except Exception as e:
                warn_event(logger, "cluster.memory.config.warn", "Could not set Dask memory config", error=str(e))

            cluster_kwargs = {
                'n_workers': gpu_count,
                'threads_per_worker': self.config.dask.threads_per_worker,
                'rmm_pool_size': getattr(self, '_safe_rmm_pool_size_str', self.config.dask.rmm_pool_size),
                'local_directory': self.config.dask.local_directory,
                'dashboard_address': f'localhost:{self.config.monitoring.dashboard_port}',
                'scheduler_port': self.config.monitoring.dashboard_port + 1,
            }

            if self.config.dask.protocol == "ucx":
                cluster_kwargs.update({
                    'protocol': "ucx",
                    'enable_tcp_over_ucx': self.config.dask.enable_tcp_over_ucx,
                    'enable_infiniband': self.config.dask.enable_infiniband,
                    'enable_nvlink': self.config.dask.enable_nvlink,
                })

            info_event(logger, "cluster.create", f"Creating LocalCUDACluster with protocol: {self.config.dask.protocol}")
            info_event(logger, "cluster.kwargs", f"Cluster kwargs: {cluster_kwargs}")
            
            info_event(logger, "cluster.create.start", "Starting LocalCUDACluster creation...")
            try:
                import signal
                import threading
                
                def timeout_handler():
                    time.sleep(30)  # 30 second timeout
                    if not hasattr(self, 'cluster') or self.cluster is None:
                        warn_event(logger, "cluster.cuda.timeout", "LocalCUDACluster creation timeout, forcing fallback")
                        raise TimeoutError("LocalCUDACluster creation timeout")
                
                timeout_thread = threading.Thread(target=timeout_handler, daemon=True)
                timeout_thread.start()
                
                self.cluster = LocalCUDACluster(**cluster_kwargs)
                info_event(logger, "cluster.create.success", "LocalCUDACluster created successfully")
            except Exception as cuda_err:
                warn_event(logger, "cluster.cuda.fallback", f"LocalCUDACluster failed, falling back to LocalCluster: {cuda_err}")
                # Fallback to regular Dask cluster
                from dask.distributed import LocalCluster
                fallback_kwargs = {
                    'n_workers': gpu_count,
                    'threads_per_worker': 2,
                    'memory_limit': '2GB',
                    'dashboard_address': f'localhost:{self.config.monitoring.dashboard_port}',
                }
                info_event(logger, "cluster.fallback.kwargs", f"Fallback cluster kwargs: {fallback_kwargs}")
                self.cluster = LocalCluster(**fallback_kwargs)
                info_event(logger, "cluster.fallback.success", "LocalCluster created successfully as fallback")

            info_event(logger, "cluster.created", "Cluster created successfully")
            
            info_event(logger, "client.create.start", "Creating Dask Client...")
            self.client = Client(self.cluster)
            info_event(logger, "client.create.success", "Client created successfully")

            info_event(logger, "cluster.workers.wait", f"Waiting for {gpu_count} workers to be ready (timeout: 300s)...")
            self.client.wait_for_workers(gpu_count, timeout=300)
            info_event(logger, "cluster.workers.ready", "Workers are ready")
            
            info_event(logger, "cluster.ready", "Workers ready", workers=gpu_count, gpu_count=gpu_count)
            info_event(logger, "cluster.dashboard", "Dashboard available", url=self.client.dashboard_link)
            
            info_event(logger, "cluster.scheduler.info", "Getting scheduler info...")
            scheduler_info = self.client.scheduler_info()
            info_event(logger, "cluster.workers.active", "Active workers", count=len(scheduler_info['workers']))
            
            # Set cluster context
            log_context.set_cluster_info(
                hostname=self.hostname,
                gpu_count=gpu_count,
                workers=len(self.client.scheduler_info()['workers']),
                dashboard_url=self.client.dashboard_link
            )
            
            self._setup_worker_monitoring()
            
            return True

        except Exception as e:
            error_event(logger, "cluster.error", f"Failed to start Dask-CUDA cluster: {e}")
            self.shutdown()
            return False

    def get_client(self) -> Optional[Client]:
        """Get the Dask client instance."""
        return self.client

    def get_cluster(self) -> Optional[LocalCUDACluster]:
        """Get the Dask-CUDA cluster instance."""
        return self.cluster

    def is_active(self) -> bool:
        """Check if the cluster is active."""
        return self.cluster is not None and self.client is not None
        
    def shutdown(self):
        """Shutdown the cluster and client gracefully."""
        info_event(logger, "cluster.shutdown.start", "Shutting down Dask-CUDA cluster...")
        try:
            if self.client:
                self.client.close()
            if self.cluster:
                self.cluster.close()
            info_event(logger, "cluster.shutdown.end", "Dask-CUDA cluster shutdown complete")
        except Exception as e:
            error_event(logger, "cluster.shutdown.error", "Error during cluster shutdown", error=str(e))
        finally:
            self.client = None
            self.cluster = None
    
    def __enter__(self):
        """Context manager entry."""
        if not self.start_cluster():
            raise RuntimeError("Failed to start Dask-CUDA cluster")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()

    def _setup_worker_monitoring(self):
        """Set up monitoring to detect worker deaths and stop pipeline."""
        try:
            self.initial_worker_count = len(self.client.scheduler_info()["workers"])
            info_event(logger, "cluster.monitor.start", "Monitoring workers for failures", initial_workers=self.initial_worker_count)
            
            def monitor_workers():
                while True:
                    try:
                        current_workers = len(self.client.scheduler_info()["workers"])
                        if current_workers < self.initial_worker_count:
                            critical_event(logger, "cluster.worker.loss", "Worker death detected", current=current_workers, initial=self.initial_worker_count)
                            critical_event(logger, "pipeline.abort", "Stopping pipeline immediately due to worker loss")
                            EMERGENCY_SHUTDOWN.set()
                            break
                        time.sleep(5)
                    except Exception as e:
                        error_event(logger, "cluster.monitor.error", "Error in worker monitoring", error=str(e))
                        break
            
            monitor_thread = threading.Thread(target=monitor_workers, daemon=True)
            monitor_thread.start()
            
        except Exception as e:
            error_event(logger, "cluster.monitor.setup.error", "Failed to setup worker monitoring", error=str(e))


@contextmanager
def managed_dask_cluster():
    """Context manager for Dask-CUDA cluster lifecycle."""
    cluster_manager = DaskClusterManager()
    try:
        if not cluster_manager.start_cluster():
            raise RuntimeError("Failed to start Dask-CUDA cluster")
        yield cluster_manager
    finally:
        cluster_manager.shutdown()


def run_pipeline():
    """Run the complete feature engineering pipeline using the new modular architecture."""
    import socket
    import time
    
    # Set up run context
    run_id = int(time.time())
    hostname = socket.gethostname()
    log_context.set_run_id(run_id)
    log_context.set_cluster_info(hostname=hostname)
    
    info_event(logger, "pipeline.start", "Feature engineering pipeline execution started", 
               run_id=run_id, hostname=hostname)

    try:
        # Step 1: Get unified configuration
        config = get_unified_config()
        
        # Step 2: Initialize pipeline orchestrator
        orchestrator = PipelineOrchestrator()
        
        # Step 3: Connect to database (non-fatal)
        if not orchestrator.connect_database():
            warn_event(logger, "db.unavailable", "Database unavailable; continuing without task tracking.")

        # Step 4: Discover tasks
        pending_tasks = orchestrator.discover_tasks()
        if not pending_tasks:
            info_event(logger, "task.discovery.empty", "No tasks to process. Pipeline complete.")
            return 0

        info_event(logger, "task.discovery.summary", "Pending tasks to process", count=len(pending_tasks))

        # Step 5: Execute pipeline with cluster management
        with managed_dask_cluster() as cluster_manager:
            if not cluster_manager.is_active():
                logger.error("Cluster manager is not active")
                raise RuntimeError("Failed to start or activate Dask cluster.")
            
            client = cluster_manager.get_client()
            if not client:
                logger.error("Failed to get Dask client")
                raise RuntimeError("Failed to get Dask client.")
            
            # Step 5a: Start DB-backed run lifecycle (best-effort)
            try:
                orchestrator.start_run(
                    dashboard_url=getattr(client, 'dashboard_link', None),
                    hostname=cluster_manager.hostname,
                )
            except Exception as e:
                warn_event(logger, "run.lifecycle.warn", "Run lifecycle tracking unavailable", error=str(e))

            # Step 5b: Execute the pipeline
            result = orchestrator.execute_pipeline(cluster_manager, client)
            
            # Step 5c: Log pipeline summary
            orchestrator.log_pipeline_summary(result, len(pending_tasks))
            
            # Step 5d: Clean up
            orchestrator.cleanup()
            
            # Step 5e: Check if emergency shutdown was triggered
            if result.emergency_shutdown:
                critical_event(logger, "pipeline.abort", "Pipeline stopped due to emergency shutdown")
                try:
                    orchestrator.end_run(status='ABORTED')
                finally:
                    return 1
            
            exit_code = 0 if result.failed_tasks == 0 else 1
            try:
                orchestrator.end_run(status='COMPLETED' if exit_code == 0 else 'FAILED')
                info_event(logger, "pipeline.end", "Pipeline execution completed successfully", 
                           run_id=run_id, exit_code=exit_code, 
                           total_tasks=len(pending_tasks), failed_tasks=result.failed_tasks)
            finally:
                return exit_code

    except Exception as e:
        error_event(logger, "pipeline.error", f"Fatal pipeline error: {e}", run_id=run_id)
        info_event(logger, "pipeline.end", "Pipeline execution failed", run_id=run_id, exit_code=1)
        return 1


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(run_pipeline())
