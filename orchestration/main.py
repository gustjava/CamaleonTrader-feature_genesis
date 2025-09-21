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
import json
import traceback
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List, Sequence
from contextlib import contextmanager

import yaml

try:  # Optional dependency for study orchestration
    import hydra
    from hydra.utils import to_absolute_path
except ModuleNotFoundError:  # pragma: no cover - handled gracefully
    hydra = None  # type: ignore

    def to_absolute_path(path: str) -> str:
        """Fallback resolver when Hydra is unavailable."""
        return os.path.abspath(path)

try:
    from omegaconf import DictConfig, OmegaConf
except ModuleNotFoundError:  # pragma: no cover - handled gracefully
    DictConfig = Any  # type: ignore

    class _OmegaConfShim:
        @staticmethod
        def to_container(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:  # type: ignore
            raise ModuleNotFoundError(
                "omegaconf is required for Hydra-based study orchestration."
            )

        @staticmethod
        def to_yaml(*_args: Any, **_kwargs: Any) -> str:  # type: ignore
            raise ModuleNotFoundError(
                "omegaconf is required for Hydra-based study orchestration."
            )

    OmegaConf = _OmegaConfShim()  # type: ignore

try:
    import optuna
except ModuleNotFoundError:  # pragma: no cover - handled gracefully
    optuna = None  # type: ignore

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
from monitoring.dask_plugins import PipelineWorkerPlugin
from orchestration.pipeline_orchestrator import PipelineOrchestrator
from orchestration.objectives import objective_study_a, objective_study_b
from features.base_engine import CriticalPipelineError
from utils.logging_utils import (
    get_logger,
)
from data_io.r2_uploader import R2ModelUploader

def setup_logging(default_path='config/logging.yaml', default_level=logging.INFO):
    """Set up logging configuration."""
    path = default_path
    if os.path.exists(path):
        with open(path, 'rt') as f:
            try:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
                
                # Apply custom currency formatter to console handlers
                try:
                    from utils.currency_formatter import CurrencyConsoleFormatter
                    formatter = CurrencyConsoleFormatter()
                    
                    # Apply to root handlers
                    for handler in logging.root.handlers:
                        if isinstance(handler, logging.StreamHandler):
                            handler.setFormatter(formatter)
                    
                    # Apply to specific logger handlers (orchestration, features, etc.)
                    for logger_name in ['orchestration', 'features', 'data_io', 'utils']:
                        logger = logging.getLogger(logger_name)
                        for handler in logger.handlers:
                            if isinstance(handler, logging.StreamHandler):
                                handler.setFormatter(formatter)
                                
                except Exception as e:
                    print(f"Warning: Could not apply currency formatter: {e}")
                
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
    logger.critical("Emergency shutdown signal received.")
    logger.critical("Initiating immediate shutdown of all processes...")
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
        logger.info(f"Received shutdown signal {signum}, initiating graceful shutdown")
        self.shutdown()
        sys.exit(0)

    def _get_gpu_count(self) -> int:
        """Get the number of available GPUs."""
        try:
            return cp.cuda.runtime.getDeviceCount()
        except Exception as e:
            logger.warning(f"Could not detect GPU count: {e}")
            return 1

    def _get_system_memory_gb(self) -> float:
        """Get total system memory in GB."""
        try:
            import psutil
            total_bytes = psutil.virtual_memory().total
            return total_bytes / (1024 ** 3)
        except Exception as e:
            logger.warning(f"Could not detect system memory: {e}")
            return 8.0  # Default fallback

    def _calculate_memory_limit(self, gpu_count: int) -> str:
        """Calculate memory limit per worker to always use 80% of total system RAM."""
        try:
            # Check if using fraction-based configuration
            memory_fraction = float(getattr(self.config.dask, 'memory_limit_fraction', 0.0) or 0.0)
            
            if memory_fraction > 0.0:
                system_memory_gb = self._get_system_memory_gb()
                
                # Use configured fraction of total system RAM, divided equally among workers
                total_memory_to_use = system_memory_gb * memory_fraction
                memory_per_worker_gb = total_memory_to_use / gpu_count
                
                # Apply safety limits
                min_memory_gb = 0.5  # Minimum 500MB per worker
                memory_per_worker_gb = max(min_memory_gb, memory_per_worker_gb)
                
                total_memory_usage = memory_per_worker_gb * gpu_count
                actual_fraction = total_memory_usage / system_memory_gb
                
                logger.info(f"Dynamic memory calculation: {gpu_count} workers, {memory_per_worker_gb:.2f}GB per worker "
                           f"(system: {system_memory_gb:.2f}GB, total: {total_memory_usage:.2f}GB, {actual_fraction:.1%}, fraction: {memory_fraction:.1%})")
                
                return f"{memory_per_worker_gb:.2f}GB"
            else:
                # Use fixed memory limit
                return self.config.dask.memory_limit
                
        except Exception as e:
            logger.warning(f"Could not calculate memory limit: {e}")
            return "2GB"  # Safe fallback

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
                free_gb = free_b / (1024 ** 3)
                total_gb = total_b / (1024 ** 3)
            except Exception:
                # Conservative defaults if we cannot query memory
                free_gb = 4.0
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

            # New: also cap by currently free memory with headroom to avoid init failures
            free_headroom = 0.85  # keep some free space for context/UCX/cublas etc.
            max_pool_by_free = max(0.25, free_gb * free_headroom)
            safe_pool_gb = max(0.25, min(desired_pool_gb, cap_gb, max_pool_by_free))

            # Initial pool should be smaller; also obey free memory headroom (tighter bound)
            max_init_by_free = max(0.25, free_gb * 0.50)
            safe_init_gb = max(0.25, min(desired_init_gb, safe_pool_gb * 0.90, max_init_by_free))

            # Ensure initial does not exceed pool size
            if safe_init_gb > safe_pool_gb:
                safe_init_gb = max(0.25, min(safe_pool_gb * 0.90, max_init_by_free))

            self._safe_rmm_pool_size_str = f"{safe_pool_gb:.2f}GB"
            self._safe_rmm_initial_pool_size_str = f"{safe_init_gb:.2f}GB"

            # Compute initial pool size (bytes) and align to 256-byte boundary as required by RMM
            bytes_per_gb = 1024 ** 3
            raw_init_bytes = int(safe_init_gb * bytes_per_gb)
            # Ensure alignment to 256 bytes and non-zero
            def _align_256(n: int) -> int:
                if n <= 0:
                    return 256
                return max(256, (n // 256) * 256)
            initial_pool_size = _align_256(raw_init_bytes)
            # Cap initial pool to not exceed intended pool size
            try:
                pool_cap_bytes = int(safe_pool_gb * bytes_per_gb)
                if initial_pool_size > pool_cap_bytes:
                    initial_pool_size = _align_256(pool_cap_bytes)
            except Exception as e:
                logger.error(f"Failed to adjust initial pool size: {e}")
                pass

            try:
                reinitialize(
                    pool_allocator=True,
                    initial_pool_size=initial_pool_size,
                    managed_memory=False
                )
                logger.info(
                    f"RMM configured (pool): initial={initial_pool_size/bytes_per_gb:.2f}GB, "
                    f"pool={safe_pool_gb:.2f}GB, cap={cap_gb:.2f}GB, total={total_gb:.2f}GB, free={free_gb:.2f}GB"
                )
            except Exception as e_pool:
                logger.warning(f"RMM pool init failed; falling back to default CUDA allocator: {e_pool}")
                os.environ.setdefault("RMM_ALLOCATOR", "cuda_malloc")
            
        except ImportError:
            logger.warning("RMM not available, using default CUDA management")
            os.environ.setdefault("RMM_ALLOCATOR", "cuda_malloc")
        except Exception as e:
            logger.error(f"Failed to configure RMM: {e}")
            os.environ.setdefault("RMM_ALLOCATOR", "cuda_malloc")

    def _check_port_availability(self, port: int) -> bool:
        """Check if a port is available for use."""
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return True
        except OSError:
            return False

    def _find_available_dashboard_port(self, start_port: int) -> int:
        """Find an available port for the dashboard, starting from start_port."""
        port = start_port
        max_attempts = 10
        
        for _ in range(max_attempts):
            if self._check_port_availability(port):
                return port
            port += 1
        
        logger.warning(f"Could not find available port starting from {start_port}, using {port}")
        return port

    def _check_dashboard_health(self, port: int) -> bool:
        """Check if the dashboard is responding properly."""
        try:
            import requests
            response = requests.get(f"http://localhost:{port}/status", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def _wait_for_dashboard_ready(self, port: int, timeout: int = 30) -> bool:
        """Wait for the dashboard to be ready with retry logic."""
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self._check_dashboard_health(port):
                return True
            time.sleep(2)
        
        logger.warning(f"Dashboard health check failed after {timeout}s")
        return False

    def start_cluster(self) -> bool:
        """
        Start the Dask-CUDA cluster with proper RMM configuration.
        Returns:
            bool: True if cluster started successfully, False otherwise
        """
        try:
            # Enforce single-threaded CPU math libs before spawning workers
            try:
                import os as _os
                _os.environ.setdefault('OMP_NUM_THREADS', '1')
                _os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
                _os.environ.setdefault('MKL_NUM_THREADS', '1')
                _os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
                _os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
                _os.environ.setdefault('BLIS_NUM_THREADS', '1')
                _os.environ.setdefault('KMP_AFFINITY', 'granularity=fine,compact,1,0')
                logger.info("CPU thread limits set for workers: OMP/BLAS/MKL=1")
            except Exception:
                pass
            logger.info("Starting Dask-CUDA cluster...")
            gpu_count = max(1, int(self._get_gpu_count()))
            logger.info(f"Detected GPU(s): {gpu_count}")

            self._configure_rmm()

            try:
                import dask
                dask.config.set({
                    'distributed.worker.memory.target': float(self.config.dask.memory_target_fraction),
                    'distributed.worker.memory.spill': float(self.config.dask.memory_spill_fraction),
                })
                logger.info(f"Dask memory config set: target={self.config.dask.memory_target_fraction}, "
                           f"spill={self.config.dask.memory_spill_fraction}")
            except Exception as e:
                logger.warning(f"Could not set Dask memory config: {e}")

            # Calculate memory limit per worker (dynamic based on GPU count)
            memory_limit_str = self._calculate_memory_limit(gpu_count)
            
            # Configure dashboard (can be disabled to reduce WebSocket logs)
            dashboard_enabled = getattr(self.config.monitoring, 'dashboard_enabled', True)
            if dashboard_enabled:
                # Find available dashboard port
                dashboard_port = self._find_available_dashboard_port(self.config.monitoring.dashboard_port)
                if dashboard_port != self.config.monitoring.dashboard_port:
                    logger.info(f"Dashboard port {self.config.monitoring.dashboard_port} was busy, using {dashboard_port}")
                dashboard_address = f'localhost:{dashboard_port}'
                scheduler_port = dashboard_port + 1
            else:
                logger.info("Dashboard disabled - WebSocket logs will be reduced")
                dashboard_address = None
                scheduler_port = 0  # Let Dask choose automatically
            
            # Optimize cluster for Optuna studies (more workers for parallel trials)
            workers_per_gpu = getattr(self.config.dask, 'workers_per_gpu', 1)
            total_workers = gpu_count * workers_per_gpu
            
            cluster_kwargs = {
                'n_workers': total_workers,
                'threads_per_worker': self.config.dask.threads_per_worker,
                'memory_limit': memory_limit_str,
                'rmm_pool_size': getattr(self, '_safe_rmm_pool_size_str', self.config.dask.rmm_pool_size),
                'local_directory': self.config.dask.local_directory,
                'dashboard_address': dashboard_address,
                'scheduler_port': scheduler_port,
            }
            
            logger.info(f"Cluster optimized for Optuna: {gpu_count} GPUs × {workers_per_gpu} workers = {total_workers} total workers")

            # Explicitly set protocol (e.g., 'tcp' for stability)
            try:
                cluster_kwargs['protocol'] = str(self.config.dask.protocol)
            except Exception as e:
                logger.error(f"Failed to set cluster protocol: {e}")
                pass

            if self.config.dask.protocol == "ucx":
                cluster_kwargs.update({
                    'protocol': "ucx",
                    'enable_tcp_over_ucx': self.config.dask.enable_tcp_over_ucx,
                    'enable_infiniband': self.config.dask.enable_infiniband,
                    'enable_nvlink': self.config.dask.enable_nvlink,
                })

            logger.info(f"Creating LocalCUDACluster with protocol: {self.config.dask.protocol}")
            logger.info(f"Cluster kwargs: {cluster_kwargs}")
            
            logger.info("Starting LocalCUDACluster creation...")
            try:
                import signal
                import threading
                
                def timeout_handler():
                    time.sleep(30)  # 30 second timeout
                    if not hasattr(self, 'cluster') or self.cluster is None:
                        raise TimeoutError("LocalCUDACluster creation timeout")
                
                timeout_thread = threading.Thread(target=timeout_handler, daemon=True)
                timeout_thread.start()
                
                self.cluster = LocalCUDACluster(**cluster_kwargs)
                logger.info("LocalCUDACluster created successfully")
            except Exception as cuda_err:
                logger.error(f"LocalCUDACluster failed: {cuda_err}")
                # No CPU fallback: abort cluster start
                raise

            logger.info("Cluster created successfully")
            
            logger.info("Creating Dask Client...")
            self.client = Client(self.cluster)
            logger.info("Client created successfully")

            # Register worker plugin to expose GPU metrics and active task to dashboard
            try:
                self.client.register_plugin(PipelineWorkerPlugin(poll_interval_s=2.0), name="pipeline-metrics")
                logger.info("Registered PipelineWorkerPlugin for dashboard custom metrics")
            except Exception as e:
                logger.warning(f"Could not register PipelineWorkerPlugin: {e}")

            logger.info(f"Waiting for {gpu_count} workers to be ready (timeout: 300s)...")
            self.client.wait_for_workers(gpu_count, timeout=300)
            logger.info("Workers are ready")
            
            logger.info(f"Workers ready: {gpu_count} workers, {gpu_count} GPUs")
            
            # Wait for dashboard to be ready with health check
            logger.info("Checking dashboard health...")
            if self._wait_for_dashboard_ready(dashboard_port, timeout=30):
                logger.info("Dashboard is healthy and ready")
            else:
                logger.warning("Dashboard health check failed, but continuing...")
            
            # Print SSH command for dashboard access with error handling
            try:
                dashboard_url = self.client.dashboard_link
                logger.info(f"Dashboard available at: {dashboard_url}")
            except Exception as dashboard_err:
                logger.warning(f"Dashboard connection issue (non-fatal): {dashboard_err}")
                logger.info("Pipeline will continue without dashboard access")
                dashboard_url = f"http://localhost:{dashboard_port}"
            
            # Get Vast.ai instance information for SSH command
            try:
                import subprocess
                result = subprocess.run(['vastai', 'show', 'instances', '--raw'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    import json
                    instances = json.loads(result.stdout)
                    if instances:
                        instance = instances[0]  # Get first running instance
                        ssh_host = instance.get('ssh_host', 'ssh2.vast.ai')
                        ssh_port = instance.get('ssh_port', '18640')
                        logger.info("=" * 80)
                        logger.info("TO ACCESS DASHBOARD REMOTELY, RUN THIS SSH COMMAND:")
                        logger.info(f"ssh -L {dashboard_port}:localhost:{dashboard_port} root@{ssh_host} -p {ssh_port}")
                        logger.info(f"Then open: http://localhost:{dashboard_port} in your browser")
                        logger.info("=" * 80)
                    else:
                        logger.info("=" * 80)
                        logger.info("TO ACCESS DASHBOARD REMOTELY, RUN THIS SSH COMMAND:")
                        logger.info(f"ssh -L {dashboard_port}:localhost:{dashboard_port} root@ssh2.vast.ai -p 18640")
                        logger.info(f"Then open: http://localhost:{dashboard_port} in your browser")
                        logger.info("=" * 80)
                else:
                    raise Exception("Could not get instance info")
            except Exception as e:
                logger.info("=" * 80)
                logger.info("TO ACCESS DASHBOARD REMOTELY, RUN THIS SSH COMMAND:")
                logger.info(f"ssh -L {dashboard_port}:localhost:{dashboard_port} root@ssh2.vast.ai -p 18640")
                logger.info(f"Then open: http://localhost:{dashboard_port} in your browser")
                logger.info("=" * 80)
            
            logger.info("Getting scheduler info...")
            scheduler_info = self.client.scheduler_info()
            logger.info(f"Active workers: {len(scheduler_info['workers'])}")
            
            self._setup_worker_monitoring()
            
            return True

        except Exception as e:
            logger.error(f"Failed to start Dask-CUDA cluster: {e}")
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
        logger.info("Shutting down Dask-CUDA cluster...")
        try:
            if self.client:
                self.client.close()
            if self.cluster:
                self.cluster.close()
            logger.info("Dask-CUDA cluster shutdown complete")
        except Exception as e:
            logger.error(f"Error during cluster shutdown: {e}")
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
            logger.info(f"Monitoring workers for failures: {self.initial_worker_count} initial workers")
            
            def monitor_workers():
                while True:
                    try:
                        current_workers = len(self.client.scheduler_info()["workers"])
                        if current_workers < self.initial_worker_count:
                            logger.critical(f"Worker death detected: {current_workers} current, {self.initial_worker_count} initial")
                            logger.critical("Stopping pipeline immediately due to worker loss")
                            EMERGENCY_SHUTDOWN.set()
                            break
                        time.sleep(5)
                    except Exception as e:
                        logger.error(f"Error in worker monitoring: {e}")
                        break
            
            monitor_thread = threading.Thread(target=monitor_workers, daemon=True)
            monitor_thread.start()
            
        except Exception as e:
            logger.error(f"Failed to setup worker monitoring: {e}")


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
    
    # Set up hostname
    hostname = socket.gethostname()
    
    logger.info("Feature engineering pipeline execution started")

    try:
        # Step 1: Get unified configuration
        config = get_unified_config()
        
        # Step 2: Initialize pipeline orchestrator
        orchestrator = PipelineOrchestrator()
        
        # Step 3: Connect to database (non-fatal)
        if not orchestrator.connect_database():
            logger.warning("Database unavailable; continuing without task tracking.")

        # Step 4: Discover tasks
        pending_tasks = orchestrator.discover_tasks()
        if not pending_tasks:
            logger.info("No tasks to process. Pipeline complete.")
            return 0

        logger.info(f"Pending tasks to process: {len(pending_tasks)}")

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
                logger.warning(f"Run lifecycle tracking unavailable: {e}")

            # Step 5b: Execute the pipeline
            result = orchestrator.execute_pipeline(cluster_manager, client)
            
            # Step 5c: Log pipeline summary
            orchestrator.log_pipeline_summary(result, len(pending_tasks))
            
            # Step 5d: Clean up
            orchestrator.cleanup()
            
            # Step 5e: Check if emergency shutdown was triggered
            if result.emergency_shutdown:
                logger.critical("Pipeline stopped due to emergency shutdown")
                try:
                    orchestrator.end_run(status='ABORTED')
                finally:
                    return 1
            
            exit_code = 0 if result.failed_tasks == 0 else 1
            try:
                orchestrator.end_run(status='COMPLETED' if exit_code == 0 else 'FAILED')
                logger.info(f"Pipeline execution completed successfully: "
                           f"exit_code={exit_code}, total_tasks={len(pending_tasks)}, failed_tasks={result.failed_tasks}")
            finally:
                return exit_code

    except Exception as e:
        logger.error(f"Fatal pipeline error: {e}")
        logger.info("Pipeline execution failed: exit_code=1")
        return 1


# =============================================================================
# Hydra-Based Optuna Study Orchestration
# =============================================================================

def _as_abs_path(path_value: Any) -> Path:
    """Convert a config path value into an absolute pathlib.Path."""
    return Path(to_absolute_path(str(path_value))).expanduser().resolve()


def _prepare_outputs(outputs_cfg: Optional[DictConfig]) -> Dict[str, Path]:
    """Resolve and create output directories declared in the study config."""
    resolved: Dict[str, Path] = {}
    if outputs_cfg is None:
        return resolved

    outputs_dict = OmegaConf.to_container(outputs_cfg, resolve=True) or {}
    for key, value in outputs_dict.items():
        if value in (None, ""):
            continue
        path = _as_abs_path(value)
        if key == "dir":
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
        resolved[key] = path
    return resolved


def _prepare_storage(study_cfg: DictConfig, outputs: Dict[str, Path]) -> Optional[str]:
    """Prepare Optuna storage according to configuration."""
    storage_cfg = getattr(study_cfg, "storage", None)
    if storage_cfg is None:
        logger.warning("No storage configuration found - using in-memory storage")
        return None

    storage_dict = OmegaConf.to_container(storage_cfg, resolve=True) or {}
    backend = str(storage_dict.get("backend", "")).lower()
    url = storage_dict.get("url")
    clean_on_start = bool(storage_dict.get("clean_on_start", False))

    logger.info(f"Storage config: backend={backend}, url={url}, clean_on_start={clean_on_start}")
    
    # Log idempotency information
    if backend == "sqlite" and url:
        url_str = str(url)
        if url_str.startswith("sqlite:///"):
            path_fragment = url_str[len("sqlite:///"):].split('?')[0]
            db_path = _as_abs_path(path_fragment)
            if db_path.exists():
                logger.info(f"📊 Existing Optuna study database found: {db_path}")
                logger.info("🔄 Study will continue from previous state (idempotent)")
            else:
                logger.info(f"🆕 No existing study database found: {db_path}")
                logger.info("🆕 Starting new Optuna study")

    if not url:
        logger.warning("No storage URL found - using in-memory storage")
        return None

    if backend == "sqlite":
        prefix = "sqlite:///"
        url_str = str(url)
        if not url_str.startswith(prefix):
            raise ValueError("SQLite storage URL must start with 'sqlite:///'")
        
        # Remove existing query parameters if any
        path_fragment = url_str[len(prefix):].split('?')[0]
        db_path = _as_abs_path(path_fragment)
        
        if clean_on_start and db_path.exists():
            db_path.unlink()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add a timeout to the connection string to improve concurrency handling
        return f"sqlite:///{db_path}?timeout=30"

    # Non-SQLite storage: ensure parent folders exist when the URL points to a file path.
    try:
        potential_path = Path(str(url))
        if potential_path.suffix and potential_path.parent:
            potential_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    return str(url)


def _ensure_optuna_available() -> None:
    if optuna is None:
        raise ModuleNotFoundError("optuna is required for Study A/B")


def _execute_full_pipeline_with_best_params(cfg: DictConfig, best_params: Dict[str, Any]) -> None:
    """Execute the full pipeline with the best parameters found by stageA optimization."""
    logger = get_logger(__name__)
    
    try:
        logger.info("🚀 Starting full pipeline execution with optimized parameters")
        
        # Get the symbol from config
        dataset_cfg = cfg.study.dataset
        symbol = str(dataset_cfg.symbol)
        
        # Initialize the pipeline orchestrator with optimized parameters
        from orchestration.pipeline_orchestrator import PipelineOrchestrator
        
        # Create a temporary config with the best parameters applied
        temp_config = get_unified_config()
        
        # Apply best parameters to the config (this would need to be implemented based on your config structure)
        _apply_best_params_to_config(temp_config, best_params)
        
        # Initialize orchestrator
        orchestrator = PipelineOrchestrator(temp_config)
        
        # Execute the pipeline for the symbol
        logger.info(f"Executing optimized pipeline for {symbol}")
        success = orchestrator.run_pipeline_for_symbol(symbol)
        
        if success:
            logger.info("✅ Full pipeline execution completed successfully with optimized parameters")
            
            # Upload study metadata to R2
            _upload_study_metadata_to_r2(cfg, best_params, symbol)
            
        else:
            logger.error("❌ Full pipeline execution failed")
            
    except Exception as e:
        logger.error(f"❌ Error executing full pipeline with best parameters: {e}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")


def _apply_best_params_to_config(config, best_params: Dict[str, Any]) -> None:
    """Apply the best parameters from Optuna to the configuration."""
    logger = get_logger(__name__)
    
    try:
        # Apply frac_diff parameters
        if 'frac_diff_d' in best_params:
            config.features.frac_diff.d_values = [best_params['frac_diff_d']]
            logger.info(f"Applied frac_diff_d: {best_params['frac_diff_d']}")
        
        if 'frac_diff_threshold' in best_params:
            config.features.frac_diff.threshold = best_params['frac_diff_threshold']
            logger.info(f"Applied frac_diff_threshold: {best_params['frac_diff_threshold']}")
        
        # Apply dcor parameters
        if 'dcor_threshold' in best_params:
            config.features.statistical_tests.dcor_min_threshold = best_params['dcor_threshold']
            logger.info(f"Applied dcor_threshold: {best_params['dcor_threshold']}")
        
        if 'dcor_top_k' in best_params:
            config.features.statistical_tests.stage1_top_n = best_params['dcor_top_k']
            logger.info(f"Applied dcor_top_k: {best_params['dcor_top_k']}")
        
        # Apply other parameters as needed based on your search space
        # This should be customized based on the actual parameter names in your search space
        
        logger.info("✅ Best parameters applied to configuration")
        
    except Exception as e:
        logger.error(f"❌ Error applying best parameters to config: {e}")


def _upload_study_metadata_to_r2(cfg: DictConfig, best_params: Dict[str, Any], symbol: str) -> None:
    """Upload comprehensive study metadata to R2."""
    logger = get_logger(__name__)
    
    try:
        # Create comprehensive metadata
        study_metadata = {
            "study_type": "stageA_preprocess_selection",
            "symbol": symbol,
            "timestamp": pd.Timestamp.now().isoformat(),
            "best_parameters": best_params,
            "config_summary": {
                "n_trials": cfg.study.optuna.n_trials,
                "direction": cfg.study.optuna.direction,
                "sampler": cfg.study.optuna.sampler,
                "pruner": cfg.study.optuna.pruner
            }
        }
        
        # Save locally first
        outputs_dir = Path(cfg.study.outputs.dir)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = outputs_dir / f"stageA_metadata_{symbol.lower()}.json"
        
        # Write and upload
        _write_json(
            metadata_path,
            study_metadata,
            upload_to_r2=True,
            study_name="stageA_metadata",
            stage="stageA"
        )
        
        logger.info(f"📤 Study metadata uploaded to R2: {metadata_path}")
        
    except Exception as e:
        logger.error(f"❌ Error uploading study metadata to R2: {e}")


def _create_trial_callback(cfg: DictConfig, outputs: Dict[str, Any]):
    """Create a callback function to track trial progress and optionally upload results."""
    logger = get_logger(__name__)
    
    def trial_callback(study, trial):
        """Callback executed after each trial completion."""
        try:
            # Check if trial has state attribute (may not be available in Dask parallelization)
            if hasattr(trial, 'state') and trial.state == optuna.trial.TrialState.COMPLETE:
                # Log progress every 10 trials
                if trial.number % 10 == 0:
                    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
                    logger.info(f"📊 Progress: {completed_trials} trials completed (latest: {trial.number})")
                
                # Save trial details every 5 trials or for the last trial
                if (trial.number % 5 == 0) or (trial.number == cfg.study.optuna.n_trials - 1):
                    _save_trial_progress(cfg, study, trial, outputs)
            else:
                # For Dask parallelization, assume trial is complete if we're in the callback
                # Log progress every 10 trials
                if trial.number % 10 == 0:
                    completed_trials = len([t for t in study.trials if hasattr(t, 'state') and t.state == optuna.trial.TrialState.COMPLETE])
                    logger.info(f"📊 Progress: {completed_trials} trials completed (latest: {trial.number})")
                
                # Save trial details every 5 trials or for the last trial
                if (trial.number % 5 == 0) or (trial.number == cfg.study.optuna.n_trials - 1):
                    _save_trial_progress(cfg, study, trial, outputs)
                    
        except Exception as e:
            logger.error(f"❌ Error in trial callback for trial {trial.number}: {e}")
    
    return trial_callback


def _save_trial_progress(cfg: DictConfig, study, trial, outputs: Dict[str, Any]) -> None:
    """Save trial progress and optionally upload to R2."""
    logger = get_logger(__name__)
    
    try:
        # Create trial summary
        trial_summary = {
            "trial_number": trial.number,
            "state": trial.state.name if hasattr(trial, 'state') else "COMPLETE",
            "values": trial.values if hasattr(trial, 'values') and trial.values else None,
            "params": trial.params if hasattr(trial, 'params') else {},
            "user_attrs": trial.user_attrs if hasattr(trial, 'user_attrs') else {},
            "timestamp": pd.Timestamp.now().isoformat(),
            "study_stats": {
                "total_trials": len(study.trials),
                "completed_trials": len([t for t in study.trials if hasattr(t, 'state') and t.state == optuna.trial.TrialState.COMPLETE]),
                "pruned_trials": len([t for t in study.trials if hasattr(t, 'state') and t.state == optuna.trial.TrialState.PRUNED]),
                "failed_trials": len([t for t in study.trials if hasattr(t, 'state') and t.state == optuna.trial.TrialState.FAIL])
            }
        }
        
        # Save locally
        outputs_dir = Path(cfg.study.outputs.dir)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        progress_path = outputs_dir / f"trial_progress_{trial.number:04d}.json"
        
        # Write locally (don't upload individual trials to avoid spam, just save locally)
        with progress_path.open("w", encoding="utf-8") as fp:
            # Import the safe JSON converter
            from orchestration.objectives import _safe_json_convert
            json.dump(_safe_json_convert(trial_summary), fp, indent=2)
        
        logger.info(f"💾 Trial {trial.number} progress saved: {progress_path}")
        
        # Save study snapshot (last 5 trials + best trial)
        if trial.number % 5 == 0:
            # Get last 5 trials
            last_5_trials = study.trials[-5:] if len(study.trials) >= 5 else study.trials
            
            # Get best trial so far
            best_trial = None
            try:
                best_trial = study.best_trial
            except ValueError:
                # No completed trials yet
                pass
            
            # Prepare trials list (last 5 + best if not already included)
            trials_to_include = []
            best_trial_number = best_trial.number if best_trial else -1
            
            # Add last 5 trials
            for t in last_5_trials:
                trials_to_include.append({
                    "number": t.number,
                    "state": t.state.name if hasattr(t, 'state') else "COMPLETE",
                    "values": t.values if hasattr(t, 'values') and t.values else None,
                    "params": t.params if hasattr(t, 'params') else {},
                    "user_attrs": t.user_attrs if hasattr(t, 'user_attrs') else {},
                    "is_best": t.number == best_trial_number
                })
            
            # Add best trial if not already in last 5
            if best_trial and best_trial_number not in [t.number for t in last_5_trials]:
                trials_to_include.insert(0, {
                    "number": best_trial.number,
                    "state": best_trial.state.name if hasattr(best_trial, 'state') else "COMPLETE",
                    "values": best_trial.values if hasattr(best_trial, 'values') and best_trial.values else None,
                    "params": best_trial.params if hasattr(best_trial, 'params') else {},
                    "user_attrs": best_trial.user_attrs if hasattr(best_trial, 'user_attrs') else {},
                    "is_best": True
                })
            
            study_snapshot = {
                "study_name": study.study_name,
                "timestamp": pd.Timestamp.now().isoformat(),
                "summary": {
                    "total_trials": len(study.trials),
                    "completed_trials": len([t for t in study.trials if hasattr(t, 'state') and t.state == optuna.trial.TrialState.COMPLETE]),
                    "pruned_trials": len([t for t in study.trials if hasattr(t, 'state') and t.state == optuna.trial.TrialState.PRUNED]),
                    "failed_trials": len([t for t in study.trials if hasattr(t, 'state') and t.state == optuna.trial.TrialState.FAIL]),
                    "best_value": best_trial.value if best_trial and hasattr(best_trial, 'value') else None
                },
                "recent_trials": trials_to_include,
                "note": f"Showing last {len(last_5_trials)} trials + best trial (if different)"
            }
            
            snapshot_path = outputs_dir / f"study_snapshot_{trial.number:04d}.json"
            
            # Upload study snapshots to R2 (every 5 trials)
            _write_json(
                snapshot_path,
                study_snapshot,
                upload_to_r2=True,
                study_name=f"stageA_snapshot_{trial.number:04d}",
                stage="stageA"
            )
            
            logger.info(f"☁️ Study snapshot uploaded to R2: trial {trial.number}")
        
    except Exception as e:
        logger.error(f"❌ Error saving trial progress: {e}")


def _ensure_optuna_available() -> None:
    """Ensure Optuna is available before running study orchestration."""
    if optuna is None:
        raise ModuleNotFoundError(
            "Optuna is required for study orchestration. Install optuna to "
            "use Study A/B workflows."
        )


def _instantiate_named_sampler(name: str, seed: Optional[int]) -> optuna.samplers.BaseSampler:
    """Instantiate an Optuna sampler by name."""
    sampler_name = (name or "").lower()
    kwargs: Dict[str, Any] = {}
    if seed is not None:
        kwargs["seed"] = int(seed)

    if sampler_name in {"tpe", "tp"}:
        return optuna.samplers.TPESampler(**kwargs)
    if sampler_name in {"random", "rand"}:
        return optuna.samplers.RandomSampler(**kwargs)
    if sampler_name in {"cmaes", "cma", "cma-es"}:
        return optuna.samplers.CmaEsSampler(seed=kwargs.get("seed"))

    raise ValueError(f"Unsupported Optuna sampler: '{name}'")


def _build_sampler(
    optuna_cfg: DictConfig,
    *,
    fixed_params: Optional[Dict[str, Any]] = None,
    overrides_cfg: Optional[DictConfig] = None,
) -> optuna.samplers.BaseSampler:
    """Construct the Optuna sampler defined in the configuration."""
    sampler_name = str(getattr(optuna_cfg, "sampler", "tpe"))
    seed = getattr(optuna_cfg, "seed", None)

    if sampler_name.lower() == "partial_fixed":
        base_sampler_name = getattr(optuna_cfg, "base_sampler", "tpe")
        base_sampler = _instantiate_named_sampler(base_sampler_name, seed)
        overrides: Dict[str, Any] = {}
        if overrides_cfg is not None and "partial_fixed" in overrides_cfg:
            overrides_raw = OmegaConf.to_container(overrides_cfg.partial_fixed, resolve=True) or {}
            overrides.update({k: v for k, v in overrides_raw.items() if v is not None})
        return optuna.samplers.PartialFixedSampler(
            fixed_params=fixed_params or {},
            base_sampler=base_sampler,
            **overrides,
        )

    return _instantiate_named_sampler(sampler_name, seed)


def _build_pruner(optuna_cfg: DictConfig) -> Optional[optuna.pruners.BasePruner]:
    """Construct the Optuna pruner defined in the configuration."""
    pruner_name = getattr(optuna_cfg, "pruner", None)
    if not pruner_name:
        return None

    name = str(pruner_name).lower()
    if name == "median":
        warmup = getattr(optuna_cfg, "median_warmup_steps", None)
        params = {"n_warmup_steps": int(warmup)} if warmup is not None else {}
        return optuna.pruners.MedianPruner(**params)
    if name == "hyperband":
        max_res = getattr(optuna_cfg, "hyperband_max_resource", None)
        min_res = getattr(optuna_cfg, "hyperband_min_resource", None)
        params: Dict[str, Any] = {}
        if max_res is not None:
            params["max_resource"] = int(max_res)
        if min_res is not None:
            params["min_resource"] = int(min_res)
        return optuna.pruners.HyperbandPruner(**params)

    if name in {"none", "null"}:
        return None

    raise ValueError(f"Unsupported Optuna pruner: '{pruner_name}'")


def _write_json(path: Path, payload: Any, upload_to_r2: bool = True, study_name: str = None, stage: str = None) -> None:
    """
    Write JSON payload to disk with pretty formatting and optionally upload to R2.
    
    Args:
        path: Local file path to write
        payload: JSON-serializable data to write
        upload_to_r2: Whether to upload to Cloudflare R2 storage
        study_name: Name of the study for R2 organization
        stage: Stage identifier (stageA, stageB) for R2 organization
    """
    logger = get_logger(__name__)
    
    # Write locally first
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        # Import the safe JSON converter
        from orchestration.objectives import _safe_json_convert
        json.dump(_safe_json_convert(payload), fp, indent=4, sort_keys=True)
    
    logger.info(f"📝 Saved JSON locally: {path}")
    
    # Upload to R2 if requested
    if upload_to_r2 and study_name and stage:
        try:
            # Get unified config for R2 settings
            config = get_unified_config()
            
            # Initialize R2 uploader (uses unified config internally)
            uploader = R2ModelUploader()
            
            # Extract symbol from config or use default
            symbol = 'EURUSD'  # Default symbol for now
            
            # Upload study results
            success = uploader.upload_study_results(
                local_json_path=str(path),
                study_name=study_name,
                stage=stage,
                symbol=symbol,
                cleanup_local=False  # Keep local copy
            )
            
            if success:
                logger.info(f"☁️ Successfully uploaded {study_name} results to R2")
            else:
                logger.warning(f"⚠️ Failed to upload {study_name} results to R2")
                
        except Exception as e:
            logger.error(f"❌ Error uploading to R2: {e}")
            # Don't fail the entire process if R2 upload fails
    elif upload_to_r2:
        logger.warning("⚠️ R2 upload requested but missing study_name or stage parameters")


def _get_n_trials(optuna_cfg: DictConfig) -> Optional[int]:
    value = getattr(optuna_cfg, "n_trials", None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_timeout(optuna_cfg: DictConfig) -> Optional[float]:
    value = getattr(optuna_cfg, "timeout", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _run_study_preprocess(cfg: DictConfig) -> None:
    """Execute Study A (preprocess_selection)."""
    _ensure_optuna_available()
    study_cfg = cfg.study
    outputs = _prepare_outputs(getattr(study_cfg, "outputs", None))
    storage_url = _prepare_storage(study_cfg, outputs)
    sampler = _build_sampler(
        study_cfg.optuna,
        overrides_cfg=getattr(study_cfg, "sampler_overrides", None),
    )
    pruner = _build_pruner(study_cfg.optuna)
    load_if_exists = True
    if getattr(study_cfg, "storage", None) is not None:
        storage_dict = OmegaConf.to_container(study_cfg.storage, resolve=True) or {}
        load_if_exists = not bool(storage_dict.get("clean_on_start", False))

    # Support single- or multi-objective based on config
    directions = getattr(study_cfg.optuna, "directions", None)
    if directions is not None:
        directions = [str(d) for d in directions]
        logger.info(
            "Starting Study A (mode=%s) | directions=%s | storage=%s",
            study_cfg.mode,
            directions,
            storage_url or "in-memory",
        )
        study = optuna.create_study(
            study_name=str(study_cfg.name),
            directions=directions,
            sampler=sampler,
            pruner=pruner,
            storage=storage_url,
            load_if_exists=load_if_exists,
        )
        is_multi_objective = True
    else:
        direction = str(study_cfg.optuna.direction)
        logger.info(
            "Starting Study A (mode=%s) | direction=%s | storage=%s",
            study_cfg.mode,
            direction,
            storage_url or "in-memory",
        )
        study = optuna.create_study(
            study_name=str(study_cfg.name),
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            storage=storage_url,
            load_if_exists=load_if_exists,
        )
        is_multi_objective = False

    n_trials = _get_n_trials(study_cfg.optuna)
    timeout = _get_timeout(study_cfg.optuna)
    logger.info("Study A configuration | n_trials=%s | timeout=%s", n_trials, timeout)
    
    # Log study status after creation/loading
    total_trials = len(study.trials)
    completed_trials = len([t for t in study.trials if hasattr(t, 'state') and t.state == optuna.trial.TrialState.COMPLETE])
    pruned_trials = len([t for t in study.trials if hasattr(t, 'state') and t.state == optuna.trial.TrialState.PRUNED])
    failed_trials = len([t for t in study.trials if hasattr(t, 'state') and t.state == optuna.trial.TrialState.FAIL])
    
    logger.info(f"📊 Study A Status: {total_trials} total trials ({completed_trials} completed, {pruned_trials} pruned, {failed_trials} failed)")
    logger.info(f"🎯 Remaining trials to complete: {n_trials - completed_trials}")

    # Use Dask cluster for Optuna parallelization
    logger.info("Running Optuna with Dask cluster for parallel trials")
    
    # Create trial callback for progress tracking and R2 upload
    trial_callback = _create_trial_callback(cfg, outputs)
    
    # Check if parallelization is enabled
    use_parallelization = getattr(study_cfg.optuna, 'use_dask_parallelization', True)
    max_concurrent_trials = getattr(study_cfg.optuna, 'max_concurrent_trials', 8)
    
    # Get the Dask client from the global cluster manager if available
    if use_parallelization:
        try:
            from dask.distributed import get_client
            dask_client = get_client()
            worker_count = len(dask_client.scheduler_info()['workers'])
            logger.info(f"Using existing Dask client with {worker_count} workers")
            
            # Nested wrapper ships with task, and reloads objectives on the worker to avoid stale code
            cfg_container = OmegaConf.to_container(cfg, resolve=True)
            def _objective_remote(trial):  # type: ignore
                import importlib
                from omegaconf import OmegaConf as _OC
                import orchestration.objectives as _obj
                try:
                    _obj = importlib.reload(_obj)
                except Exception:
                    pass
                cfg_built = _OC.create(cfg_container)
                return _obj.objective_study_a(trial, cfg_built)
            
            # Submit trials to Dask cluster in controlled batches
            # Use configured max_concurrent_trials or number of workers, whichever is smaller
            effective_max_concurrent = min(worker_count, max_concurrent_trials) if max_concurrent_trials > 0 else worker_count
            logger.info(f"Submitting trials in batches of {effective_max_concurrent} (workers: {worker_count}, max_concurrent: {max_concurrent_trials})")
            
            # Count existing completed trials in the study
            existing_completed_trials = len([t for t in study.trials if hasattr(t, 'state') and t.state == optuna.trial.TrialState.COMPLETE])
            logger.info(f"Found {existing_completed_trials} existing completed trials in study")
            
            completed_trials = existing_completed_trials
            active_futures = []
            
            while completed_trials < n_trials:
                # Submit new trials up to the limit
                while len(active_futures) < effective_max_concurrent and completed_trials + len(active_futures) < n_trials:
                    trial = study.ask()
                    future = dask_client.submit(_objective_remote, trial)
                    active_futures.append((trial, future))
                    logger.info(f"Submitted trial {trial.number} (active: {len(active_futures)})")
                
                # Wait for at least one trial to complete
                if active_futures:
                    # Use as_completed to get results as they finish
                    from dask.distributed import as_completed
                    for future in as_completed([f for _, f in active_futures]):
                        # Find the trial for this future
                        trial_for_future = None
                        for t, f in active_futures:
                            if f == future:
                                trial_for_future = t
                                break
                        
                        if trial_for_future is not None:
                            try:
                                result = future.result()
                                study.tell(trial_for_future, result)
                                # Execute callback for this trial
                                trial_callback(study, trial_for_future)
                                completed_trials += 1
                                logger.info(f"Completed trial {trial_for_future.number} ({completed_trials}/{n_trials})")
                            except Exception as e:
                                logger.error(f"Trial {trial_for_future.number} failed: {e}")
                                study.tell(trial_for_future, state=optuna.trial.TrialState.FAIL)
                                completed_trials += 1
                            
                            # Remove completed trial from active list
                            active_futures = [(t, f) for t, f in active_futures if f != future]
                
        except Exception as e:
            logger.warning(f"Could not use Dask client, falling back to sequential: {e}")
            study.optimize(
                lambda trial: objective_study_a(trial, cfg), 
                n_trials=n_trials, 
                timeout=timeout,
                callbacks=[trial_callback]
            )
    else:
        logger.info("Dask parallelization disabled, using sequential execution")
        study.optimize(
            lambda trial: objective_study_a(trial, cfg), 
            n_trials=n_trials, 
            timeout=timeout,
            callbacks=[trial_callback]
        )

    # Determine best params depending on single vs multi-objective
    best_params = None
    if is_multi_objective:
        # Prefer Optuna's best_trials for multi-objective, fallback to Pareto front
        pareto_trials = []
        try:
            pareto_trials = list(getattr(study, 'best_trials', []) or [])
        except Exception:
            pass
        if not pareto_trials:
            try:
                pareto_trials = list(study.get_pareto_front_trials())  # type: ignore[attr-defined]
            except Exception:
                pareto_trials = []
        if pareto_trials:
            # Choose the Pareto-front trial with max Sharpe (assumes objective order: Sharpe, Turnover, MaxDD)
            try:
                best_t = max(pareto_trials, key=lambda t: (t.values or [float('-inf')])[0])
                best_params = best_t.params
            except Exception as e:
                logger.warning(f"Failed to select best trial from Pareto front: {e}")
        else:
            logger.warning("No Pareto-front trials found; skipping best params export.")
    else:
        # Guard: when no trials completed successfully, Optuna backend can raise on best_trial
        try:
            best_params = study.best_trial.params  # type: ignore[attr-defined]
        except Exception:
            logger.warning("Study A completed without successful trials; skipping best params export.")

    best_params_path = outputs.get("best_stage12_path")
    if best_params_path and best_params:
        _write_json(
            best_params_path,
            best_params,
            upload_to_r2=True,
            study_name="stageA_preprocess_selection",
            stage="stageA"
        )
        logger.info("Best preprocessing parameters saved to %s", best_params_path)

        # Execute full pipeline with best parameters to generate final model
        logger.info("Executing full pipeline with optimized parameters...")
        _execute_full_pipeline_with_best_params(cfg, best_params)
    elif best_params_path and not best_params:
        logger.warning("Best params path configured but no best params selected; results not saved.")
    else:
        logger.warning("No output path configured for best_stage12; results not saved.")


def _run_study_modeling(cfg: DictConfig) -> None:
    """Execute Study B (modeling)."""
    _ensure_optuna_available()
    study_cfg = cfg.study
    outputs = _prepare_outputs(getattr(study_cfg, "outputs", None))
    fixed_cfg = getattr(study_cfg, "fixed_params", None)
    if fixed_cfg is None or not getattr(fixed_cfg, "load_from", None):
        raise ValueError("Study B requires 'study.fixed_params.load_from' to be defined in the config.")

    fixed_params_path = _as_abs_path(fixed_cfg.load_from)
    if not fixed_params_path.exists():
        raise FileNotFoundError(
            f"Fixed-parameter handoff file not found: {fixed_params_path}. "
            "Run Study A first or adjust 'study.fixed_params.load_from'."
        )

    with fixed_params_path.open("r", encoding="utf-8") as fp:
        fixed_params = json.load(fp)

    storage_url = _prepare_storage(study_cfg, outputs)
    sampler = _build_sampler(
        study_cfg.optuna,
        fixed_params=fixed_params,
        overrides_cfg=getattr(study_cfg, "sampler_overrides", None),
    )
    pruner = _build_pruner(study_cfg.optuna)
    load_if_exists = True
    if getattr(study_cfg, "storage", None) is not None:
        storage_dict = OmegaConf.to_container(study_cfg.storage, resolve=True) or {}
        load_if_exists = not bool(storage_dict.get("clean_on_start", False))

    directions = [str(direction) for direction in study_cfg.optuna.directions]
    logger.info(
        "Starting Study B (mode=%s) | directions=%s | storage=%s",
        study_cfg.mode,
        directions,
        storage_url or "in-memory",
    )

    study = optuna.create_study(
        study_name=str(study_cfg.name),
        directions=directions,
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,
        load_if_exists=load_if_exists,
    )

    n_trials = _get_n_trials(study_cfg.optuna)
    timeout = _get_timeout(study_cfg.optuna)
    logger.info("Study B configuration | n_trials=%s | timeout=%s", n_trials, timeout)
    
    # Log study status after creation/loading
    total_trials = len(study.trials)
    completed_trials = len([t for t in study.trials if hasattr(t, 'state') and t.state == optuna.trial.TrialState.COMPLETE])
    pruned_trials = len([t for t in study.trials if hasattr(t, 'state') and t.state == optuna.trial.TrialState.PRUNED])
    failed_trials = len([t for t in study.trials if hasattr(t, 'state') and t.state == optuna.trial.TrialState.FAIL])
    
    logger.info(f"📊 Study B Status: {total_trials} total trials ({completed_trials} completed, {pruned_trials} pruned, {failed_trials} failed)")
    logger.info(f"🎯 Remaining trials to complete: {n_trials - completed_trials}")

    def _objective(trial: optuna.trial.Trial):
        return objective_study_b(trial, cfg, fixed_params)

    # Use Dask cluster for Optuna parallelization
    logger.info("Running Study B Optuna with Dask cluster for parallel trials")
    
    # Get the Dask client from the global cluster manager if available
    try:
        from dask.distributed import get_client
        dask_client = get_client()
        worker_count = len(dask_client.scheduler_info()['workers'])
        logger.info(f"Using existing Dask client with {worker_count} workers for Study B")
        
        # Submit trials to Dask cluster in controlled batches
        effective_max_concurrent = min(worker_count, 8)  # Cap at 8 concurrent trials
        logger.info(f"Study B: Submitting trials in batches of {effective_max_concurrent} (workers: {worker_count})")
        
        # Count existing completed trials in the study
        existing_completed_trials = len([t for t in study.trials if hasattr(t, 'state') and t.state == optuna.trial.TrialState.COMPLETE])
        logger.info(f"Study B: Found {existing_completed_trials} existing completed trials in study")
        
        completed_trials = existing_completed_trials
        active_futures = []
        
        while completed_trials < n_trials:
            # Submit new trials up to the limit
            while len(active_futures) < effective_max_concurrent and completed_trials + len(active_futures) < n_trials:
                trial = study.ask()
                future = dask_client.submit(_objective, trial)
                active_futures.append((trial, future))
                logger.info(f"Study B: Submitted trial {trial.number} (active: {len(active_futures)})")
            
            # Wait for at least one trial to complete
            if active_futures:
                from dask.distributed import as_completed
                for future in as_completed([f for _, f in active_futures]):
                    # Find the trial for this future
                    trial_for_future = None
                    for t, f in active_futures:
                        if f == future:
                            trial_for_future = t
                            break
                    
                    if trial_for_future is not None:
                        try:
                            result = future.result()
                            study.tell(trial_for_future, result)
                            completed_trials += 1
                            logger.info(f"Study B: Completed trial {trial_for_future.number} ({completed_trials}/{n_trials})")
                        except Exception as e:
                            logger.error(f"Study B Trial {trial_for_future.number} failed: {e}")
                            study.tell(trial_for_future, state=optuna.trial.TrialState.FAIL)
                            completed_trials += 1
                        
                        # Remove completed trial from active list
                        active_futures = [(t, f) for t, f in active_futures if f != future]
                
    except Exception as e:
        logger.warning(f"Could not use Dask client for Study B, falling back to sequential: {e}")
        study.optimize(_objective, n_trials=n_trials, timeout=timeout)

    pareto_path = outputs.get("pareto_path")
    if pareto_path and hasattr(study, 'best_trials') and study.best_trials:
        pareto_payload = [
            {
                "number": trial.number,
                "values": trial.values,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
            }
            for trial in study.best_trials
        ]
        _write_json(
            pareto_path, 
            pareto_payload,
            upload_to_r2=True,
            study_name="stageB_modeling_optimization", 
            stage="stageB"
        )
        logger.info("Saved Study B Pareto front to %s", pareto_path)


def _hydra_args_present(argv: Sequence[str]) -> bool:
    """Check CLI arguments for Hydra-style overrides, regardless of availability."""
    hydra_markers = {"-m", "--multirun", "--config-name", "--config-path"}
    for arg in argv[1:]:
        cleaned = arg.lstrip('+')
        if cleaned.startswith("study") or cleaned.startswith("hydra."):
            return True
        if arg in hydra_markers:
            return True
    return False


def _should_run_hydra(argv: Sequence[str]) -> bool:
    """Detect whether CLI arguments indicate Hydra-based execution."""
    if hydra is None:
        return False
    return _hydra_args_present(argv)


if hydra is not None:

    @hydra.main(config_path="../config", config_name="config", version_base=None)
    def study_main(cfg: DictConfig) -> None:
        """Hydra entrypoint for Study A/B orchestration."""
        logger.info("Hydra study orchestration started (mode=%s)", cfg.study.mode)

        mode = str(cfg.study.mode)
        if mode == "preprocess_selection":
            # Ensure a Dask cluster is available for Optuna parallelization
            try:
                with managed_dask_cluster():
                    _run_study_preprocess(cfg)
            except Exception as e:
                logger.warning(f"Could not start Dask cluster for Study A (fallback to sequential): {e}")
                _run_study_preprocess(cfg)
        elif mode == "modeling":
            # Ensure a Dask cluster is available for Optuna parallelization
            try:
                with managed_dask_cluster():
                    _run_study_modeling(cfg)
            except Exception as e:
                logger.warning(f"Could not start Dask cluster for Study B (fallback to sequential): {e}")
                _run_study_modeling(cfg)
        else:
            raise ValueError(f"Unknown study mode: {mode}")

else:

    def study_main(*_args: Any, **_kwargs: Any) -> None:  # type: ignore
        raise ModuleNotFoundError(
            "Hydra is required for study orchestration. Install hydra-core to "
            "run Study A/B workflows."
        )


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    hydra_requested = _hydra_args_present(sys.argv)
    
    # Se não há argumentos Hydra explícitos, verificar se a configuração tem study nos defaults
    if not hydra_requested and hydra is not None:
        try:
            # Verificar se config.yaml tem study nos defaults
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_content = f.read()
                    if 'study: stageA' in config_content or 'study: stageB' in config_content:
                        hydra_requested = True
                        logger.info("Detected study configuration in defaults, using Hydra mode")
        except Exception as e:
            logger.debug(f"Could not check config for study defaults: {e}")
    
    if hydra_requested and hydra is None:
        logger.error(
            "Hydra was requested via CLI overrides, but hydra-core is not installed. "
            "Install hydra-core to run Study A/B workflows."
        )
        sys.exit(1)
    if hydra_requested:
        study_main()
    else:
        sys.exit(run_pipeline())
