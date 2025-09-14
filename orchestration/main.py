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
from monitoring.dask_plugins import PipelineWorkerPlugin
from orchestration.pipeline_orchestrator import PipelineOrchestrator
from features.base_engine import CriticalPipelineError
from utils.logging_utils import (
    get_logger,
)

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
                    f"pool={safe_pool_gb:.2f}GB, cap={cap_gb:.2f}GB, total={total_gb:.2f}GB"
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
            
            cluster_kwargs = {
                'n_workers': gpu_count,
                'threads_per_worker': self.config.dask.threads_per_worker,
                'memory_limit': memory_limit_str,
                'rmm_pool_size': getattr(self, '_safe_rmm_pool_size_str', self.config.dask.rmm_pool_size),
                'local_directory': self.config.dask.local_directory,
                'dashboard_address': dashboard_address,
                'scheduler_port': scheduler_port,
            }

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


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(run_pipeline())
