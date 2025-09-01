"""
Main Orchestration Script for Dynamic Stage 0 Pipeline

This script manages the Dask-CUDA cluster lifecycle and provides the foundation
for the GPU-accelerated feature engineering pipeline using the new modular architecture.
"""

import logging
import sys
import os
import signal
import time
import threading
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
from orchestration.data_processor import process_currency_pair_worker
from features.base_engine import CriticalPipelineError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global flag for emergency shutdown
EMERGENCY_SHUTDOWN = threading.Event()


def emergency_shutdown_handler(signum, frame):
    """Handle emergency shutdown signals."""
    logger.critical("🚨 EMERGENCY SHUTDOWN SIGNAL RECEIVED")
    logger.critical("🛑 Initiating immediate shutdown of all processes...")
    EMERGENCY_SHUTDOWN.set()
    sys.exit(1)


# Register signal handlers for emergency shutdown
signal.signal(signal.SIGINT, emergency_shutdown_handler)
signal.signal(signal.SIGTERM, emergency_shutdown_handler)


class DaskClusterManager:
    """Manages the Dask-CUDA cluster lifecycle for Dynamic Stage 0 pipeline."""

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
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown()
        sys.exit(0)

    def _get_gpu_count(self) -> int:
        """Get the number of available GPUs."""
        try:
            return cp.cuda.runtime.getDeviceCount()
        except Exception as e:
            logger.warning(f"Could not detect GPU count: {e}")
            return 1

    def _configure_rmm(self):
        """Configure RMM (RAPIDS Memory Manager) for optimal memory management."""
        try:
            from rmm import reinitialize
            
            # Use configuration from unified config
            initial_pool_size = int(self.config.dask.rmm_initial_pool_size.replace('GB', '')) * 1024**3
            
            reinitialize(
                pool_allocator=True,
                initial_pool_size=initial_pool_size,
                managed_memory=False
            )
            
            logger.info(f"✅ RMM configured with {initial_pool_size / (1024**3):.1f}GB pool")
            
        except ImportError:
            logger.warning("⚠️ RMM not available, using default CUDA memory management")
            os.environ.setdefault("RMM_ALLOCATOR", "cuda_malloc")
        except Exception as e:
            logger.error(f"Failed to configure RMM: {e}")
            os.environ.setdefault("RMM_ALLOCATOR", "cuda_malloc")

    def start_cluster(self) -> bool:
        """
        Start the Dask-CUDA cluster with proper RMM configuration.
        Returns:
            bool: True if cluster started successfully, False otherwise
        """
        try:
            logger.info("Starting Dask-CUDA cluster...")
            gpu_count = self._get_gpu_count()
            logger.info(f"Detected {gpu_count} GPU(s)")

            # Configure RMM before creating cluster
            self._configure_rmm()

            # Configure worker memory via dask config to avoid deprecation warnings
            try:
                import dask
                dask.config.set({
                    'distributed.worker.memory.target': float(self.config.dask.memory_target_fraction),
                    'distributed.worker.memory.spill': float(self.config.dask.memory_spill_fraction),
                })
                logger.info("Dask memory config set (target=%.2f, spill=%.2f)",
                            float(self.config.dask.memory_target_fraction),
                            float(self.config.dask.memory_spill_fraction))
            except Exception as e:
                logger.warning(f"Could not set Dask memory config: {e}")

            # Build cluster configuration from unified config (omit deprecated kwargs)
            cluster_kwargs = {
                'n_workers': gpu_count,
                'threads_per_worker': self.config.dask.threads_per_worker,
                'rmm_pool_size': self.config.dask.rmm_pool_size,
                'local_directory': self.config.dask.local_directory,
            }

            # Add UCX configuration if enabled
            if self.config.dask.protocol == "ucx":
                cluster_kwargs.update({
                    'protocol': "ucx",
                    'enable_tcp_over_ucx': self.config.dask.enable_tcp_over_ucx,
                    'enable_infiniband': self.config.dask.enable_infiniband,
                    'enable_nvlink': self.config.dask.enable_nvlink,
                })

            try:
                logger.info("Creating LocalCUDACluster with UCX...")
                self.cluster = LocalCUDACluster(**cluster_kwargs)
            except Exception as ucx_err:
                logger.warning(f"UCX unavailable ({ucx_err}); falling back to TCP.")
                # Remove UCX-specific parameters for TCP fallback
                for key in ['protocol', 'enable_tcp_over_ucx', 'enable_infiniband', 'enable_nvlink']:
                    cluster_kwargs.pop(key, None)
                self.cluster = LocalCUDACluster(**cluster_kwargs)

            logger.info("✓ Cluster created successfully")
            self.client = Client(self.cluster)
            logger.info("✓ Client created successfully")

            logger.info("Waiting for workers to be ready...")
            self.client.wait_for_workers(gpu_count, timeout=300)
            logger.info(f"✓ {gpu_count} workers ready")
            logger.info(f"✓ Dashboard URL: {self.client.dashboard_link}")
            logger.info(f"✓ Active workers: {len(self.client.scheduler_info()['workers'])}")
            
            # Set up worker death monitoring
            self._setup_worker_monitoring()
            
            return True

        except Exception as e:
            logger.error(f"Failed to start Dask-CUDA cluster: {e}", exc_info=True)
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
            # Get initial worker count
            self.initial_worker_count = len(self.client.scheduler_info()["workers"])
            logger.info(f"Monitoring {self.initial_worker_count} workers for failures")
            
            # Set up periodic check
            def monitor_workers():
                while True:
                    try:
                        current_workers = len(self.client.scheduler_info()["workers"])
                        if current_workers < self.initial_worker_count:
                            logger.critical("🚨 WORKER DEATH DETECTED!")
                            logger.critical(f"Workers: {current_workers}/{self.initial_worker_count}")
                            logger.critical("🛑 STOPPING PIPELINE IMMEDIATELY")
                            EMERGENCY_SHUTDOWN.set()
                            break
                        time.sleep(5)  # Check every 5 seconds
                    except Exception as e:
                        logger.error(f"Error in worker monitoring: {e}")
                        break
            
            # Start monitoring in background thread
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
    """Run the complete Dynamic Stage 0 pipeline using the new modular architecture."""
    logger.info("=" * 60)
    logger.info("Dynamic Stage 0 - Pipeline Execution (Modular Architecture)")
    logger.info("=" * 60)

    try:
        # Get unified configuration
        config = get_unified_config()
        
        # Initialize pipeline orchestrator
        orchestrator = PipelineOrchestrator()
        
        # Connect to database (non-fatal; continue without DB if unavailable)
        if not orchestrator.connect_database():
            logger.warning("Database unavailable; continuing without task tracking.")

        # Discover tasks
        pending_tasks = orchestrator.discover_tasks()
        if not pending_tasks:
            logger.info("No tasks to process. Pipeline complete.")
            return 0

        logger.info(f"Processing {len(pending_tasks)} currency pairs that need feature engineering")

        # Execute pipeline with cluster management
        with managed_dask_cluster() as cluster_manager:
            if not cluster_manager.is_active():
                logger.error("Cluster manager is not active")
                raise RuntimeError("Failed to start or activate Dask cluster.")
            
            client = cluster_manager.get_client()
            if not client:
                logger.error("Failed to get Dask client")
                raise RuntimeError("Failed to get Dask client.")
            
            # Execute the pipeline
            result = orchestrator.execute_pipeline(cluster_manager, client)
            
            # Log pipeline summary
            orchestrator.log_pipeline_summary(result, len(pending_tasks))
            
            # Clean up
            orchestrator.cleanup()
            
            # Check if emergency shutdown was triggered
            if result.emergency_shutdown:
                logger.critical("🚨 PIPELINE STOPPED DUE TO EMERGENCY SHUTDOWN")
                return 1
            
            return 0 if result.failed_tasks == 0 else 1

    except Exception as e:
        logger.error(f"FATAL PIPELINE ERROR: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(run_pipeline())
