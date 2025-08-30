"""
Main Orchestration Script for Dynamic Stage 0 Pipeline

This script manages the Dask-CUDA cluster lifecycle and provides the foundation
for the GPU-accelerated feature stationarization pipeline.
"""

import logging
import sys
import os
import signal
import time
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client
    import cupy as cp
    import cudf
    import dask_cudf
except ImportError as e:
    print(f"Error importing Dask-CUDA libraries: {e}")
    print("Make sure the GPU environment is properly set up.")
    sys.exit(1)

from config import get_config
from config.settings import get_settings
from data_io.db_handler import DatabaseHandler, get_pending_currency_pairs, update_task_status
# Alterado para carregar do loader local
from data_io.local_loader import LocalDataLoader


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DaskClusterManager:
    """Manages the Dask-CUDA cluster lifecycle for Dynamic Stage 0 pipeline."""

    def __init__(self):
        """Initialize the cluster manager with configuration."""
        self.config = get_config()
        self.settings = get_settings()
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

    def start_cluster(self) -> bool:
        """
        Start the Dask-CUDA cluster.
        Returns:
            bool: True if cluster started successfully, False otherwise
        """
        try:
            logger.info("Starting Dask-CUDA cluster...")
            gpu_count = self._get_gpu_count()
            logger.info(f"Detected {gpu_count} GPU(s)")

            # Simplificado: Deixa o LocalCUDACluster gerenciar o RMM com base nos parâmetros
            self.cluster = LocalCUDACluster(
                n_workers=gpu_count,
                rmm_pool_size=self.settings.dask.rmm_pool_size,
                protocol='tcp',
                dashboard_address=':8787',
                silence_logs=logging.WARNING
            )

            logger.info(f"Cluster created with {gpu_count} workers")
            self.client = Client(self.cluster)
            logger.info("Waiting for cluster to be ready...")
            self.client.wait_for_workers(gpu_count, timeout=60)
            logger.info("Dask-CUDA cluster is active and ready!")
            logger.info(f"Dashboard URL: {self.client.dashboard_link}")
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


@contextmanager
def managed_dask_cluster():
    """Context manager for Dask-CUDA cluster lifecycle."""
    cluster_manager = DaskClusterManager()
    try:
        yield cluster_manager
    finally:
        cluster_manager.shutdown()


def process_currency_pair(
    task: Dict[str, Any],
    client: Client,
    db_handler: DatabaseHandler,
    local_loader: LocalDataLoader
) -> bool:
    """
    Process a single currency pair through the pipeline.
    """
    task_id = task['task_id']
    currency_pair = task['currency_pair']
    # O 'r2_path' agora é o caminho relativo que será usado para encontrar os dados locais
    relative_data_path = task['r2_path']

    try:
        logger.info(f"Starting processing for {currency_pair} (Task ID: {task_id})")
        if not db_handler.update_task_status(task_id, 'RUNNING'):
            logger.error(f"Failed to update status to RUNNING for {currency_pair}")
            return False

        logger.info(f"Loading data for {currency_pair} from local path derived from: {relative_data_path}")
        df = local_loader.load_currency_pair_data(relative_data_path, client)

        if df is None:
            logger.error(f"Failed to load data for {currency_pair}")
            db_handler.update_task_status(task_id, 'FAILED', f"Failed to load data from local path: {relative_data_path}")
            return False

        logger.info(f"Verifying data loading for {currency_pair}...")
        data_length = len(df)
        logger.info(f"Total rows for {currency_pair}: {data_length}")

        # Placeholder para a lógica da Tarefa 7 em diante
        # features_df = calculate_native_features(df)
        # ... etc

        if not db_handler.update_task_status(task_id, 'COMPLETED'):
            logger.error(f"Failed to update status to COMPLETED for {currency_pair}")
            return False

        logger.info(f"Successfully completed processing for {currency_pair}")
        return True

    except Exception as e:
        logger.error(f"Error processing {currency_pair}: {e}", exc_info=True)
        db_handler.update_task_status(task_id, 'FAILED', f"Processing error: {str(e)}")
        return False


def run_pipeline():
    """Run the complete Dynamic Stage 0 pipeline."""
    logger.info("=" * 60)
    logger.info("Dynamic Stage 0 - Pipeline Execution")
    logger.info("=" * 60)

    try:
        settings = get_settings()
        db_handler = DatabaseHandler()
        local_loader = LocalDataLoader()

        pending_tasks = get_pending_currency_pairs()
        if not pending_tasks:
            logger.info("No pending tasks found. Pipeline complete.")
            return 0

        with managed_dask_cluster() as cluster_manager:
            if not cluster_manager.is_active():
                 raise RuntimeError("Failed to start or activate Dask cluster.")
            client = cluster_manager.get_client()

            successful_tasks, failed_tasks = 0, 0
            for task in pending_tasks:
                success = process_currency_pair(
                    task=task,
                    client=client,
                    db_handler=db_handler,
                    local_loader=local_loader
                )
                if success:
                    successful_tasks += 1
                else:
                    failed_tasks += 1

            logger.info("=" * 60)
            logger.info("PIPELINE EXECUTION SUMMARY")
            logger.info(f"Total tasks processed: {len(pending_tasks)}")
            logger.info(f"Successful tasks: {successful_tasks}")
            logger.info(f"Failed tasks: {failed_tasks}")
            logger.info("=" * 60)

        return 0 if failed_tasks == 0 else 1

    except Exception as e:
        logger.error(f"FATAL PIPELINE ERROR: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(run_pipeline())
