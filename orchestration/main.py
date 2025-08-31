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
from data_io.db_handler import DatabaseHandler, update_task_status
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
            logger.info("Detected %d GPU(s)", gpu_count)

            # RMM via cudaMallocAsync (pool + menos fragmentação)
            os.environ.setdefault("RMM_ALLOCATOR", "cuda_async")

            # calcula pool ~80% da VRAM da 1ª GPU
            free_b, total_b = cp.cuda.runtime.memGetInfo()
            pool_bytes = int(total_b * 0.80)
            pool_str = f"{pool_bytes // (1024**3)}GB"

            common_kwargs = dict(
                n_workers=gpu_count,
                threads_per_worker=1,          # GPU: 1 thread por worker
                memory_limit="auto",           # deixar Dask decidir
                rmm_pool_size=pool_str,        # evita hardcode "24GB"
                dashboard_address=":8787",
                silence_logs=logging.WARNING,
            )

            try:
                logger.info("Creating LocalCUDACluster with UCX...")
                self.cluster = LocalCUDACluster(protocol="ucx", **common_kwargs)
            except Exception as ucx_err:
                logger.warning("UCX unavailable (%s); falling back to TCP.", ucx_err)
                self.cluster = LocalCUDACluster(protocol="tcp", **common_kwargs)

            logger.info("✓ Cluster created successfully")
            self.client = Client(self.cluster)
            logger.info("✓ Client created successfully")

            logger.info("Waiting for workers to be ready...")
            self.client.wait_for_workers(gpu_count, timeout=120)
            logger.info("✓ %d workers ready", gpu_count)
            logger.info("✓ Dashboard URL: %s", self.client.dashboard_link)
            logger.info("✓ Active workers: %d", len(self.client.scheduler_info()["workers"]))
            return True

        except Exception as e:
            logger.error("Failed to start Dask-CUDA cluster: %s", e, exc_info=True)
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
        if not cluster_manager.start_cluster():
            raise RuntimeError("Failed to start Dask-CUDA cluster")
        yield cluster_manager
    finally:
        cluster_manager.shutdown()


def save_processed_data_cudf(
    gdf: 'cudf.DataFrame',
    currency_pair: str,
    settings,
    task_id: str,
    db_handler: DatabaseHandler
) -> bool:
    """
    Save processed cuDF DataFrame to Feather v2 format.
    """
    try:
        import pyarrow.feather as feather
        import pathlib
        
        out_dir = pathlib.Path(settings.output.output_path) / currency_pair
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{currency_pair}.feather"

        table = gdf.to_arrow()
        feather.write_feather(table, str(out_path), compression="lz4", version=2)
        logger.info("Saved single Feather file: %s (%d rows)", out_path, len(gdf))
        return True
    except Exception as e:
        logger.error("Failed to save cuDF: %s", e, exc_info=True)
        db_handler.update_task_status(task_id, 'FAILED', f"Data saving error: {e}")
        return False


def save_processed_data(
    df: dask_cudf.DataFrame,
    currency_pair: str,
    settings,
    task_id: str,
    db_handler: DatabaseHandler
) -> bool:
    """
    Save processed DataFrame with all features to Feather v2 files (Arrow IPC).
    
    Args:
        df: Processed DataFrame with all features
        currency_pair: Currency pair identifier
        settings: Application settings
        task_id: Task ID for status updates
        db_handler: Database handler for status updates
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        logger.info(f"Saving processed data with {len(df.columns)} columns to Feather v2 files for {currency_pair}")
        
        # Get the output path from settings
        output_path = settings.output.output_path
        compression = "lz4"  # Fast compression for Feather v2
        
        # Create output directory structure
        import pathlib
        import os
        import gc
        import pyarrow.feather as feather
        
        output_dir = pathlib.Path(output_path) / currency_pair
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Log data statistics before saving
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Data types: {df.dtypes.to_dict()}")
        logger.info(f"Number of partitions: {df.npartitions}")
        
        # Save using Feather v2 with partitioning to avoid memory spikes
        logger.info(f"Saving to: {output_dir} using Feather v2 (Arrow IPC) - partitioned approach")
        
        # Ensure global ordering by timestamp if 'ts' column exists
        if 'ts' in df.columns:
            logger.info("Ensuring global ordering by timestamp...")
            df = df.set_index('ts', shuffle='tasks')
        
        # Save each partition as a separate Feather v2 file
        partition_files = []
        for i, part in enumerate(df.to_delayed()):
            try:
                # Compute partition in GPU
                gpart = part.compute()  # cudf.DataFrame da partição (GPU)
                
                # Convert to Arrow (CPU)
                table = gpart.to_arrow()
                
                # Save as Feather v2
                part_filename = f"part-{i:05d}.feather"
                part_path = output_dir / part_filename
                
                feather.write_feather(
                    table,
                    str(part_path),
                    compression=compression,
                    version=2  # garante Feather v2
                )
                
                partition_files.append(part_filename)
                logger.info(f"Saved partition {i+1}/{df.npartitions}: {part_filename} ({len(gpart)} rows)")
                
                # Clean up GPU memory
                del gpart, table
                gc.collect()
                
            except Exception as part_error:
                logger.error(f"Error saving partition {i}: {part_error}")
                raise
        
        # Create a consolidated single file for easier access (optional)
        logger.info("Creating consolidated single Feather v2 file...")
        try:
            # Read all partitions and concatenate in CPU
            import glob
            import pyarrow as pa
            
            # Get all partition files
            part_files = sorted(glob.glob(str(output_dir / "part-*.feather")))
            
            if part_files:
                # Read and concatenate all partitions
                tables = []
                for part_file in part_files:
                    table = feather.read_feather(part_file)
                    tables.append(table)
                
                # Concatenate all tables
                consolidated_table = pa.concat_tables(tables)
                
                # Save consolidated file
                consolidated_path = output_dir / f"{currency_pair}.feather"
                feather.write_feather(
                    consolidated_table,
                    str(consolidated_path),
                    compression=compression,
                    version=2
                )
                
                logger.info(f"Created consolidated file: {consolidated_path}")
                
                # Clean up
                del tables, consolidated_table
                gc.collect()
                
        except Exception as consolidate_error:
            logger.warning(f"Could not create consolidated file: {consolidate_error}")
            # Continue anyway, partitioned files are still available
        
        # Verify the saved data
        saved_files = list(output_dir.glob("*.feather"))
        total_size_mb = sum(f.stat().st_size for f in saved_files) / (1024 * 1024)
        
        logger.info(f"Successfully saved processed data for {currency_pair}")
        logger.info(f"Files created: {len(saved_files)}")
        logger.info(f"Partition files: {len(partition_files)}")
        logger.info(f"Total size: {total_size_mb:.2f} MB")
        
        return True
        
    except Exception as save_error:
        logger.error(f"Failed to save processed data for {currency_pair}: {save_error}", exc_info=True)
        db_handler.update_task_status(task_id, 'FAILED', f"Data saving error: {str(save_error)}")
        return False


def process_currency_pair(
    task: Dict[str, Any],
    db_handler=None,
    local_loader=None
) -> bool:
    """
    Process a single currency pair through the pipeline.
    Caminho A: cuDF por worker (um par por worker)
    """
    # ❌ NÃO passe db_handler/local_loader do driver.
    # ✅ Crie no worker:
    from data_io.db_handler import DatabaseHandler
    from data_io.local_loader import LocalDataLoader

    currency_pair = task['currency_pair']
    relative_data_path = task['r2_path']

    # Register task in database only after successful processing
    db = DatabaseHandler()
    if not db.connect():
        logger.error("DB connect failed in worker")
        return False

    try:
        logger.info("Starting processing for %s", currency_pair)

        loader = LocalDataLoader()

        # ⚠️ carregue como **cuDF** no worker (um par por worker)
        gdf = loader.load_currency_pair_data_feather_sync(relative_data_path)
        if gdf is None:
            gdf = loader.load_currency_pair_data_sync(relative_data_path)
        if gdf is None:
            db.update_task_status(task_id, 'FAILED', f"Failed to load data from {relative_data_path}")
            return False

        logger.info("Data loaded: %d rows, %d cols", len(gdf), len(gdf.columns))

        # ---- feature engines em cuDF (versões compatíveis) ----
        from features import StationarizationEngine, StatisticalTests, SignalProcessor, GARCHModels
        settings = get_settings()

        # Use None for client - the engines should work without Dask client
        station = StationarizationEngine(settings, None)
        stats   = StatisticalTests(settings, None)
        sig     = SignalProcessor(settings, None)
        garch   = GARCHModels(settings, None)

        # todas trabalhando sobre **cuDF**:
        gdf = station.process_currency_pair(gdf)
        gdf = stats.process_cudf(gdf)          # exponha um .process_cudf no módulo
        gdf = sig.process_cudf(gdf)            # idem
        gdf = garch.process_cudf(gdf)          # idem

        logger.info("Final cuDF shape: %s", (len(gdf), len(gdf.columns)))

        # salvar (versão cuDF)
        if not save_processed_data_cudf(gdf, currency_pair, settings, None, db):
            return False

        # Register task in database only after successful processing
        task_id = db.register_currency_pair(currency_pair, relative_data_path)
        if task_id:
            db.update_task_status(task_id, 'COMPLETED')
            logger.info("Successfully completed %s (Task ID: %s)", currency_pair, task_id)
        else:
            logger.warning("Failed to register task in database for %s", currency_pair)
        
        return True
        
        return True

    except Exception as e:
        logger.error("Error processing %s: %s", currency_pair, e, exc_info=True)
        return False
    finally:
        try: db.close()
        except: pass


def run_pipeline():
    """Run the complete Dynamic Stage 0 pipeline."""
    logger.info("=" * 60)
    logger.info("Dynamic Stage 0 - Pipeline Execution")
    logger.info("=" * 60)

    try:
        settings = get_settings()
        db_handler = DatabaseHandler()
        local_loader = LocalDataLoader()

        # Connect to database
        if not db_handler.connect():
            logger.error("Failed to connect to database. Exiting.")
            return 1

        # Discover all available currency pairs in /data
        logger.info("Discovering currency pairs in local data directory...")
        available_pairs = local_loader.discover_currency_pairs()
        
        if not available_pairs:
            logger.info("No currency pairs found in data directory. Pipeline complete.")
            return 0

        logger.info(f"Found {len(available_pairs)} currency pairs in data directory")

        # Filter pairs that need processing
        pending_tasks = []
        for pair_info in available_pairs:
            currency_pair = pair_info['currency_pair']
            
            # Check if output file already exists (idempotent approach)
            output_path = f"{settings.output.output_path}/{currency_pair}/{currency_pair}.feather"
            if os.path.exists(output_path):
                logger.info(f"Skipping {currency_pair} - output file already exists: {output_path}")
                continue
            
            # Create task without registering in database yet
            task = {
                'task_id': None,  # Will be assigned after successful processing
                'currency_pair': currency_pair,
                'r2_path': pair_info['data_path'],
                'file_type': pair_info['file_type'],
                'file_size_mb': pair_info['file_size_mb'],
                'filename': pair_info['filename']
            }
            pending_tasks.append(task)
            logger.info(f"Added {currency_pair} to processing queue")

        if not pending_tasks:
            logger.info("All currency pairs already processed. Pipeline complete.")
            return 0

        logger.info(f"Processing {len(pending_tasks)} currency pairs that need feature engineering")

        with managed_dask_cluster() as cluster_manager:
            if not cluster_manager.is_active():
                logger.error("Cluster manager is not active")
                raise RuntimeError("Failed to start or activate Dask cluster.")
            
            client = cluster_manager.get_client()
            if not client:
                logger.error("Failed to get Dask client")
                raise RuntimeError("Failed to get Dask client.")
            
            logger.info(f"Cluster status: {cluster_manager.is_active()}")
            logger.info(f"Client status: {client is not None}")
            logger.info(f"Dashboard link: {client.dashboard_link}")
            
            # Diagnostic logs to verify workers and GPU availability
            logger.info("Workers: %s", list(client.scheduler_info()["workers"].keys()))
            try:
                dev_ids = client.run(lambda: __import__("cupy").cuda.runtime.getDevice())
                logger.info("CUDA device per worker: %s", dev_ids)
            except Exception as e:
                logger.warning("Could not get CUDA device info: %s", e)

            logger.info(f"Processing {len(pending_tasks)} tasks in parallel using {len(client.scheduler_info()['workers'])} workers")

            # Submit all tasks to the cluster for parallel processing
            futures = []
            for task in pending_tasks:
                future = client.submit(
                    process_currency_pair,
                    task=task,
                    db_handler=None,      # será criado no worker
                    local_loader=None     # idem
                )
                futures.append(future)

            # Wait for all tasks to complete and gather results
            logger.info("Waiting for all tasks to complete...")
            results = client.gather(futures)
            
            # Count successful and failed tasks
            successful_tasks = sum(1 for result in results if result)
            failed_tasks = len(results) - successful_tasks

            logger.info("=" * 60)
            logger.info("PIPELINE EXECUTION SUMMARY")
            logger.info(f"Total currency pairs found: {len(available_pairs)}")
            logger.info(f"Already processed: {len(available_pairs) - len(pending_tasks)}")
            logger.info(f"Tasks submitted: {len(pending_tasks)}")
            logger.info(f"Successful tasks: {successful_tasks}")
            logger.info(f"Failed tasks: {failed_tasks}")
            logger.info("=" * 60)

        return 0 if failed_tasks == 0 else 1

    except Exception as e:
        logger.error(f"FATAL PIPELINE ERROR: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(run_pipeline())
