"""
Pipeline Orchestrator for Feature Engineering Pipeline

This module handles the high-level orchestration of the feature engineering pipeline,
separating orchestration logic from processing logic.
"""

import logging
import os
import signal
import threading
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from dask.distributed import Client, as_completed
from dask_cuda import LocalCUDACluster

from config.unified_config import get_unified_config as get_settings
from data_io.db_handler import DatabaseHandler
from data_io.local_loader import LocalDataLoader
from features.base_engine import CriticalPipelineError
from orchestration.data_processor import DataProcessor, process_currency_pair_dask_worker
from utils.logging_utils import get_logger, set_currency_pair_context
from utils.pipeline_visualizer import PipelineVisualizer
from monitoring.pipeline_dashboard import PipelineDashboard
from monitoring.smart_alerts import SmartAlertSystem

logger = get_logger(__name__, component="orchestration.orchestrator")


@dataclass
class ProcessingTask:
    """Represents a single processing task."""
    task_id: Optional[str]
    currency_pair: str
    r2_path: str
    file_type: str
    file_size_mb: float
    filename: str


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    emergency_shutdown: bool
    error_message: Optional[str] = None


class PipelineOrchestrator:
    """
    Orchestrates the complete feature engineering pipeline.
    
    This class handles:
    - Task discovery and queuing
    - Cluster management coordination
    - Task execution monitoring
    - Error handling and recovery
    - Pipeline lifecycle management
    """
    
    def __init__(self):
        """Initialize the pipeline orchestrator."""
        self.settings = get_settings()
        self.db_handler = DatabaseHandler()
        self.local_loader = LocalDataLoader()
        self.emergency_shutdown = threading.Event()
        self.current_run_id: Optional[int] = None
        
        # Transparency/monitoring helpers
        self.dashboard = PipelineDashboard()
        self.visualizer = PipelineVisualizer(self.settings)
        self.alerts = SmartAlertSystem()
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    # ---------------- Run lifecycle ----------------
    def start_run(self, dashboard_url: Optional[str] = None, hostname: Optional[str] = None) -> Optional[int]:
        """Starts a new pipeline run, if the database is available."""
        try:
            if not self.db_handler.connect():
                logger.warning("Database unavailable; run lifecycle tracking disabled.")
                return None
            self.current_run_id = self.db_handler.create_run(hostname=hostname, dashboard_url=dashboard_url)
            logger.info(f"Started pipeline run {self.current_run_id} on {hostname} with dashboard at {dashboard_url}")
            return self.current_run_id
        except Exception as e:
            logger.warning(f"Could not start pipeline run: {e}")
            return None

    def end_run(self, status: str = 'COMPLETED') -> None:
        """Ends the current pipeline run, if one is active."""
        try:
            if self.current_run_id:
                self.db_handler.end_run(self.current_run_id, status=status)
                logger.info(f"Ended pipeline run {self.current_run_id} with status={status}")
        except Exception as e:
            logger.warning(f"Could not end pipeline run: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.emergency_shutdown.set()
    
    def discover_tasks(self) -> List[ProcessingTask]:
        """
        Discover all available currency pairs that need processing.
        
        Returns:
            List of ProcessingTask objects representing work to be done.
        """
        logger.info("Discovering currency pairs in local data directory...")
        
        available_pairs = self.local_loader.discover_currency_pairs()
        if not available_pairs:
            logger.info("No currency pairs found in data directory.")
            return []
        
        logger.info(f"Currency pairs found: {len(available_pairs)}")
        try:
            self.dashboard.set_total_pairs(len(available_pairs))
        except Exception:
            pass
        
        # Clean up existing feather files for debugging if configured
        if self.settings.development.clean_existing_output:
            self._clean_existing_output_files()
        
        # Create processing tasks
        pending_tasks = []
        for pair_info in available_pairs:
            currency_pair = pair_info['currency_pair']
            
            # Check if output file already exists (idempotent approach)
            out_dir = Path(self.settings.output.output_path) / currency_pair
            consolidated = out_dir / f"{currency_pair}.parquet"
            if not self.settings.development.force_reprocessing:
                if consolidated.exists():
                    logger.info(f"Skipping pair {currency_pair}; consolidated output exists")
                    continue
                # Also skip if partitioned parquet outputs exist
                try:
                    if out_dir.exists() and any(out_dir.glob("part-*.parquet")):
                        logger.info(f"Skipping pair {currency_pair}; partitioned output exists")
                        continue
                except Exception:
                    pass
            
            # Remove existing file if force reprocessing is enabled
            if consolidated.exists() and self.settings.development.force_reprocessing:
                try:
                    consolidated.unlink()
                    logger.info(f"Removed consolidated output for reprocessing: {consolidated}")
                except Exception as e:
                    logger.warning(f"Could not remove consolidated output {consolidated}: {e}")
                # Remove partitioned outputs as well
                try:
                    for p in out_dir.glob("part-*.parquet"):
                        p.unlink()
                    logger.info(f"Removed existing partitioned outputs: {out_dir}")
                except Exception as e:
                    logger.warning(f"Could not remove partitioned outputs {out_dir}: {e}")
            
            task = ProcessingTask(
                task_id=None,  # Will be assigned after successful processing
                currency_pair=currency_pair,
                r2_path=pair_info['data_path'],
                file_type=pair_info['file_type'],
                file_size_mb=pair_info['file_size_mb'],
                filename=pair_info['filename']
            )
            pending_tasks.append(task)
            logger.info(f"Added pair {currency_pair} to processing queue")
        
        return pending_tasks
    
    def _clean_existing_output_files(self):
        """Clean up existing parquet files for debugging."""
        logger.info("Cleaning up existing parquet files for debug...")
        import glob
        
        parquet_files = glob.glob(f"{self.settings.output.output_path}/*/*.parquet")
        for parquet_file in parquet_files:
            try:
                os.remove(parquet_file)
                logger.info(f"Removed file: {parquet_file}")
            except Exception as e:
                logger.warning(f"Could not remove file {parquet_file}: {e}")
        
        logger.info(f"Cleanup complete, removed {len(parquet_files)} files")
    
    def connect_database(self) -> bool:
        """Connect to the database for task tracking."""
        if not self.db_handler.connect():
            logger.error("Failed to connect to database.")
            return False
        return True
    
    def execute_pipeline(self, cluster_manager, client: Client) -> PipelineResult:
        """
        Execute the complete pipeline with the given cluster and client.
        
        Args:
            cluster_manager: The Dask cluster manager
            client: The Dask distributed client
            
        Returns:
            PipelineResult with execution summary
        """
        # Discover tasks
        pending_tasks = self.discover_tasks()
        if not pending_tasks:
            logger.info("All currency pairs already processed. Pipeline complete.")
            return PipelineResult(
                total_tasks=0,
                successful_tasks=0,
                failed_tasks=0,
                emergency_shutdown=False
            )
        
        logger.info(f"Pending tasks to process: {len(pending_tasks)}")
        
        # Verify cluster status
        if not cluster_manager.is_active():
            error_msg = "Cluster manager is not active"
            logger.error(error_msg)
            return PipelineResult(
                total_tasks=len(pending_tasks),
                successful_tasks=0,
                failed_tasks=0,
                emergency_shutdown=True,
                error_message=error_msg
            )
        
        if not client:
            error_msg = "Failed to get Dask client"
            logger.error(error_msg)
            return PipelineResult(
                total_tasks=len(pending_tasks),
                successful_tasks=0,
                failed_tasks=0,
                emergency_shutdown=True,
                error_message=error_msg
            )
        
        # Log cluster diagnostics
        self._log_cluster_diagnostics(client)
        
        # Process tasks in parallel, one currency pair per GPU worker
        return self._process_tasks_distributed(client, pending_tasks)
    
    def _log_cluster_diagnostics(self, client: Client):
        """Log diagnostic information about the cluster."""
        # Get cluster information
        workers = list(client.scheduler_info()["workers"].keys())
        worker_count = len(workers)
        dashboard_url = str(client.dashboard_link)
        
        # Get GPU count (assume 1 GPU per worker for now)
        gpu_count = worker_count
        
        logger.info(f"Cluster is ready for processing: {worker_count} workers, {gpu_count} GPUs, dashboard at {dashboard_url}")
        
        # Log worker information
        logger.info(f"Workers connected: {workers}")
        
        # Log CUDA device information
        try:
            dev_ids = client.run(lambda: __import__("cupy").cuda.runtime.getDevice())
            logger.info(f"CUDA device per worker: {dev_ids}")
        except Exception as e:
            logger.warning(f"Could not get CUDA device info: {e}")
    

    def _process_tasks_on_driver(self, client: Client, tasks: List[ProcessingTask]) -> PipelineResult:
        """
        Process tasks sequentially on the driver, leveraging Dask-CUDA within each task
        to use all GPUs for a single currency pair. Fail-fast on first error.
        """
        logger.info("Starting driver-side processing (multi-GPU per task)")

        processor = DataProcessor(client, run_id=self.current_run_id)
        successful_tasks = 0
        failed_tasks = 0

        for i, task in enumerate(tasks):
            if self.emergency_shutdown.is_set():
                logger.critical("Emergency shutdown detected - stopping all processing")
                break

            logger.info(f"Processing currency pair {i+1}/{len(tasks)}: {task.currency_pair} ({task.filename}, {round(task.file_size_mb, 1)}MB)")
            try:
                self.dashboard.set_current_pair(task.currency_pair)
            except Exception:
                pass

            try:
                # Set currency pair context for all subsequent logs
                set_currency_pair_context(task.currency_pair)
                ok = processor.process_currency_pair_dask(task.currency_pair, task.r2_path, client)
                if ok:
                    successful_tasks += 1
                    logger.info(f"Task completed successfully: {task.currency_pair}")
                    try:
                        self.dashboard.increment_completed()
                    except Exception:
                        pass
                    # Free worker memory proactively between tasks
                    try:
                        client.run(lambda: (__import__('gc').collect(), __import__('cupy').get_default_memory_pool().free_all_blocks()))
                    except Exception as e:
                        # keep as debug
                        pass
                else:
                    failed_tasks += 1
                    logger.error(f"Task failed to process: {task.currency_pair}")
                    logger.warning("Stopping pipeline due to task failure")
                    break
            except CriticalPipelineError as e:
                failed_tasks += 1
                logger.error(f"Critical pipeline error for {task.currency_pair}: {e}")
                self.emergency_shutdown.set()
                break
            except Exception as e:
                failed_tasks += 1
                logger.error(f"Critical error processing task {task.currency_pair}: {e}")
                self.emergency_shutdown.set()
                break

        return PipelineResult(
            total_tasks=len(tasks),
            successful_tasks=successful_tasks,
            failed_tasks=failed_tasks,
            emergency_shutdown=self.emergency_shutdown.is_set()
        )

    def _process_tasks_distributed(self, client: Client, tasks: List[ProcessingTask]) -> PipelineResult:
        """Process tasks in parallel, one currency pair per Dask-CUDA worker (GPU).

        Submits one task per currency pair using the worker-side implementation that
        runs on a single GPU. Dask-CUDA ensures each worker is pinned to a unique GPU.
        """
        if not tasks:
            return PipelineResult(total_tasks=0, successful_tasks=0, failed_tasks=0, emergency_shutdown=False)

        try:
            workers = list(client.scheduler_info().get("workers", {}).keys())
            logger.info(f"Distributed processing start: {len(tasks)} tasks across {len(workers)} workers")
        except Exception:
            workers = []
            logger.info(f"Distributed processing start: {len(tasks)} tasks")

        # Submit up to one task per worker; queue the rest. Target specific workers to avoid multi-task per GPU.
        task_queue = list(tasks)
        futures_by_worker = {}
        future_to_task = {}

        # Helper to submit next task to a specific worker
        def _submit_next(worker_addr: str):
            if not task_queue:
                return None
            task = task_queue.pop(0)
            logger.info(f"Submitting {task.currency_pair} to worker {worker_addr}")
            # Set currency pair context for submission logs
            set_currency_pair_context(task.currency_pair)
            fut = client.submit(
                process_currency_pair_dask_worker,
                task.currency_pair,
                task.r2_path,
                key=f"pair-{task.currency_pair}",
                pure=False,
                workers=[worker_addr],
                allow_other_workers=False,
            )
            futures_by_worker[worker_addr] = fut
            future_to_task[fut] = task
            return fut

        # Prime the pipeline: one per worker
        for w in workers:
            if self.emergency_shutdown.is_set() or not task_queue:
                break
            try:
                _submit_next(w)
            except Exception as e:
                logger.error(f"Failed initial submit to worker {w}: {e}")

        successful_tasks = 0
        failed_tasks = 0

        # Process completions and keep workers busy until queue empties
        ac = as_completed(list(futures_by_worker.values()), with_results=False, raise_errors=False)
        for fut in ac:
            task = future_to_task.get(fut)
            worker_addr = None
            # Find which worker this future was targeting
            try:
                for w, f in list(futures_by_worker.items()):
                    if f == fut:
                        worker_addr = w
                        break
            except Exception:
                worker_addr = None

            if self.emergency_shutdown.is_set():
                logger.critical("Emergency shutdown detected - cancelling remaining tasks")
                try:
                    for f in list(futures_by_worker.values()):
                        if not f.done():
                            f.cancel()
                except Exception:
                    pass
                break

            try:
                ok = fut.result()
            except Exception as e:
                ok = False
                pname = task.currency_pair if task else "<unknown>"
                logger.error(f"Task failed for {pname}: {e}")

            pname = task.currency_pair if task else "<unknown>"
            if ok:
                successful_tasks += 1
                logger.info(f"Task completed successfully: {pname}")
                try:
                    self.dashboard.increment_completed()
                except Exception:
                    pass
            else:
                failed_tasks += 1
                logger.error(f"Task reported failure: {pname}")

            # Free worker memory proactively between tasks
            try:
                client.run(lambda: (__import__('gc').collect(), __import__('cupy').get_default_memory_pool().free_all_blocks()), workers=[worker_addr] if worker_addr else None)
            except Exception:
                pass

            # Submit next task to this worker if available
            if worker_addr and task_queue:
                try:
                    new_fut = _submit_next(worker_addr)
                    if new_fut:
                        ac.add(new_fut)
                except Exception as e:
                    logger.error(f"Failed to submit next task to {worker_addr}: {e}")

        return PipelineResult(
            total_tasks=len(tasks),
            successful_tasks=successful_tasks,
            failed_tasks=failed_tasks,
            emergency_shutdown=self.emergency_shutdown.is_set(),
        )
    
    def log_pipeline_summary(self, result: PipelineResult, total_discovered: int):
        """Log a summary of the pipeline execution."""
        logger.info(f"Pipeline execution summary: {total_discovered} found, {total_discovered - result.total_tasks} already processed, {result.successful_tasks + result.failed_tasks} attempted, {result.successful_tasks} successful, {result.failed_tasks} failed")
        
        if result.emergency_shutdown:
            logger.error("Pipeline stopped due to emergency shutdown")
        else:
            # Generate visual artifacts for transparency (best-effort)
            try:
                diagram = self.visualizer.generate_pipeline_diagram()
                logger.info(f"Pipeline diagram generated: {diagram}")
                evo_path = self.visualizer.create_feature_evolution_report({
                    'stages': [],
                    'summary': {
                        'total_found': total_discovered,
                        'attempted': result.successful_tasks + result.failed_tasks,
                        'success': result.successful_tasks,
                        'failed': result.failed_tasks,
                    }
                })
                logger.info(f"Feature evolution report saved: {evo_path}")
                impact = self.visualizer.generate_config_impact_analysis()
                logger.info(f"Config impact analysis: {impact}")
                # Alerts: feature count deviation wrt discovered
                try:
                    self.alerts.check_feature_count_anomalies(result.successful_tasks + result.failed_tasks, total_discovered)
                    usage_hist = self.dashboard.show_memory_usage_evolution()
                    self.alerts.check_memory_usage_patterns(usage_hist)
                    alerts = self.alerts.get_alerts()
                    logger.info(f"Alert summary: {alerts}")
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"Could not generate visualization artifacts: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.db_handler.close()
        except Exception as e:
            logger.warning(f"Error during database cleanup: {e}")
