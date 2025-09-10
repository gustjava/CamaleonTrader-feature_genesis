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

from dask.distributed import Client
from dask_cuda import LocalCUDACluster

from config.unified_config import get_unified_config as get_settings
from data_io.db_handler import DatabaseHandler
from data_io.local_loader import LocalDataLoader
from features.base_engine import CriticalPipelineError
from orchestration.data_processor import DataProcessor
from utils.logging_utils import (
    get_logger,
    info_event,
    warn_event,
    error_event,
)
from utils import log_context

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
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    # ---------------- Run lifecycle ----------------
    def start_run(self, dashboard_url: Optional[str] = None, hostname: Optional[str] = None) -> Optional[int]:
        """Starts a new pipeline run, if the database is available."""
        try:
            if not self.db_handler.connect():
                warn_event(logger, "db.unavailable", "Database unavailable; run lifecycle tracking disabled.")
                return None
            self.current_run_id = self.db_handler.create_run(hostname=hostname, dashboard_url=dashboard_url)
            log_context.set_run_id(self.current_run_id)
            info_event(logger, "run.start", "Started pipeline run", run_id=self.current_run_id, hostname=hostname, dashboard_url=dashboard_url)
            return self.current_run_id
        except Exception as e:
            warn_event(logger, "run.start.warn", "Could not start pipeline run", error=str(e))
            return None

    def end_run(self, status: str = 'COMPLETED') -> None:
        """Ends the current pipeline run, if one is active."""
        try:
            if self.current_run_id:
                self.db_handler.end_run(self.current_run_id, status=status)
                info_event(logger, "run.end", "Ended pipeline run", run_id=self.current_run_id, status=status)
        except Exception as e:
            warn_event(logger, "run.end.warn", "Could not end pipeline run", error=str(e))
    
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
        info_event(logger, "task.discovery.start", "Discovering currency pairs in local data directory...")
        
        available_pairs = self.local_loader.discover_currency_pairs()
        if not available_pairs:
            info_event(logger, "task.discovery.none", "No currency pairs found in data directory.")
            return []
        
        info_event(logger, "task.discovery.found", "Currency pairs found", count=len(available_pairs))
        
        # Clean up existing feather files for debugging if configured
        if self.settings.development.clean_existing_output:
            self._clean_existing_output_files()
        
        # Create processing tasks
        pending_tasks = []
        for pair_info in available_pairs:
            currency_pair = pair_info['currency_pair']
            
            # Check if output file already exists (idempotent approach)
            out_dir = Path(self.settings.output.output_path) / currency_pair
            consolidated = out_dir / f"{currency_pair}.feather"
            if not self.settings.development.force_reprocessing:
                if consolidated.exists():
                    info_event(logger, "task.skip.existing", "Skipping pair; consolidated output exists", pair=currency_pair)
                    continue
                # Also skip if partitioned feather outputs exist
                try:
                    if out_dir.exists() and any(out_dir.glob("part-*.feather")):
                        info_event(logger, "task.skip.partitioned", "Skipping pair; partitioned output exists", pair=currency_pair)
                        continue
                except Exception:
                    pass
            
            # Remove existing file if force reprocessing is enabled
            if consolidated.exists() and self.settings.development.force_reprocessing:
                try:
                    consolidated.unlink()
                    info_event(logger, "output.remove", "Removed consolidated output for reprocessing", path=str(consolidated))
                except Exception as e:
                    warn_event(logger, "output.remove.warn", "Could not remove consolidated output", path=str(consolidated), error=str(e))
                # Remove partitioned outputs as well
                try:
                    for p in out_dir.glob("part-*.feather"):
                        p.unlink()
                    info_event(logger, "output.remove", "Removed existing partitioned outputs", path=str(out_dir))
                except Exception as e:
                    warn_event(logger, "output.remove.warn", "Could not remove partitioned outputs", path=str(out_dir), error=str(e))
            
            task = ProcessingTask(
                task_id=None,  # Will be assigned after successful processing
                currency_pair=currency_pair,
                r2_path=pair_info['data_path'],
                file_type=pair_info['file_type'],
                file_size_mb=pair_info['file_size_mb'],
                filename=pair_info['filename']
            )
            pending_tasks.append(task)
            info_event(logger, "task.queue.added", "Added pair to processing queue", pair=currency_pair)
        
        return pending_tasks
    
    def _clean_existing_output_files(self):
        """Clean up existing feather files for debugging."""
        info_event(logger, "output.cleanup.start", "Cleaning up existing feather files for debug...")
        import glob
        
        feather_files = glob.glob(f"{self.settings.output.output_path}/*/*.feather")
        for feather_file in feather_files:
            try:
                os.remove(feather_file)
                info_event(logger, "output.cleanup.removed", "Removed file", path=feather_file)
            except Exception as e:
                warn_event(logger, "output.cleanup.warn", "Could not remove file", path=feather_file, error=str(e))
        
        info_event(logger, "output.cleanup.end", "Cleanup complete", removed=len(feather_files))
    
    def connect_database(self) -> bool:
        """Connect to the database for task tracking."""
        if not self.db_handler.connect():
            error_event(logger, "db.connect.error", "Failed to connect to database.")
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
        
        info_event(logger, "task.discovery.summary", "Pending tasks to process", count=len(pending_tasks))
        
        # Verify cluster status
        if not cluster_manager.is_active():
            error_msg = "Cluster manager is not active"
            error_event(logger, "cluster.inactive", error_msg)
            return PipelineResult(
                total_tasks=len(pending_tasks),
                successful_tasks=0,
                failed_tasks=0,
                emergency_shutdown=True,
                error_message=error_msg
            )
        
        if not client:
            error_msg = "Failed to get Dask client"
            error_event(logger, "client.missing", error_msg)
            return PipelineResult(
                total_tasks=len(pending_tasks),
                successful_tasks=0,
                failed_tasks=0,
                emergency_shutdown=True,
                error_message=error_msg
            )
        
        # Log cluster diagnostics
        self._log_cluster_diagnostics(client)
        
        # Process tasks on the driver (use all GPUs per task)
        return self._process_tasks_on_driver(client, pending_tasks)
    
    def _log_cluster_diagnostics(self, client: Client):
        """Log diagnostic information about the cluster."""
        # Get cluster information
        workers = list(client.scheduler_info()["workers"].keys())
        worker_count = len(workers)
        dashboard_url = str(client.dashboard_link)
        
        # Get GPU count from context or detect
        gpu_count = log_context.get_context().get('gpu_count', worker_count)
        
        info_event(logger, "cluster.ready", "Cluster is ready for processing", 
                   gpu_count=gpu_count, workers=worker_count, dashboard_url=dashboard_url)
        
        # Log worker information
        info_event(logger, "cluster.workers", "Workers connected", workers=workers)
        
        # Log CUDA device information
        try:
            dev_ids = client.run(lambda: __import__("cupy").cuda.runtime.getDevice())
            info_event(logger, "cluster.cuda.devices", "CUDA device per worker", devices=dev_ids)
        except Exception as e:
            warn_event(logger, "cluster.cuda.warn", "Could not get CUDA device info", error=str(e))
    

    def _process_tasks_on_driver(self, client: Client, tasks: List[ProcessingTask]) -> PipelineResult:
        """
        Process tasks sequentially on the driver, leveraging Dask-CUDA within each task
        to use all GPUs for a single currency pair. Fail-fast on first error.
        """
        info_event(logger, "task.execution.start", "Starting driver-side processing (multi-GPU per task)")

        processor = DataProcessor(client, run_id=self.current_run_id)
        successful_tasks = 0
        failed_tasks = 0

        for i, task in enumerate(tasks):
            if self.emergency_shutdown.is_set():
                logger.critical("Emergency shutdown detected - stopping all processing")
                break

            info_event(
                logger,
                "task.start",
                "Processing currency pair",
                index=i + 1,
                total=len(tasks),
                pair=task.currency_pair,
                filename=task.filename,
                size_mb=round(task.file_size_mb, 1),
                r2_path=task.r2_path,
            )

            try:
                ok = processor.process_currency_pair_dask(task.currency_pair, task.r2_path, client)
                if ok:
                    successful_tasks += 1
                    info_event(logger, "task.success", "Task completed successfully", pair=task.currency_pair)
                    # Free worker memory proactively between tasks
                    try:
                        client.run(lambda: (__import__('gc').collect(), __import__('cupy').get_default_memory_pool().free_all_blocks()))
                    except Exception as e:
                        # keep as debug
                        pass
                else:
                    failed_tasks += 1
                    error_event(logger, "task.failure", "Task failed to process", pair=task.currency_pair)
                    warn_event(logger, "pipeline.failfast", "Stopping pipeline due to task failure")
                    break
            except CriticalPipelineError as e:
                failed_tasks += 1
                error_event(logger, "task.critical", "Critical pipeline error", pair=task.currency_pair, error=str(e))
                self.emergency_shutdown.set()
                break
            except Exception as e:
                failed_tasks += 1
                error_event(logger, "task.critical", "Critical error processing task", pair=task.currency_pair, error=str(e))
                self.emergency_shutdown.set()
                break

        return PipelineResult(
            total_tasks=len(tasks),
            successful_tasks=successful_tasks,
            failed_tasks=failed_tasks,
            emergency_shutdown=self.emergency_shutdown.is_set()
        )
    
    def log_pipeline_summary(self, result: PipelineResult, total_discovered: int):
        """Log a summary of the pipeline execution."""
        info_event(logger, "pipeline.summary", "Pipeline execution summary",
                   total_found=total_discovered,
                   already_processed=total_discovered - result.total_tasks,
                   attempted=result.successful_tasks + result.failed_tasks,
                   success=result.successful_tasks,
                   failed=result.failed_tasks)
        
        if result.emergency_shutdown:
            error_event(logger, "pipeline.abort", "Pipeline stopped due to emergency shutdown")
        else:
            pass
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.db_handler.close()
        except Exception as e:
            warn_event(logger, "db.cleanup.warn", "Error during database cleanup", error=str(e))
