"""
Pipeline Orchestrator for Dynamic Stage 0 Pipeline

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

logger = logging.getLogger(__name__)


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
        try:
            if not self.db_handler.connect():
                logger.warning("Database unavailable; run lifecycle tracking disabled.")
                return None
            self.current_run_id = self.db_handler.create_run(hostname=hostname, dashboard_url=dashboard_url)
            return self.current_run_id
        except Exception as e:
            logger.warning(f"Could not start pipeline run: {e}")
            return None

    def end_run(self, status: str = 'COMPLETED') -> None:
        try:
            if self.current_run_id:
                self.db_handler.end_run(self.current_run_id, status=status)
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
        
        logger.info(f"Found {len(available_pairs)} currency pairs in data directory")
        
        # Clean up existing feather files for debugging if configured
        if self.settings.development.clean_existing_output:
            self._clean_existing_output_files()
        
        # Create processing tasks
        pending_tasks = []
        for pair_info in available_pairs:
            currency_pair = pair_info['currency_pair']
            
            # Check if output file already exists (idempotent approach)
            output_path = Path(self.settings.output.output_path) / currency_pair / f"{currency_pair}.feather"
            if output_path.exists() and not self.settings.development.force_reprocessing:
                logger.info(f"Skipping {currency_pair} - output already exists")
                continue
            
            # Remove existing file if force reprocessing is enabled
            if output_path.exists() and self.settings.development.force_reprocessing:
                try:
                    output_path.unlink()
                    logger.info(f"Removed existing output file for reprocessing: {output_path}")
                except Exception as e:
                    logger.warning(f"Could not remove file {output_path}: {e}")
            
            task = ProcessingTask(
                task_id=None,  # Will be assigned after successful processing
                currency_pair=currency_pair,
                r2_path=pair_info['data_path'],
                file_type=pair_info['file_type'],
                file_size_mb=pair_info['file_size_mb'],
                filename=pair_info['filename']
            )
            pending_tasks.append(task)
            logger.info(f"Added {currency_pair} to processing queue")
        
        return pending_tasks
    
    def _clean_existing_output_files(self):
        """Clean up existing feather files for debugging."""
        logger.info("🧹 Cleaning up existing feather files for debug...")
        import glob
        
        feather_files = glob.glob(f"{self.settings.output.output_path}/*/*.feather")
        for feather_file in feather_files:
            try:
                os.remove(feather_file)
                logger.info(f"Removed: {feather_file}")
            except Exception as e:
                logger.warning(f"Could not remove {feather_file}: {e}")
        
        logger.info(f"Cleaned up {len(feather_files)} feather files")
    
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
        
        logger.info(f"Processing {len(pending_tasks)} currency pairs that need feature engineering")
        
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
        
        # Process tasks on the driver (use all GPUs per task)
        return self._process_tasks_on_driver(client, pending_tasks)
    
    def _log_cluster_diagnostics(self, client: Client):
        """Log diagnostic information about the cluster."""
        logger.info(f"Cluster status: Active")
        logger.info(f"Client status: Connected")
        logger.info(f"Dashboard link: {client.dashboard_link}")
        
        # Log worker information
        workers = list(client.scheduler_info()["workers"].keys())
        logger.info(f"Workers: {workers}")
        
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
        logger.info("=" * 60)
        logger.info("STARTING DRIVER-SIDE PROCESSING (MULTI-GPU PER TASK)")
        logger.info("=" * 60)

        processor = DataProcessor(client, run_id=self.current_run_id)
        successful_tasks = 0
        failed_tasks = 0

        for i, task in enumerate(tasks):
            if self.emergency_shutdown.is_set():
                logger.critical("🚨 EMERGENCY SHUTDOWN DETECTED - STOPPING ALL PROCESSING")
                break

            logger.info("=" * 60)
            logger.info(f"TASK {i+1}/{len(tasks)}: {task.currency_pair}")
            logger.info("=" * 60)
            logger.info(f"File: {task.filename} ({task.file_size_mb:.1f} MB)")
            logger.info(f"R2 Path: {task.r2_path}")

            try:
                ok = processor.process_currency_pair_dask(task.currency_pair, task.r2_path, client)
                if ok:
                    successful_tasks += 1
                    logger.info("=" * 60)
                    logger.info(f"✅ SUCCESS: {task.currency_pair} completed successfully")
                    logger.info("=" * 60)
                else:
                    failed_tasks += 1
                    logger.error("=" * 60)
                    logger.error(f"❌ FAILURE: {task.currency_pair} failed to process")
                    logger.error("=" * 60)
                    logger.critical("FAIL-FAST: Stopping pipeline due to task failure")
                    break
            except CriticalPipelineError as e:
                failed_tasks += 1
                logger.critical("=" * 60)
                logger.critical(f"🚨 CRITICAL PIPELINE ERROR processing {task.currency_pair}")
                logger.critical(f"Error: {str(e)}")
                logger.critical("=" * 60)
                self.emergency_shutdown.set()
                break
            except Exception as e:
                failed_tasks += 1
                logger.critical("=" * 60)
                logger.critical(f"❌ CRITICAL ERROR processing {task.currency_pair}")
                logger.critical(f"Error: {str(e)}")
                logger.critical("=" * 60)
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
        logger.info("=" * 60)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info(f"Total currency pairs found: {total_discovered}")
        logger.info(f"Already processed: {total_discovered - result.total_tasks}")
        logger.info(f"Tasks attempted: {result.successful_tasks + result.failed_tasks}")
        logger.info(f"Successful tasks: {result.successful_tasks}")
        logger.info(f"Failed tasks: {result.failed_tasks}")
        
        if result.emergency_shutdown:
            logger.critical("🚨 PIPELINE STOPPED DUE TO EMERGENCY SHUTDOWN")
        else:
            logger.info("=" * 60)
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.db_handler.close()
        except Exception as e:
            logger.warning(f"Error during database cleanup: {e}")
