"""
No-Database Handler for Dynamic Stage 0 Pipeline

This module provides a database-free alternative that handles all database
operations in-memory or via file-based storage, eliminating the need for MySQL.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import os
from pathlib import Path

from config.unified_config import get_unified_config as get_config
from config.unified_config import get_unified_config as get_settings

logger = logging.getLogger(__name__)


class NoDatabaseHandler:
    """No-database handler that stores state in memory and files."""
    
    def __init__(self):
        """Initialize the no-database handler."""
        self.config = get_config()
        self.settings = get_settings()
        self.hostname = "localhost"
        
        # In-memory storage
        self.processing_tasks = {}
        self.feature_status = {}
        self.pipeline_runs = {}
        self.engine_stage_events = {}
        self.task_metrics = {}
        self.task_artifacts = {}
        
        # File-based storage directory
        self.storage_dir = Path("./data_io/storage")
        self.storage_dir.mkdir(exist_ok=True)
        
        # Load existing data if available
        self._load_from_files()
    
    def _load_from_files(self):
        """Load existing data from files."""
        try:
            for storage_type in ['processing_tasks', 'feature_status', 'pipeline_runs', 
                               'engine_stage_events', 'task_metrics', 'task_artifacts']:
                file_path = self.storage_dir / f"{storage_type}.json"
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        setattr(self, storage_type, data)
        except Exception as e:
            logger.warning(f"Could not load existing data: {e}")
    
    def _save_to_files(self):
        """Save current data to files."""
        try:
            for storage_type in ['processing_tasks', 'feature_status', 'pipeline_runs', 
                               'engine_stage_events', 'task_metrics', 'task_artifacts']:
                file_path = self.storage_dir / f"{storage_type}.json"
                data = getattr(self, storage_type)
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Could not save data to files: {e}")
    
    def connect(self) -> bool:
        """
        "Connect" to the no-database system.
        
        Returns:
            bool: Always True (no actual connection needed)
        """
        logger.info("Using no-database handler (MySQL disabled)")
        return True
    
    def create_tables(self):
        """No-op for no-database handler."""
        pass
    
    # ------------------ Run lifecycle APIs ------------------
    def create_run(self, hostname: Optional[str] = None, dashboard_url: Optional[str] = None,
                   git_sha: Optional[str] = None, config_snapshot: Optional[str] = None) -> Optional[int]:
        if not getattr(self.settings.monitoring, 'metrics_enabled', False):
            return None
        
        run_id = len(self.pipeline_runs) + 1
        self.pipeline_runs[run_id] = {
            'run_id': run_id,
            'started_at': datetime.utcnow().isoformat(),
            'ended_at': None,
            'hostname': hostname,
            'dashboard_url': dashboard_url,
            'git_sha': git_sha,
            'status': 'RUNNING',
            'config_snapshot': config_snapshot
        }
        self._save_to_files()
        logger.info(f"Created pipeline run {run_id}")
        return run_id

    def end_run(self, run_id: int, status: str = 'COMPLETED') -> bool:
        if not run_id or run_id not in self.pipeline_runs:
            return False
        if not getattr(self.settings.monitoring, 'metrics_enabled', False):
            return True
        
        self.pipeline_runs[run_id]['ended_at'] = datetime.utcnow().isoformat()
        self.pipeline_runs[run_id]['status'] = status
        self._save_to_files()
        logger.info(f"Ended pipeline run {run_id} with status={status}")
        return True

    # ------------------ Stage events APIs ------------------
    def start_stage(self, run_id: Optional[int], task_id: Optional[int], engine_name: str,
                    rows_before: Optional[int] = None, cols_before: Optional[int] = None,
                    message: Optional[str] = None) -> Optional[int]:
        if not getattr(self.settings.monitoring, 'metrics_enabled', False):
            return None
        
        stage_id = len(self.engine_stage_events) + 1
        self.engine_stage_events[stage_id] = {
            'stage_id': stage_id,
            'run_id': run_id,
            'task_id': task_id,
            'engine_name': engine_name,
            'status': 'START',
            'start_time': datetime.utcnow().isoformat(),
            'end_time': None,
            'rows_before': rows_before,
            'cols_before': cols_before,
            'cols_after': None,
            'new_cols': None,
            'hostname': self.hostname,
            'message': message,
            'error_message': None,
            'details': None
        }
        self._save_to_files()
        return stage_id

    def end_stage(self, stage_id: int, rows_after: Optional[int] = None, cols_after: Optional[int] = None,
                  new_cols: Optional[int] = None, details: Optional[str] = None) -> bool:
        if not getattr(self.settings.monitoring, 'metrics_enabled', False):
            return True
        
        if stage_id not in self.engine_stage_events:
            return False
        
        self.engine_stage_events[stage_id].update({
            'status': 'END',
            'end_time': datetime.utcnow().isoformat(),
            'rows_after': rows_after,
            'cols_after': cols_after,
            'new_cols': new_cols,
            'details': details
        })
        self._save_to_files()
        return True

    def error_stage(self, stage_id: int, error_message: str, details: Optional[str] = None) -> bool:
        if not getattr(self.settings.monitoring, 'metrics_enabled', False):
            return True
        
        if stage_id not in self.engine_stage_events:
            return False
        
        self.engine_stage_events[stage_id].update({
            'status': 'ERROR',
            'end_time': datetime.utcnow().isoformat(),
            'error_message': error_message,
            'details': details
        })
        self._save_to_files()
        return True

    # ------------------ Metrics & Artifacts ------------------
    def add_metrics(self, run_id: Optional[int], task_id: Optional[int], stage: str, metrics: Dict[str, Any]) -> bool:
        if not getattr(self.settings.monitoring, 'metrics_enabled', False):
            return True
        
        for key, value in (metrics or {}).items():
            metric_id = len(self.task_metrics) + 1
            self.task_metrics[metric_id] = {
                'metric_id': metric_id,
                'run_id': run_id,
                'task_id': task_id,
                'stage': stage,
                'key': str(key),
                'value_text': json.dumps(value) if not isinstance(value, (int, float)) else None,
                'value_float': str(float(value)) if isinstance(value, (int, float)) else None,
                'created_at': datetime.utcnow().isoformat()
            }
        
        self._save_to_files()
        return True

    def add_artifact(self, run_id: Optional[int], task_id: Optional[int], stage: str, path: str, kind: str = 'file', meta: Optional[Dict[str, Any]] = None) -> bool:
        if not getattr(self.settings.monitoring, 'metrics_enabled', False):
            return True
        
        artifact_id = len(self.task_artifacts) + 1
        self.task_artifacts[artifact_id] = {
            'artifact_id': artifact_id,
            'run_id': run_id,
            'task_id': task_id,
            'stage': stage,
            'path': path,
            'kind': kind,
            'meta': json.dumps(meta) if meta else None,
            'created_at': datetime.utcnow().isoformat()
        }
        self._save_to_files()
        return True

    def clear_old_records(self):
        """Clear old processing records to start fresh."""
        self.processing_tasks.clear()
        self.feature_status.clear()
        self._save_to_files()
        logger.info("Cleared old processing records")
    
    def get_pending_currency_pairs(self) -> List[Dict[str, Any]]:
        """
        Get the list of pending currency pairs.
        
        Returns:
            List[Dict[str, Any]]: List of pending tasks
        """
        pending_tasks = []
        
        for task_id, task in self.processing_tasks.items():
            status_info = self.feature_status.get(task_id, {})
            current_status = status_info.get('status', 'PENDING')
            
            # Include if pending, failed, or running for more than 1 hour
            if current_status in ['PENDING', 'FAILED']:
                pending_tasks.append({
                    'task_id': task_id,
                    'currency_pair': task['currency_pair'],
                    'r2_path': task['r2_path'],
                    'added_timestamp': task['added_timestamp'],
                    'current_status': current_status,
                    'start_time': status_info.get('start_time'),
                    'end_time': status_info.get('end_time'),
                    'hostname': status_info.get('hostname'),
                    'error_message': status_info.get('error_message')
                })
        
        logger.info(f"Found {len(pending_tasks)} pending currency pairs")
        return pending_tasks
    
    def update_task_status(self, task_id: int, status: str, error_message: Optional[str] = None) -> bool:
        """
        Update the status of a task.
        
        Args:
            task_id: The task ID to update
            status: The new status ('RUNNING', 'COMPLETED', or 'FAILED')
            error_message: Optional error message for FAILED status
            
        Returns:
            bool: True if update successful, False otherwise
        """
        if status not in ['RUNNING', 'COMPLETED', 'FAILED']:
            logger.error(f"Invalid status: {status}. Must be one of: RUNNING, COMPLETED, FAILED")
            return False
        
        current_time = datetime.utcnow().isoformat()
        
        if status == 'RUNNING':
            self.feature_status[task_id] = {
                'task_id': task_id,
                'status': 'RUNNING',
                'start_time': current_time,
                'end_time': None,
                'hostname': self.hostname,
                'error_message': None
            }
        elif status == 'COMPLETED':
            if task_id in self.feature_status:
                self.feature_status[task_id].update({
                    'status': 'COMPLETED',
                    'end_time': current_time
                })
        elif status == 'FAILED':
            if task_id in self.feature_status:
                self.feature_status[task_id].update({
                    'status': 'FAILED',
                    'end_time': current_time,
                    'error_message': error_message or 'Unknown error'
                })
        
        self._save_to_files()
        logger.info(f"Updated task {task_id} status to {status}")
        return True
    
    def get_task_info(self, task_id: int) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific task.
        
        Args:
            task_id: The task ID to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: Task information or None if not found
        """
        if task_id not in self.processing_tasks:
            logger.warning(f"Task {task_id} not found")
            return None
        
        task = self.processing_tasks[task_id]
        status_info = self.feature_status.get(task_id, {})
        
        return {
            'task_id': task_id,
            'currency_pair': task['currency_pair'],
            'r2_path': task['r2_path'],
            'added_timestamp': task['added_timestamp'],
            'status': status_info.get('status'),
            'start_time': status_info.get('start_time'),
            'end_time': status_info.get('end_time'),
            'hostname': status_info.get('hostname'),
            'error_message': status_info.get('error_message')
        }
    
    def close(self):
        """Close the no-database handler."""
        self._save_to_files()
        logger.info("No-database handler closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def is_currency_pair_processed(self, currency_pair: str) -> bool:
        """
        Check if a currency pair has been successfully processed.
        
        Args:
            currency_pair: The currency pair to check (e.g., 'EURUSD')
            
        Returns:
            bool: True if processed, False otherwise
        """
        for task_id, task in self.processing_tasks.items():
            if task['currency_pair'] == currency_pair:
                status_info = self.feature_status.get(task_id, {})
                if status_info.get('status') == 'COMPLETED':
                    logger.info(f"Currency pair {currency_pair} already processed")
                    return True
        
        logger.info(f"Currency pair {currency_pair} not yet processed")
        return False

    def register_currency_pair(self, currency_pair: str, data_path: str) -> Optional[int]:
        """
        Register a new currency pair for processing.
        
        Args:
            currency_pair: The currency pair identifier
            data_path: Path to the data files
            
        Returns:
            Optional[int]: Task ID if successful, None otherwise
        """
        task_id = len(self.processing_tasks) + 1
        self.processing_tasks[task_id] = {
            'task_id': task_id,
            'currency_pair': currency_pair,
            'r2_path': data_path,
            'added_timestamp': datetime.utcnow().isoformat()
        }
        self._save_to_files()
        logger.info(f"Registered currency pair {currency_pair} with task ID {task_id}")
        return task_id

    def get_processed_features(self, currency_pair: str) -> Optional[Dict[str, Any]]:
        """
        Get information about features that were generated for a currency pair.
        
        Args:
            currency_pair: The currency pair to check
            
        Returns:
            Optional[Dict]: Feature information if processed, None otherwise
        """
        for task_id, task in self.processing_tasks.items():
            if task['currency_pair'] == currency_pair:
                status_info = self.feature_status.get(task_id, {})
                if status_info.get('status') == 'COMPLETED':
                    return {
                        'task_id': task_id,
                        'currency_pair': currency_pair,
                        'data_path': task['r2_path'],
                        'status': 'COMPLETED',
                        'start_time': status_info.get('start_time'),
                        'end_time': status_info.get('end_time'),
                        'hostname': status_info.get('hostname')
                    }
        return None


# Convenience functions for direct use
def get_pending_currency_pairs() -> List[Dict[str, Any]]:
    """
    Convenience function to get pending currency pairs.
    
    Returns:
        List[Dict[str, Any]]: List of pending tasks
    """
    with NoDatabaseHandler() as db:
        return db.get_pending_currency_pairs()


def update_task_status(task_id: int, status: str, error_message: Optional[str] = None) -> bool:
    """
    Convenience function to update task status.
    
    Args:
        task_id: The task ID to update
        status: The new status
        error_message: Optional error message
        
    Returns:
        bool: True if successful, False otherwise
    """
    with NoDatabaseHandler() as db:
        return db.update_task_status(task_id, status, error_message)


