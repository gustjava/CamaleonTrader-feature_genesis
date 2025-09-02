"""
Database Handler for Dynamic Stage 0 Pipeline

This module provides database operations using SQLAlchemy Core for:
- Database connection management
- Fetching pending currency pairs
- Updating task status
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import socket

from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, DateTime, Text, Enum
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool

from config.unified_config import get_unified_config as get_config
from config.unified_config import get_unified_config as get_settings

logger = logging.getLogger(__name__)


class DatabaseHandler:
    """Database handler for Dynamic Stage 0 pipeline state management."""
    
    def __init__(self):
        """Initialize database handler with configuration."""
        self.config = get_config()
        self.settings = get_settings()
        self.engine: Optional[Engine] = None
        self.hostname = socket.gethostname()
        
        # Define table metadata
        self.metadata = MetaData()

        # Define processing_tasks table
        self.processing_tasks = Table(
            'processing_tasks',
            self.metadata,
            Column('task_id', Integer, primary_key=True, autoincrement=True),
            Column('currency_pair', String(16), nullable=False, unique=True),
            Column('r2_path', String(1024), nullable=False),
            Column('added_timestamp', DateTime, nullable=False, default=datetime.utcnow)
        )
        
        # Define feature_status table
        self.feature_status = Table(
            'feature_status',
            self.metadata,
            Column('status_id', Integer, primary_key=True, autoincrement=True),
            Column('task_id', Integer, nullable=False),
            Column('status', Enum('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', name='status_enum'), nullable=False),
            Column('start_time', DateTime, nullable=True),
            Column('end_time', DateTime, nullable=True),
            Column('hostname', String(255), nullable=True),
            Column('error_message', Text, nullable=True)
        )

        # New: pipeline_runs table
        self.pipeline_runs = Table(
            'pipeline_runs',
            self.metadata,
            Column('run_id', Integer, primary_key=True, autoincrement=True),
            Column('started_at', DateTime, nullable=False, default=datetime.utcnow),
            Column('ended_at', DateTime, nullable=True),
            Column('hostname', String(255), nullable=True),
            Column('dashboard_url', String(1024), nullable=True),
            Column('git_sha', String(64), nullable=True),
            Column('status', Enum('RUNNING', 'COMPLETED', 'FAILED', 'ABORTED', name='run_status_enum'), nullable=False, default='RUNNING'),
            Column('config_snapshot', Text, nullable=True)
        )

        # New: engine_stage_events table
        self.engine_stage_events = Table(
            'engine_stage_events',
            self.metadata,
            Column('stage_id', Integer, primary_key=True, autoincrement=True),
            Column('run_id', Integer, nullable=True),
            Column('task_id', Integer, nullable=True),
            Column('engine_name', String(64), nullable=False),
            Column('status', Enum('START', 'END', 'ERROR', name='stage_status_enum'), nullable=False),
            Column('start_time', DateTime, nullable=True),
            Column('end_time', DateTime, nullable=True),
            Column('rows_before', Integer, nullable=True),
            Column('rows_after', Integer, nullable=True),
            Column('cols_before', Integer, nullable=True),
            Column('cols_after', Integer, nullable=True),
            Column('new_cols', Integer, nullable=True),
            Column('hostname', String(255), nullable=True),
            Column('message', String(1024), nullable=True),
            Column('error_message', Text, nullable=True),
            Column('details', Text, nullable=True)
        )

        # New: task_metrics table (key/value by stage)
        self.task_metrics = Table(
            'task_metrics',
            self.metadata,
            Column('metric_id', Integer, primary_key=True, autoincrement=True),
            Column('run_id', Integer, nullable=True),
            Column('task_id', Integer, nullable=True),
            Column('stage', String(64), nullable=False),
            Column('key', String(128), nullable=False),
            Column('value_text', Text, nullable=True),
            Column('value_float', String(64), nullable=True),
            Column('created_at', DateTime, nullable=False, default=datetime.utcnow)
        )

        # New: task_artifacts table (references to saved JSONs/files)
        self.task_artifacts = Table(
            'task_artifacts',
            self.metadata,
            Column('artifact_id', Integer, primary_key=True, autoincrement=True),
            Column('run_id', Integer, nullable=True),
            Column('task_id', Integer, nullable=True),
            Column('stage', String(64), nullable=False),
            Column('path', String(1024), nullable=False),
            Column('kind', String(64), nullable=True),
            Column('meta', Text, nullable=True),
            Column('created_at', DateTime, nullable=False, default=datetime.utcnow)
        )
    
    def connect(self) -> bool:
        """
        Connect to the database using SQLAlchemy.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Create database URL
            db_url = self.settings.database.get_url()

            def _mk_engine(url: str) -> Engine:
                return create_engine(
                    url,
                    poolclass=QueuePool,
                    pool_size=self.settings.database.pool_size,
                    max_overflow=self.settings.database.max_overflow,
                    pool_timeout=self.settings.database.pool_timeout,
                    pool_recycle=self.settings.database.pool_recycle,
                    echo=False,
                )

            # Try connect to target database
            try:
                self.engine = _mk_engine(db_url)
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
            except OperationalError as e:
                msg = str(e).lower()
                # Unknown database: attempt to create it then reconnect
                if 'unknown database' in msg or 'does not exist' in msg or '1049' in msg:
                    logger.warning("Database not found; attempting to create it...")
                    # Build server-level URL (no database)
                    db = self.settings.database
                    server_url = (
                        f"mysql+pymysql://{db.username}:{db.password}@{db.host}:{db.port}/?charset={db.charset}"
                    )
                    server_engine = _mk_engine(server_url)
                    try:
                        with server_engine.connect() as sconn:
                            sconn.execute(text(
                                f"CREATE DATABASE IF NOT EXISTS `{db.database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                            ))
                            logger.info(f"Database '{db.database}' created/verified.")
                    finally:
                        server_engine.dispose()
                    # Recreate engine to target DB
                    self.engine = _mk_engine(db_url)
                    with self.engine.connect() as conn:
                        conn.execute(text("SELECT 1"))
                else:
                    raise

            # Create tables if they don't exist
            self.create_tables()

            logger.info(
                f"Database connection established to {self.settings.database.host}:{self.settings.database.port}/{self.settings.database.database}"
            )
            return True

        except OperationalError as e:
            logger.error(f"Database connection failed: {e}")
            return False
        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemy error during connection: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during database connection: {e}")
            return False

    def create_tables(self):
        """Create database tables if they don't exist."""
        try:
            self.metadata.create_all(self.engine)
            logger.info("Database tables created/verified successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise

    # ------------------ Run lifecycle APIs ------------------
    def create_run(self, hostname: Optional[str] = None, dashboard_url: Optional[str] = None,
                   git_sha: Optional[str] = None, config_snapshot: Optional[str] = None) -> Optional[int]:
        if not self.engine:
            logger.error("Database not connected. Call connect() first.")
            return None
        if not getattr(self.settings.monitoring, 'metrics_enabled', True):
            return None
        try:
            with self.engine.begin() as conn:
                stmt = self.pipeline_runs.insert().values(
                    started_at=datetime.utcnow(), hostname=hostname, dashboard_url=dashboard_url,
                    git_sha=git_sha, status='RUNNING', config_snapshot=config_snapshot
                )
                res = conn.execute(stmt)
                run_id = int(res.inserted_primary_key[0])
                logger.info(f"Created pipeline run {run_id}")
                return run_id
        except Exception as e:
            logger.error(f"Error creating pipeline run: {e}")
            return None

    def end_run(self, run_id: int, status: str = 'COMPLETED') -> bool:
        if not self.engine:
            return False
        if not run_id:
            return False
        if not getattr(self.settings.monitoring, 'metrics_enabled', True):
            return True
        try:
            with self.engine.begin() as conn:
                stmt = self.pipeline_runs.update().where(self.pipeline_runs.c.run_id == run_id).values(
                    ended_at=datetime.utcnow(), status=status
                )
                conn.execute(stmt)
                logger.info(f"Ended pipeline run {run_id} with status={status}")
                return True
        except Exception as e:
            logger.error(f"Error ending pipeline run {run_id}: {e}")
            return False

    # ------------------ Stage events APIs ------------------
    def start_stage(self, run_id: Optional[int], task_id: Optional[int], engine_name: str,
                    rows_before: Optional[int] = None, cols_before: Optional[int] = None,
                    message: Optional[str] = None) -> Optional[int]:
        if not self.engine:
            return None
        if not getattr(self.settings.monitoring, 'metrics_enabled', True):
            return None
        try:
            with self.engine.begin() as conn:
                stmt = self.engine_stage_events.insert().values(
                    run_id=run_id, task_id=task_id, engine_name=engine_name, status='START',
                    start_time=datetime.utcnow(), rows_before=rows_before, cols_before=cols_before,
                    hostname=self.hostname, message=message
                )
                res = conn.execute(stmt)
                stage_id = int(res.inserted_primary_key[0])
                return stage_id
        except Exception as e:
            logger.error(f"Error starting stage {engine_name} for task {task_id}: {e}")
            return None

    def end_stage(self, stage_id: int, rows_after: Optional[int] = None, cols_after: Optional[int] = None,
                  new_cols: Optional[int] = None, details: Optional[str] = None) -> bool:
        if not self.engine:
            return False
        if not getattr(self.settings.monitoring, 'metrics_enabled', True):
            return True
        try:
            with self.engine.begin() as conn:
                stmt = self.engine_stage_events.update().where(self.engine_stage_events.c.stage_id == stage_id).values(
                    status='END', end_time=datetime.utcnow(), rows_after=rows_after, cols_after=cols_after,
                    new_cols=new_cols, details=details
                )
                conn.execute(stmt)
                return True
        except Exception as e:
            logger.error(f"Error ending stage_id {stage_id}: {e}")
            return False

    def error_stage(self, stage_id: int, error_message: str, details: Optional[str] = None) -> bool:
        if not self.engine:
            return False

    # ------------------ Metrics & Artifacts ------------------
    def add_metrics(self, run_id: Optional[int], task_id: Optional[int], stage: str, metrics: Dict[str, Any]) -> bool:
        if not self.engine:
            return False
        if not getattr(self.settings.monitoring, 'metrics_enabled', True):
            return True
        try:
            with self.engine.begin() as conn:
                rows = []
                import json as _json
                for k, v in (metrics or {}).items():
                    val_text = None
                    val_float = None
                    # Coerce
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        val_float = str(float(v))
                    else:
                        try:
                            val_text = _json.dumps(v, ensure_ascii=False)
                        except Exception:
                            val_text = str(v)
                    rows.append({'run_id': run_id, 'task_id': task_id, 'stage': stage, 'key': str(k), 'value_text': val_text, 'value_float': val_float})
                if rows:
                    conn.execute(self.task_metrics.insert(), rows)
            return True
        except Exception as e:
            logger.warning(f"add_metrics failed: {e}")
            return False

    def add_artifact(self, run_id: Optional[int], task_id: Optional[int], stage: str, path: str, kind: str = 'file', meta: Optional[Dict[str, Any]] = None) -> bool:
        if not self.engine:
            return False
        if not getattr(self.settings.monitoring, 'metrics_enabled', True):
            return True
        try:
            with self.engine.begin() as conn:
                import json as _json
                meta_json = None
                if meta is not None:
                    try:
                        meta_json = _json.dumps(meta, ensure_ascii=False)
                    except Exception:
                        meta_json = str(meta)
                conn.execute(self.task_artifacts.insert().values(
                    run_id=run_id, task_id=task_id, stage=stage, path=path, kind=kind, meta=meta_json
                ))
            return True
        except Exception as e:
            logger.warning(f"add_artifact failed: {e}")
            return False
        if not getattr(self.settings.monitoring, 'metrics_enabled', True):
            return True
        try:
            with self.engine.begin() as conn:
                stmt = self.engine_stage_events.update().where(self.engine_stage_events.c.stage_id == stage_id).values(
                    status='ERROR', end_time=datetime.utcnow(), error_message=error_message, details=details
                )
                conn.execute(stmt)
                return True
        except Exception as e:
            logger.error(f"Error marking error for stage_id {stage_id}: {e}")
            return False

    def clear_old_records(self):
        """Clear old processing records to start fresh."""
        try:
            with self.engine.begin() as conn:
                # Clear feature_status table
                conn.execute(text("DELETE FROM feature_status"))
                # Clear processing_tasks table
                conn.execute(text("DELETE FROM processing_tasks"))
                logger.info("Cleared old processing records")
        except Exception as e:
            logger.error(f"Error clearing old records: {e}")
            raise
    
    def get_pending_currency_pairs(self) -> List[Dict[str, Any]]:
        """
        Fetch the list of pending currency pairs using SELECT ... NOT IN ... query.
        
        Returns:
            List[Dict[str, Any]]: List of pending tasks with currency pair and task info
        """
        if not self.engine:
            logger.error("Database not connected. Call connect() first.")
            return []
        
        try:
            with self.engine.connect() as conn:
                # Query to get tasks that are not completed
                query = text("""
                    SELECT 
                        pt.task_id,
                        pt.currency_pair,
                        pt.r2_path,
                        pt.added_timestamp,
                        COALESCE(fs.status, 'PENDING') as current_status,
                        fs.start_time,
                        fs.end_time,
                        fs.hostname,
                        fs.error_message
                    FROM processing_tasks pt
                    LEFT JOIN feature_status fs ON pt.task_id = fs.task_id
                    WHERE fs.status IS NULL 
                       OR fs.status IN ('PENDING', 'FAILED')
                       OR (fs.status = 'RUNNING' AND fs.start_time < DATE_SUB(NOW(), INTERVAL 1 HOUR))
                    ORDER BY pt.added_timestamp ASC
                """)
                
                result = conn.execute(query)
                pending_tasks = []
                
                for row in result:
                    task_dict = {
                        'task_id': row.task_id,
                        'currency_pair': row.currency_pair,
                        'r2_path': row.r2_path,
                        'added_timestamp': row.added_timestamp,
                        'current_status': row.current_status,
                        'start_time': row.start_time,
                        'end_time': row.end_time,
                        'hostname': row.hostname,
                        'error_message': row.error_message
                    }
                    pending_tasks.append(task_dict)
                
                logger.info(f"Found {len(pending_tasks)} pending currency pairs")
                return pending_tasks
                
        except SQLAlchemyError as e:
            logger.error(f"Error fetching pending currency pairs: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching pending currency pairs: {e}")
            return []
    
    def update_task_status(self, task_id: int, status: str, error_message: Optional[str] = None) -> bool:
        """
        Update the status of a task to RUNNING, COMPLETED, or FAILED.
        
        Args:
            task_id: The task ID to update
            status: The new status ('RUNNING', 'COMPLETED', or 'FAILED')
            error_message: Optional error message for FAILED status
            
        Returns:
            bool: True if update successful, False otherwise
        """
        if not self.engine:
            logger.error("Database not connected. Call connect() first.")
            return False
        
        if status not in ['RUNNING', 'COMPLETED', 'FAILED']:
            logger.error(f"Invalid status: {status}. Must be one of: RUNNING, COMPLETED, FAILED")
            return False
        
        try:
            with self.engine.begin() as conn:
                current_time = datetime.utcnow()
                
                if status == 'RUNNING':
                    # Insert or update status to RUNNING
                    query = text("""
                        INSERT INTO feature_status (task_id, status, start_time, hostname)
                        VALUES (:task_id, 'RUNNING', :start_time, :hostname)
                        ON DUPLICATE KEY UPDATE 
                            status = 'RUNNING',
                            start_time = :start_time,
                            hostname = :hostname,
                            error_message = NULL
                    """)
                    
                    conn.execute(query, {
                        'task_id': task_id,
                        'start_time': current_time,
                        'hostname': self.hostname
                    })
                    
                elif status == 'COMPLETED':
                    # Update status to COMPLETED
                    query = text("""
                        UPDATE feature_status 
                        SET status = 'COMPLETED', 
                            end_time = :end_time
                        WHERE task_id = :task_id
                    """)
                    
                    conn.execute(query, {
                        'task_id': task_id,
                        'end_time': current_time
                    })
                    
                elif status == 'FAILED':
                    # Update status to FAILED with error message
                    query = text("""
                        UPDATE feature_status 
                        SET status = 'FAILED', 
                            end_time = :end_time,
                            error_message = :error_message
                        WHERE task_id = :task_id
                    """)
                    
                    conn.execute(query, {
                        'task_id': task_id,
                        'end_time': current_time,
                        'error_message': error_message or 'Unknown error'
                    })
                
                logger.info(f"Updated task {task_id} status to {status}")
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Error updating task {task_id} status to {status}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error updating task {task_id} status: {e}")
            return False
    
    def get_task_info(self, task_id: int) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific task.
        
        Args:
            task_id: The task ID to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: Task information or None if not found
        """
        if not self.engine:
            logger.error("Database not connected. Call connect() first.")
            return None
        
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT 
                        pt.task_id,
                        pt.currency_pair,
                        pt.r2_path,
                        pt.added_timestamp,
                        fs.status,
                        fs.start_time,
                        fs.end_time,
                        fs.hostname,
                        fs.error_message
                    FROM processing_tasks pt
                    LEFT JOIN feature_status fs ON pt.task_id = fs.task_id
                    WHERE pt.task_id = :task_id
                """)
                
                result = conn.execute(query, {'task_id': task_id})
                row = result.fetchone()
                
                if row:
                    return {
                        'task_id': row.task_id,
                        'currency_pair': row.currency_pair,
                        'r2_path': row.r2_path,
                        'added_timestamp': row.added_timestamp,
                        'status': row.status,
                        'start_time': row.start_time,
                        'end_time': row.end_time,
                        'hostname': row.hostname,
                        'error_message': row.error_message
                    }
                else:
                    logger.warning(f"Task {task_id} not found")
                    return None
                    
        except SQLAlchemyError as e:
            logger.error(f"Error retrieving task {task_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error retrieving task {task_id}: {e}")
            return None
    
    def close(self):
        """Close the database connection and dispose of the engine."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        if not self.connect():
            raise RuntimeError("Failed to connect to database")
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
        if not self.engine:
            logger.error("Database not connected. Call connect() first.")
            return False
        
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT fs.status 
                    FROM processing_tasks pt
                    JOIN feature_status fs ON pt.task_id = fs.task_id
                    WHERE pt.currency_pair = :currency_pair
                    ORDER BY fs.status_id DESC
                    LIMIT 1
                """)
                
                result = conn.execute(query, {'currency_pair': currency_pair})
                row = result.fetchone()
                
                if row and row[0] == 'COMPLETED':
                    logger.info(f"Currency pair {currency_pair} already processed")
                    return True
                else:
                    logger.info(f"Currency pair {currency_pair} not yet processed")
                    return False
                    
        except SQLAlchemyError as e:
            logger.error(f"Error checking processing status for {currency_pair}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking processing status for {currency_pair}: {e}")
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
        if not self.engine:
            logger.error("Database not connected. Call connect() first.")
            return None
        
        try:
            with self.engine.begin() as conn:
                # Insert new task
                insert_query = text("""
                    INSERT INTO processing_tasks (currency_pair, r2_path, added_timestamp)
                    VALUES (:currency_pair, :data_path, :timestamp)
                    ON DUPLICATE KEY UPDATE 
                        r2_path = :data_path,
                        added_timestamp = :timestamp
                """)
                
                result = conn.execute(insert_query, {
                    'currency_pair': currency_pair,
                    'data_path': data_path,
                    'timestamp': datetime.utcnow()
                })
                
                # Get the task ID
                task_id_query = text("""
                    SELECT task_id FROM processing_tasks 
                    WHERE currency_pair = :currency_pair
                """)
                
                task_result = conn.execute(task_id_query, {'currency_pair': currency_pair})
                task_row = task_result.fetchone()
                
                if task_row:
                    task_id = task_row[0]
                    logger.info(f"Registered currency pair {currency_pair} with task ID {task_id}")
                    return task_id
                else:
                    logger.error(f"Failed to get task ID for {currency_pair}")
                    return None
                    
        except SQLAlchemyError as e:
            logger.error(f"Error registering currency pair {currency_pair}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error registering currency pair {currency_pair}: {e}")
            return None

    def get_processed_features(self, currency_pair: str) -> Optional[Dict[str, Any]]:
        """
        Get information about features that were generated for a currency pair.
        
        Args:
            currency_pair: The currency pair to check
            
        Returns:
            Optional[Dict]: Feature information if processed, None otherwise
        """
        if not self.engine:
            logger.error("Database not connected. Call connect() first.")
            return None
        
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT pt.task_id, pt.currency_pair, pt.r2_path,
                           fs.status, fs.start_time, fs.end_time, fs.hostname
                    FROM processing_tasks pt
                    JOIN feature_status fs ON pt.task_id = fs.task_id
                    WHERE pt.currency_pair = :currency_pair
                    AND fs.status = 'COMPLETED'
                    ORDER BY fs.status_id DESC
                    LIMIT 1
                """)
                
                result = conn.execute(query, {'currency_pair': currency_pair})
                row = result.fetchone()
                
                if row:
                    return {
                        'task_id': row[0],
                        'currency_pair': row[1],
                        'data_path': row[2],
                        'status': row[3],
                        'start_time': row[4],
                        'end_time': row[5],
                        'hostname': row[6]
                    }
                else:
                    return None
                    
        except SQLAlchemyError as e:
            logger.error(f"Error getting processed features for {currency_pair}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting processed features for {currency_pair}: {e}")
            return None


# Convenience functions for direct use
def get_pending_currency_pairs() -> List[Dict[str, Any]]:
    """
    Convenience function to get pending currency pairs.
    
    Returns:
        List[Dict[str, Any]]: List of pending tasks
    """
    with DatabaseHandler() as db:
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
    with DatabaseHandler() as db:
        return db.update_task_status(task_id, status, error_message)
