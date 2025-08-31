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

from config import get_config
from config.settings import get_settings

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
    
    def connect(self) -> bool:
        """
        Connect to the database using SQLAlchemy.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Create database URL
            db_url = self.config.get_database_url()
            
            # Create engine with connection pooling
            self.engine = create_engine(
                db_url,
                poolclass=QueuePool,
                pool_size=self.settings.database.pool_size,
                max_overflow=self.settings.database.max_overflow,
                pool_timeout=self.settings.database.pool_timeout,
                pool_recycle=self.settings.database.pool_recycle,
                echo=False  # Set to True for SQL debugging
            )
            
            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            
            # Create tables if they don't exist
            self.create_tables()
            
            logger.info(f"Database connection established to {self.settings.database.host}:{self.settings.database.port}")
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
