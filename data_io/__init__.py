"""
Data I/O module for Dynamic Stage 0 pipeline.

This module handles all data input/output operations, including
R2/S3 connectivity, database operations, and data loading/saving.
"""

from .db_handler import DatabaseHandler, get_pending_currency_pairs, update_task_status
from .r2_loader import R2DataLoader, load_currency_pair_data as r2_load_currency_pair_data, load_currency_pair_data_sync as r2_load_currency_pair_data_sync, validate_data_path as r2_validate_data_path
from .local_loader import LocalDataLoader, load_currency_pair_data, load_currency_pair_data_sync, validate_data_path

__all__ = [
    'DatabaseHandler', 
    'get_pending_currency_pairs', 
    'update_task_status',
    'R2DataLoader',
    'LocalDataLoader',
    'load_currency_pair_data',
    'load_currency_pair_data_sync',
    'validate_data_path',
    'r2_load_currency_pair_data',
    'r2_load_currency_pair_data_sync',
    'r2_validate_data_path'
]
