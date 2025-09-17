"""
Timestamp utilities for consistent datetime handling across the pipeline.

This module provides utilities to normalize timestamp dtypes to avoid
merge warnings and ensure consistent datetime handling.
"""

import logging
from typing import Union, Optional
import dask_cudf
import cudf

logger = logging.getLogger(__name__)


def normalize_timestamp_dtypes(
    df1: Union[dask_cudf.DataFrame, cudf.DataFrame], 
    df2: Union[dask_cudf.DataFrame, cudf.DataFrame],
    timestamp_col: str = 'timestamp',
    target_dtype: str = 'datetime64[ns]'
) -> tuple:
    """
    Normalize timestamp column dtypes in two DataFrames to avoid merge warnings.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame  
        timestamp_col: Name of the timestamp column
        target_dtype: Target dtype for normalization (default: 'datetime64[ns]')
        
    Returns:
        Tuple of (df1_normalized, df2_normalized)
    """
    try:
        # Check if timestamp column exists in both DataFrames
        if timestamp_col in df1.columns and timestamp_col in df2.columns:
            # Normalize both timestamp columns to the same dtype
            df1[timestamp_col] = df1[timestamp_col].astype(target_dtype)
            df2[timestamp_col] = df2[timestamp_col].astype(target_dtype)
            logger.debug(f"Normalized timestamp dtypes to {target_dtype}")
        else:
            logger.warning(f"Timestamp column '{timestamp_col}' not found in one or both DataFrames")
            
    except Exception as e:
        logger.warning(f"Could not normalize timestamp dtypes: {e}")
    
    return df1, df2


def safe_merge_with_timestamp(
    df1: Union[dask_cudf.DataFrame, cudf.DataFrame],
    df2: Union[dask_cudf.DataFrame, cudf.DataFrame], 
    on: str = 'timestamp',
    how: str = 'left',
    timestamp_col: str = 'timestamp',
    target_dtype: str = 'datetime64[ns]'
) -> Union[dask_cudf.DataFrame, cudf.DataFrame]:
    """
    Perform a merge with automatic timestamp dtype normalization.
    
    Args:
        df1: Left DataFrame
        df2: Right DataFrame
        on: Column name to merge on
        how: Type of merge (default: 'left')
        timestamp_col: Name of the timestamp column
        target_dtype: Target dtype for normalization
        
    Returns:
        Merged DataFrame
    """
    # Normalize timestamp dtypes before merge
    df1_norm, df2_norm = normalize_timestamp_dtypes(df1, df2, timestamp_col, target_dtype)
    
    # Perform the merge
    return df1_norm.merge(df2_norm, on=on, how=how)

