"""
Local Data Loader for Dynamic Stage 0 Pipeline

This module handles loading currency pair data from the local filesystem
to GPU memory using dask_cudf for high-performance data ingestion.
The data is expected to be synced from R2 to a local directory
by an external script (e.g., onstart.sh).
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any

import dask_cudf
import cudf
from dask.distributed import Client

from config.settings import get_settings

logger = logging.getLogger(__name__)


class LocalDataLoader:
    """Handles data loading from the local filesystem to GPU memory."""

    def __init__(self):
        """Initialize the Local data loader with configuration."""
        self.settings = get_settings()
        self.local_data_root = "/data" # Matches the sync destination in onstart.sh

    def _get_local_path(self, r2_path: str) -> Path:
        """
        Construct the full local path for a currency pair's data.

        Args:
            r2_path: The original base path from R2 (e.g., 'data/forex/EURUSD/').
                     We'll use this to find the corresponding local directory.

        Returns:
            Path object for the local data directory.
        """
        # r2_path might be like 'data/forex/EURUSD/'. We want to join it to our local root.
        # Path.joinpath handles slashes correctly.
        return Path(self.local_data_root) / r2_path

    def load_currency_pair_data(
        self,
        currency_pair_path: str,
        client: Client
    ) -> Optional[dask_cudf.DataFrame]:
        """
        Load currency pair data from the local disk to GPU memory.

        Args:
            currency_pair_path: The relative path to the currency pair data,
                                which will be joined with the local data root.
            client: Dask client for distributed loading.

        Returns:
            Optional[dask_cudf.DataFrame]: Loaded data as dask_cudf DataFrame, None if failed.
        """
        try:
            local_path = self._get_local_path(currency_pair_path)
            logger.info(f"Attempting to load data from local path: {local_path}")

            if not local_path.exists() or not local_path.is_dir():
                logger.error(f"Local data path does not exist or is not a directory: {local_path}")
                return None

            # Read all parquet files in the directory
            df = dask_cudf.read_parquet(local_path)

            logger.info("Distributing data across GPU cluster...")
            df = df.persist()
            client.wait(df)

            num_partitions = df.npartitions
            logger.info(f"Successfully loaded data from {local_path} with {num_partitions} partitions")

            sample_df = df.head()
            logger.info(f"Data schema: {sample_df.columns.tolist()}")
            logger.info(f"Data types: {sample_df.dtypes.to_dict()}")

            return df

        except Exception as e:
            logger.error(f"Failed to load data from {currency_pair_path}: {e}", exc_info=True)
            return None

    def load_currency_pair_data_sync(
        self,
        currency_pair_path: str
    ) -> Optional[cudf.DataFrame]:
        """
        Load currency pair data synchronously (single GPU).

        Args:
            currency_pair_path: The relative path to the currency pair data.

        Returns:
            Optional[cudf.DataFrame]: Loaded data as cudf DataFrame, None if failed.
        """
        try:
            local_path = self._get_local_path(currency_pair_path)
            logger.info(f"Loading data synchronously from: {local_path}")

            if not local_path.exists() or not local_path.is_dir():
                logger.error(f"Local data path does not exist or is not a directory: {local_path}")
                return None

            # Read all parquet files in the directory
            df = cudf.read_parquet(local_path)

            logger.info(f"Successfully loaded {len(df)} rows from {local_path}")
            logger.info(f"Data schema: {df.columns.tolist()}")
            logger.info(f"Data types: {df.dtypes.to_dict()}")

            return df

        except Exception as e:
            logger.error(f"Failed to load data from {currency_pair_path}: {e}", exc_info=True)
            return None

    def validate_data_path(self, currency_pair_path: str) -> bool:
        """
        Validate that the data path exists and contains parquet files.

        Args:
            currency_pair_path: The relative path to validate.

        Returns:
            bool: True if path exists and contains parquet files.
        """
        try:
            local_path = self._get_local_path(currency_pair_path)
            
            if not local_path.exists():
                logger.warning(f"Local path does not exist: {local_path}")
                return False
            
            if not local_path.is_dir():
                logger.warning(f"Local path is not a directory: {local_path}")
                return False
            
            # Check for parquet files
            parquet_files = list(local_path.glob("*.parquet"))
            if not parquet_files:
                logger.warning(f"No parquet files found in: {local_path}")
                return False
            
            logger.info(f"Found {len(parquet_files)} parquet files in {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating path {currency_pair_path}: {e}")
            return False

    def get_data_info(self, currency_pair_path: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about the data without loading it completely.

        Args:
            currency_pair_path: The relative path to the data.

        Returns:
            Optional[Dict[str, Any]]: Metadata about the data, None if failed.
        """
        try:
            local_path = self._get_local_path(currency_pair_path)
            
            if not local_path.exists() or not local_path.is_dir():
                return None
            
            # Get parquet files
            parquet_files = list(local_path.glob("*.parquet"))
            
            if not parquet_files:
                return None
            
            # Get basic info from first file
            first_file = parquet_files[0]
            sample_df = cudf.read_parquet(first_file, nrows=1)
            
            info = {
                "path": str(local_path),
                "num_files": len(parquet_files),
                "columns": sample_df.columns.tolist(),
                "dtypes": sample_df.dtypes.to_dict(),
                "file_size_mb": sum(f.stat().st_size for f in parquet_files) / (1024 * 1024)
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting data info for {currency_pair_path}: {e}")
            return None

    def list_available_pairs(self) -> list:
        """
        List all available currency pairs in the local data directory.

        Returns:
            list: List of available currency pair paths.
        """
        try:
            local_root = Path(self.local_data_root)
            
            if not local_root.exists():
                logger.warning(f"Local data root does not exist: {local_root}")
                return []
            
            available_pairs = []
            
            # Walk through the directory structure
            for item in local_root.rglob("*"):
                if item.is_dir() and any(item.glob("*.parquet")):
                    # Convert to relative path
                    relative_path = item.relative_to(local_root)
                    available_pairs.append(str(relative_path))
            
            logger.info(f"Found {len(available_pairs)} available currency pairs")
            return available_pairs
            
        except Exception as e:
            logger.error(f"Error listing available pairs: {e}")
            return []


# Convenience functions
def load_currency_pair_data(currency_pair_path: str, client: Client) -> Optional[dask_cudf.DataFrame]:
    """Convenience function to load currency pair data."""
    loader = LocalDataLoader()
    return loader.load_currency_pair_data(currency_pair_path, client)


def load_currency_pair_data_sync(currency_pair_path: str) -> Optional[cudf.DataFrame]:
    """Convenience function to load currency pair data synchronously."""
    loader = LocalDataLoader()
    return loader.load_currency_pair_data_sync(currency_pair_path)


def validate_data_path(currency_pair_path: str) -> bool:
    """Convenience function to validate data path."""
    loader = LocalDataLoader()
    return loader.validate_data_path(currency_pair_path)
