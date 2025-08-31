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
from typing import Optional, Dict, Any, List

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
            currency_pair_path: The relative path to the currency pair data file,
                                which will be joined with the local data root.
            client: Dask client for distributed loading.

        Returns:
            Optional[dask_cudf.DataFrame]: Loaded data as dask_cudf DataFrame, None if failed.
        """
        try:
            local_path = self._get_local_path(currency_pair_path)
            logger.info(f"Attempting to load data from local path: {local_path}")

            if not local_path.exists():
                logger.error(f"Local data file does not exist: {local_path}")
                return None

            # Check if it's a file (not directory)
            if not local_path.is_file():
                logger.error(f"Local data path is not a file: {local_path}")
                return None

            # Read the single parquet file
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
        Load currency pair data from the local disk synchronously (single GPU).

        Args:
            currency_pair_path: The relative path to the currency pair data file.

        Returns:
            Optional[cudf.DataFrame]: Loaded data as cudf DataFrame, None if failed.
        """
        try:
            local_path = self._get_local_path(currency_pair_path)
            logger.info(f"Loading data synchronously from: {local_path}")

            if not local_path.exists():
                logger.error(f"Local data file does not exist: {local_path}")
                return None

            if not local_path.is_file():
                logger.error(f"Local data path is not a file: {local_path}")
                return None

            # Read the single parquet file
            df = cudf.read_parquet(local_path)

            logger.info(f"Successfully loaded data from {local_path}")
            logger.info(f"Data shape: {df.shape}")
            logger.info(f"Data types: {df.dtypes.to_dict()}")

            return df

        except Exception as e:
            logger.error(f"Failed to load data from {currency_pair_path}: {e}", exc_info=True)
            return None

    def load_currency_pair_data_feather(
        self,
        currency_pair_path: str,
        client: Client
    ) -> Optional[dask_cudf.DataFrame]:
        """
        Load currency pair data from Feather v2 files to GPU memory.

        Args:
            currency_pair_path: The relative path to the currency pair data,
                                which will be joined with the local data root.
            client: Dask client for distributed loading.

        Returns:
            Optional[dask_cudf.DataFrame]: Loaded data as dask_cudf DataFrame, None if failed.
        """
        try:
            local_path = self._get_local_path(currency_pair_path)
            logger.info(f"Attempting to load Feather v2 data from local path: {local_path}")

            if not local_path.exists() or not local_path.is_dir():
                logger.error(f"Local data path does not exist or is not a directory: {local_path}")
                return None

            # Check for Feather v2 files
            feather_files = list(local_path.glob("*.feather"))
            if not feather_files:
                logger.error(f"No Feather v2 files found in: {local_path}")
                return None

            # If there's a consolidated file, use it
            consolidated_file = local_path / f"{local_path.name}.feather"
            if consolidated_file.exists():
                logger.info(f"Loading consolidated Feather v2 file: {consolidated_file}")
                df = dask_cudf.read_feather(consolidated_file)
            else:
                # Load partitioned files
                logger.info(f"Loading {len(feather_files)} partitioned Feather v2 files")
                df = dask_cudf.read_feather(local_path)

            logger.info("Distributing data across GPU cluster...")
            df = df.persist()
            client.wait(df)

            num_partitions = df.npartitions
            logger.info(f"Successfully loaded Feather v2 data from {local_path} with {num_partitions} partitions")

            sample_df = df.head()
            logger.info(f"Data schema: {sample_df.columns.tolist()}")
            logger.info(f"Data types: {sample_df.dtypes.to_dict()}")

            return df

        except Exception as e:
            logger.error(f"Failed to load Feather v2 data from {currency_pair_path}: {e}", exc_info=True)
            return None

    def load_currency_pair_data_feather_sync(
        self,
        currency_pair_path: str
    ) -> Optional[cudf.DataFrame]:
        """
        Load currency pair data from Feather v2 files synchronously (single GPU).

        Args:
            currency_pair_path: The relative path to the currency pair data.

        Returns:
            Optional[cudf.DataFrame]: Loaded data as cudf DataFrame, None if failed.
        """
        try:
            local_path = self._get_local_path(currency_pair_path)
            logger.info(f"Loading Feather v2 data synchronously from: {local_path}")

            if not local_path.exists() or not local_path.is_dir():
                logger.error(f"Local data path does not exist or is not a directory: {local_path}")
                return None

            # Check for Feather v2 files
            feather_files = list(local_path.glob("*.feather"))
            if not feather_files:
                logger.error(f"No Feather v2 files found in: {local_path}")
                return None

            # If there's a consolidated file, use it
            consolidated_file = local_path / f"{local_path.name}.feather"
            if consolidated_file.exists():
                logger.info(f"Loading consolidated Feather v2 file: {consolidated_file}")
                df = cudf.read_feather(consolidated_file)
            else:
                # Load partitioned files and concatenate
                logger.info(f"Loading and concatenating {len(feather_files)} partitioned Feather v2 files")
                import glob
                
                # Get all feather files in order
                part_files = sorted(glob.glob(str(local_path / "part-*.feather")))
                if not part_files:
                    logger.error(f"No partition files found in: {local_path}")
                    return None
                
                # Load and concatenate all partitions
                dfs = []
                for part_file in part_files:
                    part_df = cudf.read_feather(part_file)
                    dfs.append(part_df)
                
                df = cudf.concat(dfs, ignore_index=True)

            logger.info(f"Successfully loaded Feather v2 data from {local_path}")
            logger.info(f"Data shape: {df.shape}")
            logger.info(f"Data types: {df.dtypes.to_dict()}")

            return df

        except Exception as e:
            logger.error(f"Failed to load Feather v2 data from {currency_pair_path}: {e}", exc_info=True)
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

    def discover_currency_pairs(self) -> List[Dict[str, Any]]:
        """
        Discover all available currency pairs in the local data directory.
        
        Returns:
            List[Dict]: List of currency pair information with paths
        """
        try:
            currency_pairs = []
            data_root = Path(self.local_data_root)
            
            if not data_root.exists():
                logger.error(f"Local data root does not exist: {data_root}")
                return []
            
            logger.info(f"Scanning for currency pairs in: {data_root}")
            
            # Look for parquet or feather files directly in the data directory
            parquet_files = list(data_root.glob("*.parquet"))
            feather_files = list(data_root.glob("*.feather"))
            
            logger.info(f"Found {len(parquet_files)} parquet files and {len(feather_files)} feather files")
            
            # Process parquet files
            for parquet_file in parquet_files:
                # Extract currency pair name from filename (e.g., "AUDUSD_master_features.parquet" -> "AUDUSD")
                filename = parquet_file.stem  # Remove extension
                
                # Try to extract currency pair from filename
                currency_pair = None
                
                # Pattern 1: "AUDUSD_master_features" -> "AUDUSD"
                if '_' in filename:
                    currency_pair = filename.split('_')[0].upper()
                # Pattern 2: "AUDUSD" -> "AUDUSD"
                else:
                    currency_pair = filename.upper()
                
                # Validate currency pair format (should be 6 characters like EURUSD)
                if len(currency_pair) == 6 and currency_pair.isalpha():
                    pair_info = {
                        'currency_pair': currency_pair,
                        'data_path': str(parquet_file.relative_to(data_root)),
                        'file_type': 'parquet',
                        'file_size_mb': parquet_file.stat().st_size / (1024 * 1024),
                        'filename': parquet_file.name
                    }
                    currency_pairs.append(pair_info)
                    logger.info(f"Found currency pair: {currency_pair} from file {parquet_file.name}")
                else:
                    logger.warning(f"Skipping invalid currency pair format from file {parquet_file.name}: {currency_pair}")
            
            # Process feather files
            for feather_file in feather_files:
                # Extract currency pair name from filename
                filename = feather_file.stem
                
                currency_pair = None
                if '_' in filename:
                    currency_pair = filename.split('_')[0].upper()
                else:
                    currency_pair = filename.upper()
                
                if len(currency_pair) == 6 and currency_pair.isalpha():
                    pair_info = {
                        'currency_pair': currency_pair,
                        'data_path': str(feather_file.relative_to(data_root)),
                        'file_type': 'feather',
                        'file_size_mb': feather_file.stat().st_size / (1024 * 1024),
                        'filename': feather_file.name
                    }
                    currency_pairs.append(pair_info)
                    logger.info(f"Found currency pair: {currency_pair} from file {feather_file.name}")
                else:
                    logger.warning(f"Skipping invalid currency pair format from file {feather_file.name}: {currency_pair}")
            
            logger.info(f"Discovered {len(currency_pairs)} currency pairs")
            return currency_pairs
            
        except Exception as e:
            logger.error(f"Error discovering currency pairs: {e}", exc_info=True)
            return []

    def get_currency_pair_info(self, currency_pair: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific currency pair.

        Args:
            currency_pair: The 6-character currency pair (e.g., 'EURUSD').

        Returns:
            Optional[Dict[str, Any]]: Detailed information about the currency pair.
        """
        try:
            data_root = Path(self.local_data_root)
            currency_pair_path = f"{currency_pair}/"
            local_path = data_root / currency_pair_path

            if not local_path.exists() or not local_path.is_dir():
                logger.warning(f"Currency pair path does not exist: {local_path}")
                return None

            # Get all parquet files
            parquet_files = list(local_path.glob("*.parquet"))

            if not parquet_files:
                logger.warning(f"No parquet files found for currency pair: {currency_pair}")
                return None

            # Get basic info from first file
            first_file = parquet_files[0]
            sample_df = cudf.read_parquet(first_file, nrows=1)

            info = {
                "currency_pair": currency_pair,
                "path": str(local_path),
                "num_files": len(parquet_files),
                "columns": sample_df.columns.tolist(),
                "dtypes": sample_df.dtypes.to_dict(),
                "total_size_mb": sum(f.stat().st_size for f in parquet_files) / (1024 * 1024)
            }
            return info

        except Exception as e:
            logger.error(f"Error getting currency pair info for {currency_pair}: {e}", exc_info=True)
            return None


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
