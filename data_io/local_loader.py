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
from dask.distributed import Client, wait
import cupy as cp

from config.unified_config import get_unified_config as get_settings

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

            # Do not persist here; keep lazy to allow column pruning later

            num_partitions = df.npartitions
            logger.info(f"Successfully loaded data from {local_path} with {num_partitions} partitions")

            sample_df = df.head()
            logger.info(f"Data schema: {len(sample_df.columns)} columns")
            # Removed verbose data types log

            return df

        except Exception as e:
            logger.error(f"Failed to load data from {currency_pair_path}: {e}", exc_info=True)
            return None

    def _process_cudf_in_chunks(
        self,
        df: cudf.DataFrame,
        chunk_size: int = 10000,
        overlap: int = 1000,
        max_memory_gb: float = 8.0
    ) -> cudf.DataFrame:
        """
        Process large cuDF DataFrames in chunks with overlap to avoid OOM.
        
        Args:
            df: Input cuDF DataFrame
            chunk_size: Number of rows per chunk
            overlap: Number of overlapping rows between chunks
            max_memory_gb: Maximum memory usage in GB
            
        Returns:
            Processed cuDF DataFrame
        """
        try:
            total_rows = len(df)
            logger.info(f"Processing DataFrame with {total_rows} rows in chunks of {chunk_size}")
            
            if total_rows <= chunk_size:
                logger.info("DataFrame fits in single chunk, processing directly")
                return df
            
            # Calculate optimal chunk size based on available memory
            available_memory = cp.cuda.runtime.memGetInfo()[0] / (1024**3)  # GB
            optimal_chunk_size = min(chunk_size, int(available_memory * max_memory_gb * 0.1))
            
            logger.info(f"Available GPU memory: {available_memory:.2f} GB")
            logger.info(f"Optimal chunk size: {optimal_chunk_size}")
            
            processed_chunks = []
            start_idx = 0
            
            while start_idx < total_rows:
                end_idx = min(start_idx + optimal_chunk_size, total_rows)
                
                # Extract chunk with overlap
                chunk_start = max(0, start_idx - overlap)
                chunk_end = end_idx
                
                logger.info(f"Processing chunk {len(processed_chunks) + 1}: rows {chunk_start}-{chunk_end}")
                
                # Extract chunk
                chunk = df.iloc[chunk_start:chunk_end].copy()
                
                # Process chunk (placeholder for feature computation)
                # This will be replaced by actual feature computation logic
                processed_chunk = chunk
                
                # Remove overlap from previous chunk (except first chunk)
                if len(processed_chunks) > 0:
                    processed_chunk = processed_chunk.iloc[overlap:]
                
                processed_chunks.append(processed_chunk)
                
                # Clear GPU memory
                del chunk
                cp.get_default_memory_pool().free_all_blocks()
                
                start_idx = end_idx
            
            # Concatenate all processed chunks
            if processed_chunks:
                result = cudf.concat(processed_chunks, ignore_index=True)
                logger.info(f"Successfully processed DataFrame in {len(processed_chunks)} chunks")
                return result
            else:
                logger.error("No chunks were processed")
                return df
                
        except Exception as e:
            logger.error(f"Error in chunked processing: {e}")
            return df

    def load_currency_pair_data_sync(self, r2_path: str) -> Optional[cudf.DataFrame]:
        """
        Load currency pair data synchronously with chunked processing for large datasets.
        
        Implements chunked processing to avoid OOM errors as specified in the technical plan.
        """
        try:
            self._log_info(f"Loading currency pair data from {r2_path}")
            
            # Check if file exists
            if not os.path.exists(r2_path):
                self._log_error(f"File not found: {r2_path}")
                return None
            
            # Get file size to determine if chunked processing is needed
            file_size = os.path.getsize(r2_path)
            file_size_gb = file_size / (1024**3)
            
            self._log_info(f"File size: {file_size_gb:.2f} GB")
            
            # Determine if chunked processing is needed based on file size and available memory
            if self._should_use_chunked_processing(file_size_gb):
                self._log_info("Using chunked processing to avoid OOM")
                return self._load_data_with_chunked_processing(r2_path)
            else:
                self._log_info("Loading entire dataset into GPU memory")
                return self._load_data_direct(r2_path)
                
        except Exception as e:
            self._critical_error(f"Error loading currency pair data: {e}")
    
    def _should_use_chunked_processing(self, file_size_gb: float) -> bool:
        """
        Determine if chunked processing should be used based on file size and available memory.
        """
        try:
            # Get available GPU memory
            gpu_memory = cp.cuda.runtime.memGetInfo()[0]  # Free memory in bytes
            available_memory_gb = gpu_memory / (1024**3)
            
            # Get threshold from settings (unified memory config)
            threshold = self.settings.memory.max_memory_gb
            
            # Use chunked processing if file size exceeds threshold
            should_chunk = file_size_gb > (available_memory_gb * threshold)
            
            self._log_info(f"Available GPU memory: {available_memory_gb:.2f} GB")
            self._log_info(f"Chunked processing threshold: {threshold}")
            self._log_info(f"Should use chunked processing: {should_chunk}")
            
            return should_chunk
            
        except Exception as e:
            self._log_warn(f"Error checking memory, defaulting to chunked processing: {e}")
            return True
    
    def _load_data_direct(self, r2_path: str) -> Optional[cudf.DataFrame]:
        """
        Load data directly into GPU memory (for smaller files).
        """
        try:
            # Try to load as cuDF first
            try:
                df = cudf.read_feather(r2_path)
                self._log_info(f"Loaded {len(df)} rows, {len(df.columns)} columns directly")
                return df
            except Exception as feather_error:
                self._log_warn(f"Failed to load as feather: {feather_error}")
                
                # Fallback to pandas then convert to cuDF
                try:
                    import pandas as pd
                    pdf = pd.read_feather(r2_path)
                    df = cudf.from_pandas(pdf)
                    self._log_info(f"Loaded via pandas: {len(df)} rows, {len(df.columns)} columns")
                    return df
                except Exception as pandas_error:
                    self._log_error(f"Failed to load via pandas: {pandas_error}")
                    return None
                    
        except Exception as e:
            self._critical_error(f"Error in direct data loading: {e}")
    
    def _load_data_with_chunked_processing(self, r2_path: str) -> Optional[cudf.DataFrame]:
        """
        Load data using chunked processing to avoid OOM errors.
        
        Implements the chunked processing strategy specified in the technical plan:
        - Process data in chunks with overlap
        - Use memory monitoring
        - Optimize for GPU memory constraints
        """
        try:
            self._log_info("Starting chunked data processing")
            
            # Get chunking parameters from settings (unified memory config)
            chunk_size = self.settings.memory.chunk_size
            overlap_size = self.settings.memory.chunk_overlap
            
            self._log_info(f"Chunk size: {chunk_size}, Overlap: {overlap_size}")
            
            # First, read the data structure to determine total size
            try:
                # Read just the first few rows to get column structure
                sample_df = cudf.read_feather(r2_path, nrows=100)
                total_columns = len(sample_df.columns)
                self._log_info(f"Detected {total_columns} columns")
                
                # Get total number of rows (this might be expensive for large files)
                # For now, we'll estimate based on file size
                file_size = os.path.getsize(r2_path)
                estimated_rows = self._estimate_total_rows(file_size, total_columns)
                
                self._log_info(f"Estimated total rows: {estimated_rows}")
                
                # Calculate number of chunks needed
                effective_chunk_size = chunk_size - overlap_size
                num_chunks = max(1, int(estimated_rows / effective_chunk_size) + 1)
                
                self._log_info(f"Will process in {num_chunks} chunks")
                
                # Process chunks
                all_chunks = []
                
                for chunk_idx in range(num_chunks):
                    self._log_info(f"Processing chunk {chunk_idx + 1}/{num_chunks}")
                    
                    # Calculate chunk boundaries
                    start_row = chunk_idx * effective_chunk_size
                    end_row = start_row + chunk_size
                    
                    # Read chunk
                    chunk_df = self._read_chunk(r2_path, start_row, end_row)
                    
                    if chunk_df is not None and len(chunk_df) > 0:
                        # Process chunk (apply any necessary transformations)
                        processed_chunk = self._process_chunk(chunk_df, chunk_idx, num_chunks)
                        all_chunks.append(processed_chunk)
                        
                        # Clear GPU memory
                        del chunk_df
                        cp.get_default_memory_pool().free_all_blocks()
                        
                        self._log_info(f"Chunk {chunk_idx + 1} processed: {len(processed_chunk)} rows")
                    else:
                        self._log_warn(f"Empty chunk {chunk_idx + 1}")
                
                # Combine all chunks
                if all_chunks:
                    final_df = cudf.concat(all_chunks, ignore_index=True)
                    self._log_info(f"Combined {len(all_chunks)} chunks into final dataset: {len(final_df)} rows")
                    return final_df
                else:
                    self._log_error("No valid chunks processed")
                    return None
                    
            except Exception as e:
                self._log_error(f"Error in chunked processing: {e}")
                # Fallback to direct loading
                return self._load_data_direct(r2_path)
                
        except Exception as e:
            self._critical_error(f"Error in chunked data loading: {e}")
    
    def _read_chunk(self, r2_path: str, start_row: int, end_row: int) -> Optional[cudf.DataFrame]:
        """
        Read a specific chunk of data from the file.
        """
        try:
            # For feather files, we can use skiprows and nrows
            # Note: This is a simplified approach - in practice, you might need more sophisticated chunking
            chunk_df = cudf.read_feather(r2_path, skiprows=start_row, nrows=end_row-start_row)
            return chunk_df
            
        except Exception as e:
            self._log_warn(f"Error reading chunk {start_row}-{end_row}: {e}")
            return None
    
    def _process_chunk(self, chunk_df: cudf.DataFrame, chunk_idx: int, total_chunks: int) -> cudf.DataFrame:
        """
        Process a single chunk of data.
        This is where you would apply any chunk-specific transformations.
        """
        try:
            # For now, just return the chunk as-is
            # In the future, you might want to apply specific transformations here
            return chunk_df
            
        except Exception as e:
            self._log_error(f"Error processing chunk {chunk_idx}: {e}")
            return chunk_df
    
    def _estimate_total_rows(self, file_size: int, num_columns: int) -> int:
        """
        Estimate total number of rows based on file size and number of columns.
        This is a rough estimation for planning chunk sizes.
        """
        try:
            # Rough estimation: assume 8 bytes per value (float64)
            bytes_per_row = num_columns * 8
            estimated_rows = file_size // bytes_per_row
            
            return max(estimated_rows, 1000)  # Minimum 1000 rows
            
        except Exception as e:
            self._log_warn(f"Error estimating rows, using default: {e}")
            return 100000  # Default estimate

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

            if not local_path.exists():
                logger.warning(f"Local data path does not exist: {local_path}")
                return None

            # Case 1: direct feather file
            if local_path.is_file() and local_path.suffix.lower() == ".feather":
                logger.info(f"Loading single Feather v2 file: {local_path}")
                df = dask_cudf.read_feather(local_path)
            # Case 2: directory containing feather parts or consolidated file
            elif local_path.is_dir():
                feather_files = list(local_path.glob("*.feather"))
                if not feather_files:
                    logger.warning(f"No Feather v2 files found in: {local_path}")
                    return None
                consolidated_file = local_path / f"{local_path.name}.feather"
                if consolidated_file.exists():
                    logger.info(f"Loading consolidated Feather v2 file: {consolidated_file}")
                    df = dask_cudf.read_feather(consolidated_file)
                else:
                    logger.info(f"Loading {len(feather_files)} partitioned Feather v2 files from directory")
                    df = dask_cudf.read_feather(local_path)
            else:
                logger.warning(f"Unsupported Feather path type: {local_path}")
                return None

            # Do not persist here; keep lazy to allow column pruning later

            num_partitions = df.npartitions
            logger.info(f"Successfully loaded Feather v2 data from {local_path} with {num_partitions} partitions")

            sample_df = df.head()
            logger.info(f"Data schema: {len(sample_df.columns)} columns")
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

            if not local_path.exists():
                logger.warning(f"Local data path does not exist: {local_path}")
                return None

            # Case 1: direct feather file
            if local_path.is_file() and local_path.suffix.lower() == ".feather":
                df = cudf.read_feather(local_path)
            # Case 2: directory with consolidated or parts
            elif local_path.is_dir():
                feather_files = list(local_path.glob("*.feather"))
                if not feather_files:
                    logger.warning(f"No Feather v2 files found in: {local_path}")
                    return None
                consolidated_file = local_path / f"{local_path.name}.feather"
                if consolidated_file.exists():
                    logger.info(f"Loading consolidated Feather v2 file: {consolidated_file}")
                    df = cudf.read_feather(consolidated_file)
                else:
                    import glob
                    part_files = sorted(glob.glob(str(local_path / "part-*.feather")))
                    if not part_files:
                        logger.warning(f"No partition files found in: {local_path}")
                        return None
                    dfs = [cudf.read_feather(p) for p in part_files]
                    df = cudf.concat(dfs, ignore_index=True)
            else:
                logger.warning(f"Unsupported Feather path type: {local_path}")
                return None

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
