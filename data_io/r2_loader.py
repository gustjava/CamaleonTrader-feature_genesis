"""
R2 Data Loader for Dynamic Stage 0 Pipeline

This module handles loading currency pair data from Cloudflare R2 storage
to GPU memory using dask_cudf for high-performance data ingestion.
"""

import logging
import os
from typing import Optional, Dict, Any, List
from pathlib import Path

import dask_cudf
import cudf
from dask.distributed import Client

from config import get_config
from config.settings import get_settings

logger = logging.getLogger(__name__)


class R2DataLoader:
    """Handles data loading from Cloudflare R2 to GPU memory."""
    
    def __init__(self):
        """Initialize the R2 data loader with configuration."""
        self.config = get_config()
        self.settings = get_settings()
        self.storage_options = self._get_storage_options()
        
    def _get_storage_options(self) -> Dict[str, Any]:
        """
        Get storage options for R2/S3 connectivity.
        
        Returns:
            Dict[str, Any]: Storage options for dask_cudf.read_parquet
        """
        r2_config = self.config.r2
        
        storage_options = {
            'key': r2_config['access_key'],
            'secret': r2_config['secret_key'],
            'endpoint_url': r2_config['endpoint_url'],
            'region': r2_config['region'],
            'use_ssl': True,
            'verify': True,
            'client_kwargs': {
                'region_name': r2_config['region']
            }
        }
        
        logger.debug(f"R2 storage options configured for endpoint: {r2_config['endpoint_url']}")
        return storage_options
    
    def _validate_r2_credentials(self) -> bool:
        """
        Validate that R2 credentials are properly configured.
        
        Returns:
            bool: True if credentials are valid, False otherwise
        """
        required_fields = ['account_id', 'access_key', 'secret_key', 'bucket_name']
        
        for field in required_fields:
            if not self.config.r2.get(field):
                logger.error(f"Missing R2 configuration field: {field}")
                return False
        
        logger.info("R2 credentials validation passed")
        return True
    
    def _build_r2_path(self, currency_pair: str, base_path: str) -> str:
        """
        Build the complete R2 path for a currency pair.
        
        Args:
            currency_pair: The currency pair (e.g., 'EURUSD')
            base_path: The base path in R2
            
        Returns:
            str: Complete R2 path for the currency pair
        """
        # Ensure base_path ends with '/'
        if not base_path.endswith('/'):
            base_path += '/'
        
        # Build the complete path
        r2_path = f"s3://{self.config.r2['bucket_name']}/{base_path}{currency_pair}/"
        
        logger.debug(f"Built R2 path: {r2_path}")
        return r2_path
    
    def _get_optimal_read_parameters(self, currency_pair: str, base_path: str) -> Dict[str, Any]:
        """
        Determine optimal parameters for dask_cudf.read_parquet based on data characteristics.
        
        Args:
            currency_pair: The currency pair to load
            base_path: The base path in R2
            
        Returns:
            Dict[str, Any]: Optimal read parameters
        """
        # Default parameters
        read_params = {
            'blocksize': None,  # Let Dask determine optimal block size
            'aggregate_files': True,  # Group small files
            'filesystem': 'arrow',  # Use PyArrow's optimized S3 interface
            'storage_options': self.storage_options,
            'engine': 'pyarrow',  # Use PyArrow engine for better performance
        }
        
        # Add filters if we know the data structure
        # This can be enhanced based on actual data characteristics
        read_params['filters'] = None
        
        logger.debug(f"Read parameters for {currency_pair}: {read_params}")
        return read_params
    
    def load_currency_pair_data(
        self, 
        currency_pair: str, 
        base_path: str,
        client: Optional[Client] = None
    ) -> Optional[dask_cudf.DataFrame]:
        """
        Load currency pair data from R2 to GPU memory.
        
        Args:
            currency_pair: The currency pair to load (e.g., 'EURUSD')
            base_path: The base path in R2 for the data
            client: Optional Dask client for distributed loading
            
        Returns:
            Optional[dask_cudf.DataFrame]: Loaded data as dask_cudf DataFrame, None if failed
        """
        try:
            logger.info(f"Loading data for currency pair: {currency_pair}")
            
            # Validate credentials
            if not self._validate_r2_credentials():
                logger.error("R2 credentials validation failed")
                return None
            
            # Build R2 path
            r2_path = self._build_r2_path(currency_pair, base_path)
            logger.info(f"Loading from R2 path: {r2_path}")
            
            # Get optimal read parameters
            read_params = self._get_optimal_read_parameters(currency_pair, base_path)
            
            # Load data using dask_cudf.read_parquet
            logger.info(f"Starting data loading for {currency_pair}...")
            
            df = dask_cudf.read_parquet(
                r2_path,
                **read_params
            )
            
            # Trigger computation to verify data is accessible
            # This will also distribute data across GPU workers
            if client:
                logger.info("Distributing data across GPU cluster...")
                df = df.persist()
                # Wait for data to be loaded and distributed
                client.wait(df)
            
            # Get basic information about the loaded data
            num_partitions = df.npartitions
            logger.info(f"Successfully loaded {currency_pair} data with {num_partitions} partitions")
            
            # Log data schema
            sample_df = df.head()
            logger.info(f"Data schema: {sample_df.columns.tolist()}")
            logger.info(f"Data types: {sample_df.dtypes.to_dict()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data for {currency_pair}: {e}")
            return None
    
    def load_currency_pair_data_sync(
        self, 
        currency_pair: str, 
        base_path: str
    ) -> Optional[cudf.DataFrame]:
        """
        Load currency pair data synchronously (for smaller datasets or testing).
        
        Args:
            currency_pair: The currency pair to load
            base_path: The base path in R2
            
        Returns:
            Optional[cudf.DataFrame]: Loaded data as cudf DataFrame, None if failed
        """
        try:
            logger.info(f"Loading data synchronously for currency pair: {currency_pair}")
            
            # Load as dask_cudf first
            ddf = self.load_currency_pair_data(currency_pair, base_path)
            
            if ddf is None:
                return None
            
            # Compute to get cudf DataFrame
            logger.info("Computing dask_cudf to cudf DataFrame...")
            df = ddf.compute()
            
            logger.info(f"Successfully loaded {currency_pair} data: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data synchronously for {currency_pair}: {e}")
            return None
    
    def get_data_info(self, currency_pair: str, base_path: str) -> Optional[Dict[str, Any]]:
        """
        Get information about the data without loading it completely.
        
        Args:
            currency_pair: The currency pair
            base_path: The base path in R2
            
        Returns:
            Optional[Dict[str, Any]]: Data information, None if failed
        """
        try:
            logger.info(f"Getting data info for currency pair: {currency_pair}")
            
            # Load data
            df = self.load_currency_pair_data(currency_pair, base_path)
            
            if df is None:
                return None
            
            # Get information
            info = {
                'currency_pair': currency_pair,
                'npartitions': df.npartitions,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'memory_usage': df.memory_usage_per_partition().compute().sum(),
            }
            
            # Get sample data for schema
            sample = df.head()
            info['sample_data'] = sample.to_dict('records')
            
            logger.info(f"Data info for {currency_pair}: {info}")
            return info
            
        except Exception as e:
            logger.error(f"Failed to get data info for {currency_pair}: {e}")
            return None
    
    def validate_data_path(self, currency_pair: str, base_path: str) -> bool:
        """
        Validate that the data path exists in R2.
        
        Args:
            currency_pair: The currency pair
            base_path: The base path in R2
            
        Returns:
            bool: True if path exists, False otherwise
        """
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            # Build R2 path
            r2_path = self._build_r2_path(currency_pair, base_path)
            
            # Create S3 client for R2
            s3_client = boto3.client(
                's3',
                endpoint_url=self.config.r2['endpoint_url'],
                aws_access_key_id=self.config.r2['access_key'],
                aws_secret_access_key=self.config.r2['secret_key'],
                region_name=self.config.r2['region']
            )
            
            # Check if path exists
            bucket = self.config.r2['bucket_name']
            prefix = r2_path.replace(f"s3://{bucket}/", "")
            
            response = s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                MaxKeys=1
            )
            
            exists = 'Contents' in response
            logger.info(f"Data path validation for {currency_pair}: {'EXISTS' if exists else 'NOT FOUND'}")
            
            return exists
            
        except Exception as e:
            logger.error(f"Failed to validate data path for {currency_pair}: {e}")
            return False


# Convenience functions for direct use
def load_currency_pair_data(
    currency_pair: str, 
    base_path: str,
    client: Optional[Client] = None
) -> Optional[dask_cudf.DataFrame]:
    """
    Convenience function to load currency pair data from R2.
    
    Args:
        currency_pair: The currency pair to load
        base_path: The base path in R2
        client: Optional Dask client
        
    Returns:
        Optional[dask_cudf.DataFrame]: Loaded data
    """
    loader = R2DataLoader()
    return loader.load_currency_pair_data(currency_pair, base_path, client)


def load_currency_pair_data_sync(
    currency_pair: str, 
    base_path: str
) -> Optional[cudf.DataFrame]:
    """
    Convenience function to load currency pair data synchronously.
    
    Args:
        currency_pair: The currency pair to load
        base_path: The base path in R2
        
    Returns:
        Optional[cudf.DataFrame]: Loaded data
    """
    loader = R2DataLoader()
    return loader.load_currency_pair_data_sync(currency_pair, base_path)


def validate_data_path(currency_pair: str, base_path: str) -> bool:
    """
    Convenience function to validate data path.
    
    Args:
        currency_pair: The currency pair
        base_path: The base path in R2
        
    Returns:
        bool: True if path exists
    """
    loader = R2DataLoader()
    return loader.validate_data_path(currency_pair, base_path)
