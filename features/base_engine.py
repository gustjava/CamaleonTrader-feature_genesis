"""
Base class for all feature engineering engines in the pipeline.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Mapping, Optional
from config.settings import Settings
from dask.distributed import Client
import dask_cudf
import cudf

logger = logging.getLogger(__name__)


class BaseFeatureEngine(ABC):
    """
    A base class for feature engineering modules to ensure a consistent interface.
    """
    def __init__(self, settings: Settings, client: Client):
        """
        Initialize the engine with shared settings and Dask client.

        Args:
            settings: The global settings object.
            client: The Dask distributed client.
        """
        self.settings = settings
        self.client = client
        # Use logger with the name of the specific engine class
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def process(self, df: dask_cudf.DataFrame) -> dask_cudf.DataFrame:
        """
        The main processing method to be implemented by each engine.

        Args:
            df: The input dask_cudf DataFrame.

        Returns:
            The DataFrame with new features added.
        """
        raise NotImplementedError("Each feature engine must implement the 'process' method.")

    # ---- helpers comuns ----
    def _log_info(self, msg: str, **fields):
        """Log info message with optional fields."""
        if fields:
            self.logger.info("%s | %s", msg, fields)
        else:
            self.logger.info(msg)

    def _log_warn(self, msg: str, **fields):
        """Log warning message with optional fields."""
        if fields:
            self.logger.warning("%s | %s", msg, fields)
        else:
            self.logger.warning(msg)

    def _log_error(self, msg: str, **fields):
        """Log error message with optional fields."""
        if fields:
            self.logger.error("%s | %s", msg, fields)
        else:
            self.logger.error(msg)

    def _ensure_sorted(self, df: dask_cudf.DataFrame, by: str) -> dask_cudf.DataFrame:
        """
        Ensure global sorting by column (deterministic shuffle in Dask).
        
        Args:
            df: Input DataFrame
            by: Column to sort by
            
        Returns:
            Sorted DataFrame
        """
        return df.set_index(by, shuffle="tasks")

    def _broadcast_scalars(self, df: dask_cudf.DataFrame, mapping: Mapping[str, Any]) -> dask_cudf.DataFrame:
        """
        Add scalar columns to DataFrame in a lazy and cheap way.
        
        Args:
            df: Input DataFrame
            mapping: Dictionary of column_name -> scalar_value
            
        Returns:
            DataFrame with new scalar columns
        """
        return df.assign(**mapping)

    def _single_partition(self, df: dask_cudf.DataFrame, cols: Optional[list] = None) -> dask_cudf.DataFrame:
        """
        Force DataFrame to single partition (entire series in one GPU worker).
        
        Args:
            df: Input DataFrame
            cols: Columns to select (optional)
            
        Returns:
            Single partition DataFrame
        """
        sub = df[cols] if cols else df
        return sub.repartition(npartitions=1)

    def _log_progress(self, message: str, **kwargs):
        """
        Helper method for consistent logging (legacy compatibility).
        """
        self._log_info(message, **kwargs)
