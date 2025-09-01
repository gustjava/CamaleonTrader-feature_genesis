#!/usr/bin/env python3
"""
Debug script for Dynamic Stage 0 Pipeline

This script analyzes the pipeline issues and provides detailed logging to identify
why stationarization and statistical tests are not generating features.
"""

import logging
import sys
import os
from typing import Dict, Any, List

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cudf
import cupy as cp
from config.unified_config import get_unified_config as get_config
from config.unified_config import get_unified_config as get_settings
from data_io.local_loader import LocalDataLoader
from features.stationarization import StationarizationEngine
from features.statistical_tests import StatisticalTests
from features.signal_processing import SignalProcessor
from features.garch_models import GARCHModels
from dask.distributed import Client

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_data_columns(df: cudf.DataFrame, currency_pair: str) -> Dict[str, Any]:
    """
    Analyze the columns in the dataset to understand what's available.
    """
    logger.info(f"🔍 ANALYZING DATA COLUMNS FOR {currency_pair}")
    logger.info("=" * 80)
    
    # Get all column names
    all_columns = list(df.columns)
    logger.info(f"Total columns: {len(all_columns)}")
    
    # Categorize columns
    price_cols = [col for col in all_columns if any(term in col.lower() for term in ['open', 'high', 'low', 'close'])]
    volume_cols = [col for col in all_columns if any(term in col.lower() for term in ['volume', 'tick'])]
    spread_cols = [col for col in all_columns if any(term in col.lower() for term in ['spread'])]
    return_cols = [col for col in all_columns if any(term in col.lower() for term in ['ret', 'return'])]
    ofi_cols = [col for col in all_columns if any(term in col.lower() for term in ['ofi', 'aggression'])]
    volatility_cols = [col for col in all_columns if any(term in col.lower() for term in ['vol', 'volatility', 'realized'])]
    
    # Log categorized columns
    logger.info(f"📊 Price columns ({len(price_cols)}): {price_cols}")
    logger.info(f"📊 Volume columns ({len(volume_cols)}): {volume_cols}")
    logger.info(f"📊 Spread columns ({len(spread_cols)}): {spread_cols}")
    logger.info(f"📊 Return columns ({len(return_cols)}): {return_cols}")
    logger.info(f"📊 OFI columns ({len(ofi_cols)}): {ofi_cols}")
    logger.info(f"📊 Volatility columns ({len(volatility_cols)}): {volatility_cols}")
    
    # Check for specific required columns
    required_for_dcor = ['returns', 'tick_volume']
    required_for_rolling_corr = ['returns', 'tick_volume', 'ofi', 'spread_rel', 'realized_vol', 'close', 'volume']
    
    missing_dcor = [col for col in required_for_dcor if col not in all_columns]
    missing_rolling = [col for col in required_for_rolling_corr if col not in all_columns]
    
    logger.info(f"❌ Missing for dCor: {missing_dcor}")
    logger.info(f"❌ Missing for rolling correlations: {missing_rolling}")
    
    # Sample data analysis
    logger.info(f"📈 Data shape: {df.shape}")
    logger.info(f"📈 Data types: {df.dtypes.to_dict()}")
    
    # Check for NaN values
    nan_counts = df.isna().sum()
    high_nan_cols = nan_counts[nan_counts > len(df) * 0.1]  # More than 10% NaN
    if not high_nan_cols.empty:
        logger.warning(f"⚠️ Columns with >10% NaN values: {high_nan_cols.to_dict()}")
    
    return {
        'total_columns': len(all_columns),
        'price_columns': price_cols,
        'volume_columns': volume_cols,
        'spread_columns': spread_cols,
        'return_columns': return_cols,
        'ofi_columns': ofi_cols,
        'volatility_columns': volatility_cols,
        'missing_dcor': missing_dcor,
        'missing_rolling': missing_rolling,
        'data_shape': df.shape,
        'high_nan_columns': high_nan_cols.to_dict() if not high_nan_cols.empty else {}
    }

def debug_stationarization(df: cudf.DataFrame, currency_pair: str, settings, client: Client) -> cudf.DataFrame:
    """
    Debug the stationarization process with detailed logging.
    """
    logger.info(f"🔧 DEBUGGING STATIONARIZATION FOR {currency_pair}")
    logger.info("=" * 80)
    
    # Create stationarization engine
    station_engine = StationarizationEngine(settings, client)
    
    # Log configuration
    logger.info(f"📋 Stationarization config:")
    logger.info(f"   - Rolling windows: {settings.features.rolling_windows}")
    logger.info(f"   - Rolling min periods: {settings.features.rolling_min_periods}")
    logger.info(f"   - Frac diff values: {settings.features.frac_diff_values}")
    
    # Check what columns are available for rolling correlations
    feature_pairs = [
        ('returns', 'tick_volume'),
        ('returns', 'ofi'),
        ('spread_rel', 'realized_vol'),
        ('close', 'volume'),
        ('returns', 'spread_rel'),
        ('tick_volume', 'spread_rel')
    ]
    
    logger.info(f"🔍 Checking feature pairs for rolling correlations:")
    for col1, col2 in feature_pairs:
        col1_exists = col1 in df.columns
        col2_exists = col2 in df.columns
        logger.info(f"   - {col1} ({'✅' if col1_exists else '❌'}) x {col2} ({'✅' if col2_exists else '❌'})")
        
        if col1_exists and col2_exists:
            # Check for non-null values
            col1_non_null = df[col1].notna().sum()
            col2_non_null = df[col2].notna().sum()
            logger.info(f"     Non-null values: {col1}({col1_non_null}), {col2}({col2_non_null})")
    
    # Process stationarization
    logger.info(f"🚀 Starting stationarization processing...")
    cols_before = len(df.columns)
    
    try:
        result_df = station_engine.process_cudf(df)
        cols_after = len(result_df.columns)
        new_cols = cols_after - cols_before
        
        logger.info(f"✅ Stationarization complete: {cols_before} -> {cols_after} cols (+{new_cols} new)")
        
        if new_cols > 0:
            new_col_names = [col for col in result_df.columns if col not in df.columns]
            logger.info(f"📊 New columns: {new_col_names}")
        else:
            logger.warning("⚠️ No new columns generated by stationarization!")
            
        return result_df
        
    except Exception as e:
        logger.error(f"❌ Error in stationarization: {e}", exc_info=True)
        return df

def debug_statistical_tests(df: cudf.DataFrame, currency_pair: str) -> cudf.DataFrame:
    """
    Debug the statistical tests process with detailed logging.
    """
    logger.info(f"🔧 DEBUGGING STATISTICAL TESTS FOR {currency_pair}")
    logger.info("=" * 80)
    
    # Create statistical tests engine
    stats_engine = StatisticalTests()
    
    # Check for frac_diff columns
    frac_diff_cols = [c for c in df.columns if "frac_diff" in c]
    logger.info(f"📊 Frac diff columns found: {frac_diff_cols}")
    
    # Check for returns and tick_volume
    has_returns = 'returns' in df.columns
    has_tick_volume = 'tick_volume' in df.columns
    
    logger.info(f"📊 Required columns for dCor:")
    logger.info(f"   - returns: {'✅' if has_returns else '❌'}")
    logger.info(f"   - tick_volume: {'✅' if has_tick_volume else '❌'}")
    
    if has_returns and has_tick_volume:
        returns_non_null = df['returns'].notna().sum()
        tick_volume_non_null = df['tick_volume'].notna().sum()
        logger.info(f"   - returns non-null: {returns_non_null}")
        logger.info(f"   - tick_volume non-null: {tick_volume_non_null}")
    
    # Process statistical tests
    logger.info(f"🚀 Starting statistical tests processing...")
    cols_before = len(df.columns)
    
    try:
        result_df = stats_engine.process_cudf(df)
        cols_after = len(result_df.columns)
        new_cols = cols_after - cols_before
        
        logger.info(f"✅ Statistical tests complete: {cols_before} -> {cols_after} cols (+{new_cols} new)")
        
        if new_cols > 0:
            new_col_names = [col for col in result_df.columns if col not in df.columns]
            logger.info(f"📊 New columns: {new_col_names}")
        else:
            logger.warning("⚠️ No new columns generated by statistical tests!")
            
        return result_df
        
    except Exception as e:
        logger.error(f"❌ Error in statistical tests: {e}", exc_info=True)
        return df

def debug_signal_processing(df: cudf.DataFrame, currency_pair: str) -> cudf.DataFrame:
    """
    Debug the signal processing with detailed logging.
    """
    logger.info(f"🔧 DEBUGGING SIGNAL PROCESSING FOR {currency_pair}")
    logger.info("=" * 80)
    
    # Create signal processor
    sig_processor = SignalProcessor()
    
    # Check for close column
    has_close = 'y_close' in df.columns
    logger.info(f"📊 Close column for Baxter-King filter: {'✅' if has_close else '❌'}")
    
    if has_close:
        close_non_null = df['y_close'].notna().sum()
        logger.info(f"   - y_close non-null: {close_non_null}")
    
    # Process signal processing
    logger.info(f"🚀 Starting signal processing...")
    cols_before = len(df.columns)
    
    try:
        result_df = sig_processor.process_cudf(df)
        cols_after = len(result_df.columns)
        new_cols = cols_after - cols_before
        
        logger.info(f"✅ Signal processing complete: {cols_before} -> {cols_after} cols (+{new_cols} new)")
        
        if new_cols > 0:
            new_col_names = [col for col in result_df.columns if col not in df.columns]
            logger.info(f"📊 New columns: {new_col_names}")
        else:
            logger.warning("⚠️ No new columns generated by signal processing!")
            
        return result_df
        
    except Exception as e:
        logger.error(f"❌ Error in signal processing: {e}", exc_info=True)
        return df

def debug_garch_models(df: cudf.DataFrame, currency_pair: str) -> cudf.DataFrame:
    """
    Debug the GARCH models with detailed logging.
    """
    logger.info(f"🔧 DEBUGGING GARCH MODELS FOR {currency_pair}")
    logger.info("=" * 80)
    
    # Create GARCH models engine
    garch_engine = GARCHModels()
    
    # Check for returns column
    has_returns = 'y_ret_1m' in df.columns
    logger.info(f"📊 Returns column for GARCH: {'✅' if has_returns else '❌'}")
    
    if has_returns:
        returns_non_null = df['y_ret_1m'].notna().sum()
        logger.info(f"   - y_ret_1m non-null: {returns_non_null}")
    
    # Process GARCH models
    logger.info(f"🚀 Starting GARCH models processing...")
    cols_before = len(df.columns)
    
    try:
        result_df = garch_engine.process_cudf(df)
        cols_after = len(result_df.columns)
        new_cols = cols_after - cols_before
        
        logger.info(f"✅ GARCH models complete: {cols_before} -> {cols_after} cols (+{new_cols} new)")
        
        if new_cols > 0:
            new_col_names = [col for col in result_df.columns if col not in df.columns]
            logger.info(f"📊 New columns: {new_col_names}")
        else:
            logger.warning("⚠️ No new columns generated by GARCH models!")
            
        return result_df
        
    except Exception as e:
        logger.error(f"❌ Error in GARCH models: {e}", exc_info=True)
        return df

def main():
    """
    Main debug function to analyze the pipeline issues.
    """
    logger.info("🔍 DYNAMIC STAGE 0 PIPELINE DEBUG")
    logger.info("=" * 80)
    
    try:
        # Get settings
        settings = get_settings()
        logger.info("✅ Settings loaded successfully")
        
        # Create a dummy client for testing
        client = None
        
        # Load data for a single currency pair
        local_loader = LocalDataLoader()
        available_pairs = local_loader.discover_currency_pairs()
        
        if not available_pairs:
            logger.error("❌ No currency pairs found in data directory")
            return 1
        
        # Use the first available pair for debugging
        pair_info = available_pairs[0]
        currency_pair = pair_info['currency_pair']
        file_path = pair_info['data_path']  # Changed from 'file_path' to 'data_path'
        
        logger.info(f"🔍 Debugging currency pair: {currency_pair}")
        logger.info(f"📁 File path: {file_path}")
        
        # Load data
        logger.info(f"📥 Loading data for {currency_pair}...")
        df = local_loader.load_data_synchronously(file_path)
        
        if df is None or df.empty:
            logger.error(f"❌ Failed to load data for {currency_pair}")
            return 1
        
        # Convert to cuDF
        gdf = cudf.DataFrame.from_pandas(df)
        logger.info(f"✅ Data loaded: {gdf.shape}")
        
        # Analyze data columns
        analysis = analyze_data_columns(gdf, currency_pair)
        
        # Debug each stage
        logger.info("\n" + "="*80)
        logger.info("STAGE-BY-STAGE DEBUG")
        logger.info("="*80)
        
        # Stage 1: Stationarization
        gdf = debug_stationarization(gdf, currency_pair, settings, client)
        
        # Stage 2: Signal Processing
        gdf = debug_signal_processing(gdf, currency_pair)
        
        # Stage 3: Statistical Tests
        gdf = debug_statistical_tests(gdf, currency_pair)
        
        # Stage 4: GARCH Models
        gdf = debug_garch_models(gdf, currency_pair)
        
        # Final summary
        logger.info("\n" + "="*80)
        logger.info("FINAL SUMMARY")
        logger.info("="*80)
        logger.info(f"📊 Final data shape: {gdf.shape}")
        logger.info(f"📊 Total columns: {len(gdf.columns)}")
        
        # Show all column names
        logger.info(f"📋 All columns: {list(gdf.columns)}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error in debug script: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
