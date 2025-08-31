#!/usr/bin/env python3
"""
Simple test script to validate Dask cluster initialization.
This script tests if the CUDA/CuPy environment is properly configured.
"""

import sys
import os
import logging
from dask.distributed import Client, LocalCluster
import dask_cudf
import cudf
import cupy as cp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_cuda_environment():
    """Test CUDA environment and CuPy installation."""
    logger.info("Testing CUDA environment...")
    
    try:
        # Test CuPy
        logger.info(f"CuPy version: {cp.__version__}")
        logger.info(f"CuPy CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
        
        # Test basic CuPy operations
        x = cp.array([1, 2, 3, 4, 5])
        y = cp.array([2, 3, 4, 5, 6])
        z = x + y
        logger.info(f"CuPy test calculation: {z}")
        
        # Test CUDA memory
        mem_info = cp.cuda.runtime.memGetInfo()
        logger.info(f"CUDA memory - Free: {mem_info[0] / 1024**3:.2f} GB, Total: {mem_info[1] / 1024**3:.2f} GB")
        
        return True
        
    except Exception as e:
        logger.error(f"CUDA environment test failed: {e}")
        return False

def test_cudf_operations():
    """Test cuDF operations."""
    logger.info("Testing cuDF operations...")
    
    try:
        # Create test data
        df = cudf.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [2, 3, 4, 5, 6],
            'c': [3, 4, 5, 6, 7]
        })
        
        # Test basic operations
        result = df['a'] + df['b']
        logger.info(f"cuDF test calculation: {result}")
        
        # Test rolling operations
        rolling_mean = df['a'].rolling(window=3).mean()
        logger.info(f"cuDF rolling mean: {rolling_mean}")
        
        return True
        
    except Exception as e:
        logger.error(f"cuDF operations test failed: {e}")
        return False

def test_dask_cluster():
    """Test Dask cluster initialization."""
    logger.info("Testing Dask cluster initialization...")
    
    try:
        # Create local cluster
        cluster = LocalCluster(
            n_workers=1,
            threads_per_worker=1,
            memory_limit="2GB"
        )
        
        # Create client
        client = Client(cluster)
        
        logger.info(f"Cluster dashboard: {client.dashboard_link}")
        logger.info(f"Cluster info: {client}")
        
        # Test basic dask_cudf operations
        df = cudf.DataFrame({
            'x': range(100),
            'y': range(100, 200)
        })
        
        ddf = dask_cudf.from_cudf(df, npartitions=2)
        result = ddf['x'].sum().compute()
        logger.info(f"dask_cudf test result: {result}")
        
        # Clean up
        client.close()
        cluster.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Dask cluster test failed: {e}")
        return False

def test_imports():
    """Test if all required modules can be imported."""
    logger.info("Testing module imports...")
    
    try:
        # Test core imports
        import numpy as np
        import pandas as pd
        import scipy
        import sklearn
        
        # Test RAPIDS imports
        import cudf
        import dask_cudf
        import cupy as cp
        
        # Test signal processing
        try:
            import cusignal
            logger.info("cusignal imported successfully")
        except ImportError:
            logger.warning("cusignal not available")
        
        # Test ML imports
        try:
            import cuml
            logger.info("cuml imported successfully")
        except ImportError:
            logger.warning("cuml not available")
        
        logger.info("All core imports successful")
        return True
        
    except Exception as e:
        logger.error(f"Import test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting cluster validation tests...")
    
    tests = [
        ("Module Imports", test_imports),
        ("CUDA Environment", test_cuda_environment),
        ("cuDF Operations", test_cudf_operations),
        ("Dask Cluster", test_dask_cluster),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results[test_name] = success
            
            if success:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name}: FAILED with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    all_passed = True
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 All tests passed! Environment is ready for the pipeline.")
        return 0
    else:
        logger.error("\n💥 Some tests failed. Please check the environment configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
