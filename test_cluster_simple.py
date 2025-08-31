#!/usr/bin/env python3

import os
import sys
import logging
import multiprocessing

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_cluster():
    """Test cluster creation step by step."""
    try:
        logger.info("Testing cluster creation...")
        
        # Test imports
        logger.info("Testing imports...")
        import cupy as cp
        from dask_cuda import LocalCUDACluster
        from dask.distributed import Client
        logger.info("✓ Imports successful")
        
        # Test GPU detection
        logger.info("Testing GPU detection...")
        gpu_count = cp.cuda.runtime.getDeviceCount()
        logger.info(f"✓ Detected {gpu_count} GPU(s)")
        
        # Test memory info
        logger.info("Testing memory info...")
        free_b, total_b = cp.cuda.runtime.memGetInfo()
        pool_bytes = int(total_b * 0.80)
        pool_str = f"{pool_bytes // (1024**3)}GB"
        logger.info(f"✓ Memory info: total={total_b/(1024**3):.1f}GB, pool={pool_str}")
        
        # Test RMM setup
        logger.info("Testing RMM setup...")
        os.environ.setdefault("RMM_ALLOCATOR", "cuda_async")
        logger.info("✓ RMM setup complete")
        
        # Test cluster creation
        logger.info("Testing cluster creation...")
        common_kwargs = dict(
            n_workers=gpu_count,
            threads_per_worker=1,
            rmm_async=True,
            rmm_pool_size=pool_str,
            jit_unspill=True,
            device_memory_limit="0.85",
            dashboard_address=":8787",
            silence_logs=logging.WARNING,
        )
        
        logger.info("Creating LocalCUDACluster...")
        cluster = LocalCUDACluster(protocol="tcp", **common_kwargs)
        logger.info("✓ Cluster created successfully")
        
        # Test client creation
        logger.info("Testing client creation...")
        client = Client(cluster)
        logger.info("✓ Client created successfully")
        
        # Test worker readiness
        logger.info("Testing worker readiness...")
        client.wait_for_workers(gpu_count, timeout=120)
        logger.info(f"✓ {gpu_count} workers ready")
        
        # Test cluster info
        logger.info("Testing cluster info...")
        logger.info(f"✓ Dashboard URL: {client.dashboard_link}")
        logger.info(f"✓ Active workers: {len(client.scheduler_info()['workers'])}")
        
        # Cleanup
        logger.info("Cleaning up...")
        client.close()
        cluster.close()
        logger.info("✓ Cleanup complete")
        
        logger.info("✓ ALL TESTS PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"✗ TEST FAILED: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    multiprocessing.freeze_support()
    success = test_cluster()
    sys.exit(0 if success else 1)
