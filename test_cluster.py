#!/usr/bin/env python3

import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client
    import cupy as cp
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

try:
    # Test GPU detection
    gpu_count = cp.cuda.runtime.getDeviceCount()
    print(f"✓ Detected {gpu_count} GPU(s)")
except Exception as e:
    print(f"✗ GPU detection error: {e}")
    sys.exit(1)

try:
    # Test cluster creation
    print("Creating cluster...")
    cluster = LocalCUDACluster(
        n_workers=gpu_count,
        threads_per_worker=1,
        memory_limit="auto",
        rmm_pool_size="24GB",
        protocol='tcp',
        dashboard_address=':8787',
        silence_logs=logging.WARNING
    )
    print("✓ Cluster created successfully")
except Exception as e:
    print(f"✗ Cluster creation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    # Test client creation
    print("Creating client...")
    client = Client(cluster)
    print("✓ Client created successfully")
except Exception as e:
    print(f"✗ Client creation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    # Test worker connection
    print("Waiting for workers...")
    client.wait_for_workers(gpu_count, timeout=60)
    print(f"✓ {gpu_count} workers ready")
    print(f"✓ Dashboard URL: {client.dashboard_link}")
    print(f"✓ Active workers: {len(client.scheduler_info()['workers'])}")
except Exception as e:
    print(f"✗ Worker connection error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    # Test simple computation
    print("Testing computation...")
    import dask_cudf
    import cudf
    
    # Create simple test data
    df = cudf.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
    ddf = dask_cudf.from_cudf(df, npartitions=2)
    
    result = ddf['a'].sum().compute()
    print(f"✓ Test computation successful: {result}")
    
except Exception as e:
    print(f"✗ Computation test error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    # Cleanup
    print("Cleaning up...")
    client.close()
    cluster.close()
    print("✓ Cleanup successful")
except Exception as e:
    print(f"✗ Cleanup error: {e}")
    sys.exit(1)

print("🎉 All tests passed! Cluster is working correctly.")
