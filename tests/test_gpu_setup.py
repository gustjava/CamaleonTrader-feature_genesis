#!/usr/bin/env python3
"""
GPU Setup Test Script for Dynamic Stage 0 Pipeline

This script verifies that all GPU-accelerated libraries are properly installed
and functioning in the Docker environment.
"""

import sys
import traceback
from typing import Dict, Any

def test_cupy() -> Dict[str, Any]:
    """Test CuPy installation and basic functionality."""
    try:
        import cupy as cp
        print(f"✓ CuPy version: {cp.__version__}")
        
        # Test GPU availability
        gpu_count = cp.cuda.runtime.getDeviceCount()
        print(f"✓ Available GPUs: {gpu_count}")
        
        # Test basic operations
        x = cp.array([1, 2, 3, 4, 5])
        y = cp.array([2, 3, 4, 5, 6])
        z = x + y
        print(f"✓ Basic CuPy operation: {z}")
        
        return {"status": "success", "gpu_count": gpu_count}
    except Exception as e:
        print(f"✗ CuPy test failed: {e}")
        return {"status": "failed", "error": str(e)}

def test_cudf() -> Dict[str, Any]:
    """Test cuDF installation and basic functionality."""
    try:
        import cudf
        print(f"✓ cuDF version: {cudf.__version__}")
        
        # Test DataFrame creation
        df = cudf.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [1.1, 2.2, 3.3, 4.4, 5.5],
            'c': ['x', 'y', 'z', 'w', 'v']
        })
        print(f"✓ cuDF DataFrame created with shape: {df.shape}")
        
        # Test basic operations
        result = df['a'] + df['b']
        print(f"✓ cuDF operation result: {result}")
        
        return {"status": "success"}
    except Exception as e:
        print(f"✗ cuDF test failed: {e}")
        return {"status": "failed", "error": str(e)}

def test_dask_cuda() -> Dict[str, Any]:
    """Test Dask-CUDA installation and cluster creation."""
    try:
        from dask_cuda import LocalCUDACluster
        import dask_cudf
        
        print(f"✓ Dask-CUDA available")
        
        # Create a small cluster for testing
        cluster = LocalCUDACluster(n_workers=1, threads_per_worker=1)
        print(f"✓ Dask-CUDA cluster created: {cluster}")
        
        # Test dask-cudf
        import dask.dataframe as dd
        df = dask_cudf.from_cudf(
            cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}),
            npartitions=1
        )
        result = df.compute()
        print(f"✓ Dask-cuDF operation successful: {result}")
        
        cluster.close()
        return {"status": "success"}
    except Exception as e:
        print(f"✗ Dask-CUDA test failed: {e}")
        return {"status": "failed", "error": str(e)}

def test_cusignal() -> Dict[str, Any]:
    """Test cuSignal installation and basic functionality."""
    try:
        import cusignal
        print(f"✓ cuSignal version: {cusignal.__version__}")
        
        # Test basic signal processing
        import cupy as cp
        x = cp.random.random(1000)
        y = cusignal.convolve(x, cp.ones(10), mode='same')
        print(f"✓ cuSignal convolution successful, output shape: {y.shape}")
        
        return {"status": "success"}
    except Exception as e:
        print(f"✗ cuSignal test failed: {e}")
        return {"status": "failed", "error": str(e)}

def test_cuml() -> Dict[str, Any]:
    """Test cuML installation and basic functionality."""
    try:
        import cuml
        print(f"✓ cuML version: {cuml.__version__}")
        
        # Test basic ML operation
        from cuml.linear_model import LinearRegression
        import cupy as cp
        
        X = cp.random.random((100, 3))
        y = cp.random.random(100)
        
        model = LinearRegression()
        model.fit(X, y)
        print(f"✓ cuML LinearRegression fit successful")
        
        return {"status": "success"}
    except Exception as e:
        print(f"✗ cuML test failed: {e}")
        return {"status": "failed", "error": str(e)}

def test_python_dependencies() -> Dict[str, Any]:
    """Test Python dependencies for the pipeline."""
    dependencies = [
        'boto3', 'sqlalchemy', 'pymysql', 'pyyaml', 
        'python-dotenv', 'structlog', 'rich'
    ]
    
    results = {}
    for dep in dependencies:
        try:
            module = __import__(dep)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {dep} version: {version}")
            results[dep] = {"status": "success", "version": version}
        except ImportError as e:
            print(f"✗ {dep} not available: {e}")
            results[dep] = {"status": "failed", "error": str(e)}
    
    return results

def main():
    """Run all GPU and dependency tests."""
    print("=" * 60)
    print("Dynamic Stage 0 - GPU Environment Test")
    print("=" * 60)
    
    tests = [
        ("CuPy", test_cupy),
        ("cuDF", test_cudf),
        ("Dask-CUDA", test_dask_cuda),
        ("cuSignal", test_cusignal),
        ("cuML", test_cuml),
        ("Python Dependencies", test_python_dependencies),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n--- Testing {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            traceback.print_exc()
            results[test_name] = {"status": "crashed", "error": str(e)}
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = result.get("status", "unknown")
        if status == "success":
            print(f"✓ {test_name}: PASSED")
            passed += 1
        else:
            print(f"✗ {test_name}: FAILED")
            if "error" in result:
                print(f"  Error: {result['error']}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! GPU environment is ready for Dynamic Stage 0.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
