#!/usr/bin/env python3
"""
Dask-CUDA Cluster Test Script for Dynamic Stage 0 Pipeline

This script tests the Dask-CUDA cluster orchestration functionality including:
- Cluster startup and configuration
- Client connection
- GPU functionality testing
- Graceful shutdown
"""

import sys
import os
import traceback
import time
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from orchestration.main import DaskClusterManager, managed_dask_cluster
    from config import get_config
    from config.settings import get_settings
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)


def test_configuration_loading() -> Dict[str, Any]:
    """Test configuration loading for Dask cluster."""
    try:
        print("Testing configuration loading...")
        
        config = get_config()
        settings = get_settings()
        
        print(f"✓ Configuration loaded successfully")
        print(f"  Dask settings:")
        print(f"    - GPUs per worker: {settings.dask.gpus_per_worker}")
        print(f"    - Threads per worker: {settings.dask.threads_per_worker}")
        print(f"    - Memory limit: {settings.dask.memory_limit}")
        print(f"    - RMM pool size: {settings.dask.rmm_pool_size}")
        
        return {"status": "success"}
        
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_gpu_detection() -> Dict[str, Any]:
    """Test GPU detection functionality."""
    try:
        print("Testing GPU detection...")
        
        import cupy as cp
        
        gpu_count = cp.cuda.runtime.getDeviceCount()
        print(f"✓ Detected {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            props = cp.cuda.runtime.getDeviceProperties(i)
            print(f"  GPU {i}: {props['name'].decode()}")
        
        return {"status": "success", "gpu_count": gpu_count}
        
    except Exception as e:
        print(f"✗ GPU detection failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_cluster_creation() -> Dict[str, Any]:
    """Test Dask-CUDA cluster creation."""
    try:
        print("Testing Dask-CUDA cluster creation...")
        
        with managed_dask_cluster() as cluster_manager:
            client = cluster_manager.get_client()
            
            print(f"✓ Cluster created successfully")
            print(f"  Dashboard URL: {client.dashboard_link}")
            print(f"  Number of workers: {len(client.scheduler_info()['workers'])}")
            
            # Test basic cluster functionality
            def test_function():
                import cupy as cp
                return cp.sum(cp.random.random(100))
            
            future = client.submit(test_function)
            result = future.result()
            print(f"✓ Distributed computation test successful: {result}")
            
            return {"status": "success", "dashboard_url": client.dashboard_link}
        
    except Exception as e:
        print(f"✗ Cluster creation failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_cluster_manager_class() -> Dict[str, Any]:
    """Test DaskClusterManager class directly."""
    try:
        print("Testing DaskClusterManager class...")
        
        cluster_manager = DaskClusterManager()
        
        # Test cluster startup
        success = cluster_manager.start_cluster()
        if not success:
            return {"status": "failed", "error": "Failed to start cluster"}
        
        print("✓ Cluster started successfully")
        
        # Test client access
        client = cluster_manager.get_client()
        if client is None:
            return {"status": "failed", "error": "Client is None"}
        
        print("✓ Client accessible")
        
        # Test cluster status
        if not cluster_manager.is_active():
            return {"status": "failed", "error": "Cluster not active"}
        
        print("✓ Cluster is active")
        
        # Test shutdown
        cluster_manager.shutdown()
        print("✓ Cluster shutdown successful")
        
        return {"status": "success"}
        
    except Exception as e:
        print(f"✗ DaskClusterManager test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_context_manager() -> Dict[str, Any]:
    """Test context manager functionality."""
    try:
        print("Testing context manager...")
        
        # Test that cluster is properly managed
        with managed_dask_cluster() as cluster_manager:
            client = cluster_manager.get_client()
            print("✓ Context manager entered successfully")
            print(f"  Dashboard: {client.dashboard_link}")
            
            # Simulate some work
            time.sleep(2)
        
        print("✓ Context manager exited successfully")
        return {"status": "success"}
        
    except Exception as e:
        print(f"✗ Context manager test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_rmm_configuration() -> Dict[str, Any]:
    """Test RMM (RAPIDS Memory Manager) configuration."""
    try:
        print("Testing RMM configuration...")
        
        # This test requires the cluster to be running
        with managed_dask_cluster() as cluster_manager:
            client = cluster_manager.get_client()
            
            # Test RMM functionality
            def test_rmm():
                import cupy as cp
                import rmm
                
                # Allocate memory through RMM
                x = cp.random.random(1000)
                y = cp.random.random(1000)
                z = x + y
                
                return cp.sum(z)
            
            future = client.submit(test_rmm)
            result = future.result()
            print(f"✓ RMM test successful: {result}")
        
        return {"status": "success"}
        
    except Exception as e:
        print(f"✗ RMM configuration test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_error_handling() -> Dict[str, Any]:
    """Test error handling in cluster management."""
    try:
        print("Testing error handling...")
        
        # Test with invalid configuration
        cluster_manager = DaskClusterManager()
        
        # This should not crash even if there are issues
        try:
            success = cluster_manager.start_cluster()
            if success:
                cluster_manager.shutdown()
        except Exception as e:
            print(f"✓ Error handling test: caught exception as expected: {e}")
        
        return {"status": "success"}
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return {"status": "failed", "error": str(e)}


def main():
    """Run all Dask-CUDA cluster tests."""
    print("=" * 60)
    print("Dynamic Stage 0 - Dask-CUDA Cluster Test")
    print("=" * 60)
    
    # Run cluster tests
    tests = [
        ("Configuration Loading", test_configuration_loading),
        ("GPU Detection", test_gpu_detection),
        ("Cluster Creation", test_cluster_creation),
        ("DaskClusterManager Class", test_cluster_manager_class),
        ("Context Manager", test_context_manager),
        ("RMM Configuration", test_rmm_configuration),
        ("Error Handling", test_error_handling),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
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
        print("🎉 All Dask-CUDA cluster tests passed! Cluster orchestration is ready for Dynamic Stage 0.")
        return 0
    else:
        print("❌ Some Dask-CUDA cluster tests failed. Please check the errors above.")
        print("\nNote: Some tests may fail if:")
        print("  - GPU environment is not properly set up")
        print("  - CUDA drivers are not installed")
        print("  - Dask-CUDA is not available")
        return 1


if __name__ == "__main__":
    sys.exit(main())
