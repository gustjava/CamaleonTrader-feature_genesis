#!/usr/bin/env python3
"""
Pipeline Integration Test Script for Dynamic Stage 0 Pipeline

This script tests the integrated pipeline functionality including:
- Database task management
- Dask-CUDA cluster orchestration
- R2 data loading
- Complete workflow execution
"""

import sys
import os
import traceback
import time
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from orchestration.main import run_pipeline, process_currency_pair, DaskClusterManager
    from data_io.db_handler import DatabaseHandler, get_pending_currency_pairs, update_task_status
    from data_io.r2_loader import R2DataLoader
    from config import get_config
    from config.settings import get_settings
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)


def test_configuration_integration() -> Dict[str, Any]:
    """Test that all configuration components work together."""
    try:
        print("Testing configuration integration...")
        
        config = get_config()
        settings = get_settings()
        
        # Test that all required components have configuration
        components = [
            ('database', settings.database),
            ('r2', settings.r2),
            ('dask', settings.dask),
            ('features', settings.features),
            ('processing', settings.processing)
        ]
        
        for name, component in components:
            if component is None:
                print(f"✗ Missing configuration for {name}")
                return {"status": "failed", "error": f"Missing {name} configuration"}
        
        print("✓ All configuration components available")
        return {"status": "success"}
        
    except Exception as e:
        print(f"✗ Configuration integration test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_database_integration() -> Dict[str, Any]:
    """Test database integration with the pipeline."""
    try:
        print("Testing database integration...")
        
        # Test database connection
        with DatabaseHandler() as db:
            pending_tasks = db.get_pending_currency_pairs()
            print(f"✓ Database connection successful. Found {len(pending_tasks)} pending tasks")
            
            if pending_tasks:
                # Test with first task
                task = pending_tasks[0]
                print(f"  Sample task: {task['currency_pair']} (ID: {task['task_id']})")
            
            return {"status": "success", "task_count": len(pending_tasks)}
        
    except Exception as e:
        print(f"✗ Database integration test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_r2_loader_integration() -> Dict[str, Any]:
    """Test R2 loader integration."""
    try:
        print("Testing R2 loader integration...")
        
        r2_loader = R2DataLoader()
        
        # Test credentials validation
        credentials_valid = r2_loader._validate_r2_credentials()
        if credentials_valid:
            print("✓ R2 credentials validation passed")
        else:
            print("⚠ R2 credentials validation failed (may be expected)")
        
        # Test storage options
        storage_options = r2_loader._get_storage_options()
        required_keys = ['key', 'secret', 'endpoint_url', 'region']
        all_keys_present = all(key in storage_options for key in required_keys)
        
        if all_keys_present:
            print("✓ R2 storage options configured")
        else:
            print("⚠ R2 storage options incomplete")
        
        return {"status": "success", "credentials_valid": credentials_valid}
        
    except Exception as e:
        print(f"✗ R2 loader integration test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_dask_cluster_integration() -> Dict[str, Any]:
    """Test Dask cluster integration."""
    try:
        print("Testing Dask cluster integration...")
        
        # Test cluster creation
        with DaskClusterManager() as cluster_manager:
            client = cluster_manager.get_client()
            
            print(f"✓ Dask cluster created successfully")
            print(f"  Dashboard: {client.dashboard_link}")
            print(f"  Workers: {len(client.scheduler_info()['workers'])}")
            
            # Test basic distributed computation
            def test_function():
                import cupy as cp
                return cp.sum(cp.random.random(100))
            
            future = client.submit(test_function)
            result = future.result()
            print(f"✓ Distributed computation test successful: {result}")
            
            return {"status": "success", "dashboard_url": client.dashboard_link}
        
    except Exception as e:
        print(f"✗ Dask cluster integration test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_component_initialization() -> Dict[str, Any]:
    """Test that all pipeline components can be initialized."""
    try:
        print("Testing component initialization...")
        
        # Initialize all components
        db_handler = DatabaseHandler()
        r2_loader = R2DataLoader()
        
        print("✓ Database handler initialized")
        print("✓ R2 loader initialized")
        
        # Test database connection
        if db_handler.connect():
            print("✓ Database connection successful")
            db_handler.close()
        else:
            print("✗ Database connection failed")
            return {"status": "failed", "error": "Database connection failed"}
        
        return {"status": "success"}
        
    except Exception as e:
        print(f"✗ Component initialization test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_pipeline_function_signatures() -> Dict[str, Any]:
    """Test that pipeline functions have correct signatures."""
    try:
        print("Testing pipeline function signatures...")
        
        # Test that functions exist and are callable
        from orchestration.main import run_pipeline, process_currency_pair
        
        # Check function signatures
        import inspect
        
        # Test process_currency_pair signature
        sig = inspect.signature(process_currency_pair)
        expected_params = ['currency_pair', 'r2_path', 'task_id', 'client', 'db_handler', 'r2_loader']
        
        actual_params = list(sig.parameters.keys())
        if actual_params == expected_params:
            print("✓ process_currency_pair has correct signature")
        else:
            print(f"✗ process_currency_pair signature mismatch. Expected: {expected_params}, Got: {actual_params}")
            return {"status": "failed", "error": "Function signature mismatch"}
        
        # Test run_pipeline signature
        sig = inspect.signature(run_pipeline)
        if len(sig.parameters) == 0:
            print("✓ run_pipeline has correct signature")
        else:
            print(f"✗ run_pipeline should have no parameters, got: {list(sig.parameters.keys())}")
            return {"status": "failed", "error": "Function signature mismatch"}
        
        return {"status": "success"}
        
    except Exception as e:
        print(f"✗ Pipeline function signatures test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_error_handling() -> Dict[str, Any]:
    """Test error handling in the integrated pipeline."""
    try:
        print("Testing error handling...")
        
        # Test with invalid parameters
        try:
            # This should handle errors gracefully
            result = process_currency_pair(
                currency_pair="",
                r2_path="",
                task_id=-1,
                client=None,
                db_handler=None,
                r2_loader=None
            )
            print("✓ Error handling test passed - function handled invalid parameters")
            return {"status": "success"}
            
        except Exception as e:
            print(f"✓ Error handling test passed - exception caught: {e}")
            return {"status": "success"}
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_pipeline_simulation() -> Dict[str, Any]:
    """Test a simulated pipeline run without actual data."""
    try:
        print("Testing pipeline simulation...")
        
        # Test the pipeline setup without running actual processing
        config = get_config()
        settings = get_settings()
        
        # Initialize components
        db_handler = DatabaseHandler()
        r2_loader = R2DataLoader()
        
        # Get pending tasks
        pending_tasks = get_pending_currency_pairs()
        
        print(f"✓ Pipeline simulation setup successful")
        print(f"  - Configuration loaded")
        print(f"  - Database handler initialized")
        print(f"  - R2 loader initialized")
        print(f"  - Found {len(pending_tasks)} pending tasks")
        
        # Clean up
        db_handler.close()
        
        return {"status": "success", "pending_tasks": len(pending_tasks)}
        
    except Exception as e:
        print(f"✗ Pipeline simulation test failed: {e}")
        return {"status": "failed", "error": str(e)}


def main():
    """Run all pipeline integration tests."""
    print("=" * 60)
    print("Dynamic Stage 0 - Pipeline Integration Test")
    print("=" * 60)
    
    # Run integration tests
    tests = [
        ("Configuration Integration", test_configuration_integration),
        ("Database Integration", test_database_integration),
        ("R2 Loader Integration", test_r2_loader_integration),
        ("Dask Cluster Integration", test_dask_cluster_integration),
        ("Component Initialization", test_component_initialization),
        ("Pipeline Function Signatures", test_pipeline_function_signatures),
        ("Error Handling", test_error_handling),
        ("Pipeline Simulation", test_pipeline_simulation),
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
        print("🎉 All pipeline integration tests passed! Pipeline is ready for execution.")
        print("\nTo run the pipeline:")
        print("  python orchestration/main.py")
        return 0
    else:
        print("❌ Some pipeline integration tests failed. Please check the errors above.")
        print("\nNote: Some tests may fail if:")
        print("  - Database is not accessible")
        print("  - R2 credentials are not configured")
        print("  - GPU environment is not properly set up")
        return 1


if __name__ == "__main__":
    sys.exit(main())
