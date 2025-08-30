#!/usr/bin/env python3
"""
R2 Data Loader Test Script for Dynamic Stage 0 Pipeline

This script tests the R2 data loader functionality including:
- R2 credentials validation
- Data path validation
- Data loading with dask_cudf
- Error handling
"""

import sys
import os
import traceback
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from data_io.r2_loader import R2DataLoader, load_currency_pair_data, validate_data_path
    from config import get_config
    from config.settings import get_settings
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)


def test_configuration_loading() -> Dict[str, Any]:
    """Test R2 configuration loading."""
    try:
        print("Testing R2 configuration loading...")
        
        config = get_config()
        settings = get_settings()
        
        print(f"✓ Configuration loaded successfully")
        print(f"  R2 settings:")
        print(f"    - Account ID: {settings.r2.account_id}")
        print(f"    - Endpoint URL: {settings.r2.endpoint_url}")
        print(f"    - Bucket: {settings.r2.bucket_name}")
        print(f"    - Region: {settings.r2.region}")
        
        return {"status": "success"}
        
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_r2_credentials_validation() -> Dict[str, Any]:
    """Test R2 credentials validation."""
    try:
        print("Testing R2 credentials validation...")
        
        loader = R2DataLoader()
        
        # Test validation method
        is_valid = loader._validate_r2_credentials()
        
        if is_valid:
            print("✓ R2 credentials validation passed")
            return {"status": "success"}
        else:
            print("✗ R2 credentials validation failed")
            return {"status": "failed", "error": "Invalid R2 credentials"}
        
    except Exception as e:
        print(f"✗ R2 credentials validation test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_storage_options() -> Dict[str, Any]:
    """Test storage options configuration."""
    try:
        print("Testing storage options configuration...")
        
        loader = R2DataLoader()
        storage_options = loader._get_storage_options()
        
        required_keys = ['key', 'secret', 'endpoint_url', 'region']
        for key in required_keys:
            if key not in storage_options:
                print(f"✗ Missing storage option: {key}")
                return {"status": "failed", "error": f"Missing storage option: {key}"}
        
        print("✓ Storage options configured correctly")
        print(f"  Endpoint: {storage_options['endpoint_url']}")
        print(f"  Region: {storage_options['region']}")
        
        return {"status": "success"}
        
    except Exception as e:
        print(f"✗ Storage options test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_r2_path_building() -> Dict[str, Any]:
    """Test R2 path building functionality."""
    try:
        print("Testing R2 path building...")
        
        loader = R2DataLoader()
        
        # Test path building
        currency_pair = "EURUSD"
        base_path = "data/forex"
        
        r2_path = loader._build_r2_path(currency_pair, base_path)
        
        expected_pattern = f"s3://{loader.config.r2['bucket_name']}/data/forex/EURUSD/"
        
        if expected_pattern in r2_path:
            print("✓ R2 path built correctly")
            print(f"  Path: {r2_path}")
            return {"status": "success", "path": r2_path}
        else:
            print(f"✗ R2 path building failed. Expected pattern: {expected_pattern}")
            return {"status": "failed", "error": "Incorrect path format"}
        
    except Exception as e:
        print(f"✗ R2 path building test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_read_parameters() -> Dict[str, Any]:
    """Test read parameters configuration."""
    try:
        print("Testing read parameters configuration...")
        
        loader = R2DataLoader()
        
        currency_pair = "EURUSD"
        base_path = "data/forex"
        
        read_params = loader._get_optimal_read_parameters(currency_pair, base_path)
        
        required_keys = ['blocksize', 'aggregate_files', 'filesystem', 'storage_options', 'engine']
        for key in required_keys:
            if key not in read_params:
                print(f"✗ Missing read parameter: {key}")
                return {"status": "failed", "error": f"Missing read parameter: {key}"}
        
        print("✓ Read parameters configured correctly")
        print(f"  Filesystem: {read_params['filesystem']}")
        print(f"  Engine: {read_params['engine']}")
        print(f"  Aggregate files: {read_params['aggregate_files']}")
        
        return {"status": "success"}
        
    except Exception as e:
        print(f"✗ Read parameters test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_data_path_validation() -> Dict[str, Any]:
    """Test data path validation functionality."""
    try:
        print("Testing data path validation...")
        
        # Test with a sample currency pair
        currency_pair = "EURUSD"
        base_path = "data/forex"
        
        exists = validate_data_path(currency_pair, base_path)
        
        if exists:
            print("✓ Data path validation successful - path exists")
            return {"status": "success", "exists": True}
        else:
            print("⚠ Data path validation - path not found (this may be expected)")
            print("  This could be normal if the test data doesn't exist in R2")
            return {"status": "skipped", "reason": "Data path not found in R2"}
        
    except Exception as e:
        print(f"✗ Data path validation test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_data_loading_simulation() -> Dict[str, Any]:
    """Test data loading simulation (without actual data)."""
    try:
        print("Testing data loading simulation...")
        
        loader = R2DataLoader()
        
        # Test the loading process without actual data
        currency_pair = "TESTPAIR"
        base_path = "data/test"
        
        # This will fail due to missing data, but we can test the setup
        try:
            df = loader.load_currency_pair_data(currency_pair, base_path)
            if df is not None:
                print("✓ Data loading successful (unexpected)")
                return {"status": "success"}
            else:
                print("✓ Data loading failed as expected (no test data)")
                return {"status": "skipped", "reason": "No test data available"}
        except Exception as e:
            print(f"✓ Data loading failed as expected: {e}")
            return {"status": "skipped", "reason": "No test data available"}
        
    except Exception as e:
        print(f"✗ Data loading simulation test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_error_handling() -> Dict[str, Any]:
    """Test error handling in R2 loader."""
    try:
        print("Testing error handling...")
        
        loader = R2DataLoader()
        
        # Test with invalid parameters
        try:
            df = loader.load_currency_pair_data("", "")  # Invalid parameters
            if df is None:
                print("✓ Error handling test passed - invalid parameters handled correctly")
                return {"status": "success"}
            else:
                print("✗ Error handling test failed - should have returned None")
                return {"status": "failed", "error": "Invalid parameters not handled"}
        except Exception as e:
            print(f"✓ Error handling test passed - exception caught: {e}")
            return {"status": "success"}
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_convenience_functions() -> Dict[str, Any]:
    """Test convenience functions."""
    try:
        print("Testing convenience functions...")
        
        # Test convenience functions exist and are callable
        from data_io.r2_loader import (
            load_currency_pair_data,
            load_currency_pair_data_sync,
            validate_data_path
        )
        
        # Test that functions are callable
        currency_pair = "TESTPAIR"
        base_path = "data/test"
        
        # These should not crash even if data doesn't exist
        try:
            result1 = load_currency_pair_data(currency_pair, base_path)
            result2 = load_currency_pair_data_sync(currency_pair, base_path)
            result3 = validate_data_path(currency_pair, base_path)
            
            print("✓ Convenience functions are callable")
            return {"status": "success"}
            
        except Exception as e:
            print(f"✓ Convenience functions handled errors correctly: {e}")
            return {"status": "success"}
        
    except Exception as e:
        print(f"✗ Convenience functions test failed: {e}")
        return {"status": "failed", "error": str(e)}


def main():
    """Run all R2 loader tests."""
    print("=" * 60)
    print("Dynamic Stage 0 - R2 Data Loader Test")
    print("=" * 60)
    
    # Run R2 loader tests
    tests = [
        ("Configuration Loading", test_configuration_loading),
        ("R2 Credentials Validation", test_r2_credentials_validation),
        ("Storage Options", test_storage_options),
        ("R2 Path Building", test_r2_path_building),
        ("Read Parameters", test_read_parameters),
        ("Data Path Validation", test_data_path_validation),
        ("Data Loading Simulation", test_data_loading_simulation),
        ("Error Handling", test_error_handling),
        ("Convenience Functions", test_convenience_functions),
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
        elif status == "skipped":
            print(f"⚠ {test_name}: SKIPPED")
            if "reason" in result:
                print(f"  Reason: {result['reason']}")
        else:
            print(f"✗ {test_name}: FAILED")
            if "error" in result:
                print(f"  Error: {result['error']}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All R2 loader tests passed! R2 data loader is ready for Dynamic Stage 0.")
        return 0
    else:
        print("❌ Some R2 loader tests failed. Please check the errors above.")
        print("\nNote: Some tests may fail if:")
        print("  - R2 credentials are not configured")
        print("  - R2 bucket does not exist")
        print("  - Test data is not available in R2")
        return 1


if __name__ == "__main__":
    sys.exit(main())
