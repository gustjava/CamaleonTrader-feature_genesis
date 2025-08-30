#!/usr/bin/env python3
"""
R2 Path Validation Test for Dynamic Stage 0 Pipeline

This script tests R2 path validation without downloading data,
using the actual R2 configuration from the upload script.
"""

import sys
import os
import traceback
from typing import Dict, Any, List

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from data_io.r2_loader import R2DataLoader, validate_data_path
    from config import get_config
    from config.settings import get_settings
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)


def test_r2_configuration() -> Dict[str, Any]:
    """Test R2 configuration with actual credentials."""
    try:
        print("Testing R2 configuration...")
        
        config = get_config()
        settings = get_settings()
        
        print(f"✓ R2 configuration loaded:")
        print(f"  - Account ID: {settings.r2.account_id}")
        print(f"  - Bucket: {settings.r2.bucket_name}")
        print(f"  - Endpoint: {settings.r2.endpoint_url}")
        print(f"  - Region: {settings.r2.region}")
        
        # Verify credentials are not empty
        if not settings.r2.access_key or not settings.r2.secret_key:
            print("✗ R2 credentials are empty")
            return {"status": "failed", "error": "Empty R2 credentials"}
        
        print("✓ R2 credentials are configured")
        return {"status": "success"}
        
    except Exception as e:
        print(f"✗ R2 configuration test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_r2_connection() -> Dict[str, Any]:
    """Test R2 connection without downloading data."""
    try:
        print("Testing R2 connection...")
        
        r2_loader = R2DataLoader()
        
        # Test credentials validation
        if not r2_loader._validate_r2_credentials():
            print("✗ R2 credentials validation failed")
            return {"status": "failed", "error": "Invalid R2 credentials"}
        
        print("✓ R2 credentials validation passed")
        
        # Test storage options
        storage_options = r2_loader._get_storage_options()
        required_keys = ['key', 'secret', 'endpoint_url', 'region']
        
        for key in required_keys:
            if key not in storage_options:
                print(f"✗ Missing storage option: {key}")
                return {"status": "failed", "error": f"Missing storage option: {key}"}
        
        print("✓ R2 storage options configured correctly")
        return {"status": "success"}
        
    except Exception as e:
        print(f"✗ R2 connection test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_currency_pair_paths() -> Dict[str, Any]:
    """Test currency pair paths in R2."""
    try:
        print("Testing currency pair paths in R2...")
        
        # Common currency pairs to test
        currency_pairs = [
            "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
            "EURGBP", "EURJPY", "GBPJPY", "CHFJPY", "NZDUSD"
        ]
        
        # Base path from the upload script
        base_path = "data"
        
        r2_loader = R2DataLoader()
        
        found_pairs = []
        missing_pairs = []
        
        for pair in currency_pairs:
            exists = r2_loader.validate_data_path(pair, base_path)
            if exists:
                found_pairs.append(pair)
                print(f"  ✓ {pair}: EXISTS")
            else:
                missing_pairs.append(pair)
                print(f"  ✗ {pair}: NOT FOUND")
        
        print(f"\nPath validation summary:")
        print(f"  - Found: {len(found_pairs)} pairs")
        print(f"  - Missing: {len(missing_pairs)} pairs")
        
        if found_pairs:
            print(f"  - Available pairs: {', '.join(found_pairs)}")
        
        return {
            "status": "success", 
            "found_pairs": found_pairs,
            "missing_pairs": missing_pairs,
            "total_tested": len(currency_pairs)
        }
        
    except Exception as e:
        print(f"✗ Currency pair paths test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_r2_path_building() -> Dict[str, Any]:
    """Test R2 path building functionality."""
    try:
        print("Testing R2 path building...")
        
        r2_loader = R2DataLoader()
        
        # Test cases
        test_cases = [
            ("EURUSD", "data"),
            ("GBPUSD", "data/forex"),
            ("USDJPY", "data/forex/"),
        ]
        
        for currency_pair, base_path in test_cases:
            r2_path = r2_loader._build_r2_path(currency_pair, base_path)
            expected_bucket = r2_loader.config.r2['bucket_name']
            
            if expected_bucket in r2_path and currency_pair in r2_path:
                print(f"  ✓ {currency_pair} ({base_path}): {r2_path}")
            else:
                print(f"  ✗ {currency_pair} ({base_path}): {r2_path}")
                return {"status": "failed", "error": f"Invalid path for {currency_pair}"}
        
        print("✓ R2 path building working correctly")
        return {"status": "success"}
        
    except Exception as e:
        print(f"✗ R2 path building test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_boto3_connection() -> Dict[str, Any]:
    """Test boto3 connection to R2."""
    try:
        print("Testing boto3 connection to R2...")
        
        import boto3
        from botocore.exceptions import ClientError
        
        config = get_config()
        
        # Create S3 client for R2
        s3_client = boto3.client(
            's3',
            endpoint_url=config.r2['endpoint_url'],
            aws_access_key_id=config.r2['access_key'],
            aws_secret_access_key=config.r2['secret_key'],
            region_name=config.r2['region']
        )
        
        # Test bucket access
        try:
            response = s3_client.head_bucket(Bucket=config.r2['bucket_name'])
            print(f"✓ Successfully connected to R2 bucket: {config.r2['bucket_name']}")
            
            # List a few objects to verify access
            response = s3_client.list_objects_v2(
                Bucket=config.r2['bucket_name'],
                MaxKeys=5
            )
            
            if 'Contents' in response:
                print(f"  - Found {len(response['Contents'])} objects in bucket")
                for obj in response['Contents'][:3]:
                    print(f"    - {obj['Key']}")
            else:
                print("  - Bucket appears to be empty")
            
            return {"status": "success"}
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                print(f"✗ Bucket {config.r2['bucket_name']} not found")
                return {"status": "failed", "error": f"Bucket not found: {config.r2['bucket_name']}"}
            else:
                print(f"✗ Error accessing bucket: {error_code}")
                return {"status": "failed", "error": f"Bucket access error: {error_code}"}
        
    except ImportError:
        print("⚠ boto3 not available, skipping boto3 connection test")
        return {"status": "skipped", "reason": "boto3 not available"}
    except Exception as e:
        print(f"✗ boto3 connection test failed: {e}")
        return {"status": "failed", "error": str(e)}


def main():
    """Run all R2 path validation tests."""
    print("=" * 60)
    print("Dynamic Stage 0 - R2 Path Validation Test")
    print("=" * 60)
    
    # Run tests
    tests = [
        ("R2 Configuration", test_r2_configuration),
        ("R2 Connection", test_r2_connection),
        ("Currency Pair Paths", test_currency_pair_paths),
        ("R2 Path Building", test_r2_path_building),
        ("Boto3 Connection", test_boto3_connection),
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
        print("🎉 All R2 path validation tests passed!")
        return 0
    else:
        print("❌ Some R2 path validation tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
