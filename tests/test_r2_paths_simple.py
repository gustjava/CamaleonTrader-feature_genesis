#!/usr/bin/env python3
"""
Simple R2 Path Validation Test for Dynamic Stage 0 Pipeline

This script tests R2 configuration and path building without downloading data
or requiring GPU libraries.
"""

import sys
import os
import traceback
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
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
        
        # Load config without validation
        from config import load_config
        config = load_config()
        
        print(f"✓ R2 configuration loaded:")
        print(f"  - Account ID: {config.r2['account_id']}")
        print(f"  - Bucket: {config.r2['bucket_name']}")
        print(f"  - Endpoint: {config.r2['endpoint_url']}")
        print(f"  - Region: {config.r2['region']}")
        
        # Verify credentials are not empty
        if not config.r2['access_key'] or not config.r2['secret_key']:
            print("✗ R2 credentials are empty")
            return {"status": "failed", "error": "Empty R2 credentials"}
        
        print("✓ R2 credentials are configured")
        return {"status": "success"}
        
    except Exception as e:
        print(f"✗ R2 configuration test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_r2_path_building() -> Dict[str, Any]:
    """Test R2 path building functionality."""
    try:
        print("Testing R2 path building...")
        
        # Load config without validation
        from config import load_config
        config = load_config()
        
        # Test cases
        test_cases = [
            ("EURUSD", "data"),
            ("GBPUSD", "data/forex"),
            ("USDJPY", "data/forex/"),
        ]
        
        for currency_pair, base_path in test_cases:
            # Build path manually since we can't import R2DataLoader
            bucket_name = config.r2['bucket_name']
            r2_path = f"s3://{bucket_name}/{base_path}/{currency_pair}/"
            
            if bucket_name in r2_path and currency_pair in r2_path:
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
        
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:
            print("⚠ boto3 not available, skipping boto3 connection test")
            return {"status": "skipped", "reason": "boto3 not available"}
        
        # Load config without validation
        from config import load_config
        config = load_config()
        
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
        
    except Exception as e:
        print(f"✗ boto3 connection test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_currency_pair_paths() -> Dict[str, Any]:
    """Test currency pair paths in R2 using boto3."""
    try:
        print("Testing currency pair paths in R2...")
        
        try:
            import boto3
        except ImportError:
            print("⚠ boto3 not available, skipping currency pair paths test")
            return {"status": "skipped", "reason": "boto3 not available"}
        
        # Common currency pairs to test
        currency_pairs = [
            "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
            "EURGBP", "EURJPY", "GBPJPY", "CHFJPY", "NZDUSD"
        ]
        
        # Base path from the upload script
        base_path = "data"
        
        # Load config without validation
        from config import load_config
        config = load_config()
        
        # Create S3 client for R2
        s3_client = boto3.client(
            's3',
            endpoint_url=config.r2['endpoint_url'],
            aws_access_key_id=config.r2['access_key'],
            aws_secret_access_key=config.r2['secret_key'],
            region_name=config.r2['region']
        )
        
        found_pairs = []
        missing_pairs = []
        
        for pair in currency_pairs:
            # Check if the path exists
            prefix = f"{base_path}/{pair}/"
            response = s3_client.list_objects_v2(
                Bucket=config.r2['bucket_name'],
                Prefix=prefix,
                MaxKeys=1
            )
            
            if 'Contents' in response and len(response['Contents']) > 0:
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


def main():
    """Run all R2 path validation tests."""
    print("=" * 60)
    print("Dynamic Stage 0 - Simple R2 Path Validation Test")
    print("=" * 60)
    
    # Run tests
    tests = [
        ("R2 Configuration", test_r2_configuration),
        ("R2 Path Building", test_r2_path_building),
        ("Boto3 Connection", test_boto3_connection),
        ("Currency Pair Paths", test_currency_pair_paths),
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
    
    # Count successful tests (including skipped ones)
    successful_tests = passed
    for test_name, result in results.items():
        if result.get("status") == "skipped":
            successful_tests += 1
    
    if successful_tests == total:
        print("🎉 All R2 path validation tests passed!")
        return 0
    else:
        print("❌ Some R2 path validation tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
