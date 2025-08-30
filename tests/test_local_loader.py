#!/usr/bin/env python3
"""
Test script for LocalDataLoader module.

This script tests the LocalDataLoader functionality without requiring
actual GPU libraries or data files.
"""

import sys
import os
import traceback
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock GPU libraries before importing
import sys
from unittest.mock import Mock, patch, MagicMock

# Mock dask_cudf and cudf
sys.modules['dask_cudf'] = Mock()
sys.modules['cudf'] = Mock()
sys.modules['dask.distributed'] = Mock()

try:
    from data_io.local_loader import LocalDataLoader, validate_data_path
    from config import get_config
    from config.settings import get_settings
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)


def test_local_loader_initialization() -> Dict[str, Any]:
    """Test LocalDataLoader initialization."""
    try:
        print("Testing LocalDataLoader initialization...")
        
        # Mock the settings to avoid validation issues
        with patch('data_io.local_loader.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.dask = Mock()
            mock_settings.dask.rmm_pool_size = "24GB"
            mock_get_settings.return_value = mock_settings
            
            loader = LocalDataLoader()
            
            # Check that the loader was initialized correctly
            assert loader.local_data_root == "/workspace/data"
            assert loader.settings == mock_settings
            
            print("✓ LocalDataLoader initialized correctly")
            return {"status": "success"}
            
    except Exception as e:
        print(f"✗ LocalDataLoader initialization test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_local_path_construction() -> Dict[str, Any]:
    """Test local path construction."""
    try:
        print("Testing local path construction...")
        
        with patch('data_io.local_loader.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_get_settings.return_value = mock_settings
            
            loader = LocalDataLoader()
            
            # Test cases
            test_cases = [
                ("EURUSD", "/workspace/data/EURUSD"),
                ("data/forex/GBPUSD", "/workspace/data/data/forex/GBPUSD"),
                ("", "/workspace/data"),
            ]
            
            for r2_path, expected_path in test_cases:
                local_path = loader._get_local_path(r2_path)
                assert str(local_path) == expected_path
                print(f"  ✓ {r2_path} -> {local_path}")
            
            print("✓ Local path construction working correctly")
            return {"status": "success"}
            
    except Exception as e:
        print(f"✗ Local path construction test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_validate_data_path() -> Dict[str, Any]:
    """Test data path validation."""
    try:
        print("Testing data path validation...")
        
        with patch('data_io.local_loader.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_get_settings.return_value = mock_settings
            
            loader = LocalDataLoader()
            
            # Test with mocked path
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.is_dir', return_value=True), \
                 patch('pathlib.Path.glob', return_value=['file1.parquet', 'file2.parquet']):
                
                result = loader.validate_data_path("EURUSD")
                assert result is True
                print("  ✓ Valid path with parquet files")
            
            # Test with non-existent path
            with patch('pathlib.Path.exists', return_value=False):
                result = loader.validate_data_path("NONEXISTENT")
                assert result is False
                print("  ✓ Non-existent path correctly rejected")
            
            # Test with directory without parquet files
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.is_dir', return_value=True), \
                 patch('pathlib.Path.glob', return_value=[]):
                
                result = loader.validate_data_path("EMPTY_DIR")
                assert result is False
                print("  ✓ Directory without parquet files correctly rejected")
            
            print("✓ Data path validation working correctly")
            return {"status": "success"}
            
    except Exception as e:
        print(f"✗ Data path validation test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_get_data_info() -> Dict[str, Any]:
    """Test data info retrieval."""
    try:
        print("Testing data info retrieval...")
        
        with patch('data_io.local_loader.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_get_settings.return_value = mock_settings
            
            loader = LocalDataLoader()
            
            # Mock file structure and cudf
            mock_parquet_files = [Mock(), Mock()]
            mock_parquet_files[0].stat.return_value.st_size = 1024 * 1024  # 1MB
            mock_parquet_files[1].stat.return_value.st_size = 2048 * 1024  # 2MB
            
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.is_dir', return_value=True), \
                 patch('pathlib.Path.glob', return_value=mock_parquet_files), \
                 patch('cudf.read_parquet') as mock_read_parquet:
                
                # Mock the sample DataFrame
                mock_df = Mock()
                mock_columns = Mock()
                mock_columns.tolist.return_value = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                mock_df.columns = mock_columns
                
                mock_dtypes = Mock()
                mock_dtypes.to_dict.return_value = {'timestamp': 'datetime64[ns]', 'open': 'float64'}
                mock_df.dtypes = mock_dtypes
                
                mock_read_parquet.return_value = mock_df
                
                info = loader.get_data_info("EURUSD")
                
                assert info is not None
                assert info['path'] == "/workspace/data/EURUSD"
                assert info['num_files'] == 2
                assert info['columns'] == ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                assert info['file_size_mb'] == 3.0  # 1MB + 2MB
                
                print("  ✓ Data info retrieved correctly")
            
            print("✓ Data info retrieval working correctly")
            return {"status": "success"}
            
    except Exception as e:
        print(f"✗ Data info retrieval test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_list_available_pairs() -> Dict[str, Any]:
    """Test listing available currency pairs."""
    try:
        print("Testing listing available pairs...")
        
        with patch('data_io.local_loader.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_get_settings.return_value = mock_settings
            
            loader = LocalDataLoader()
            
            # Mock directory structure
            mock_items = [
                Mock(is_dir=lambda: True, relative_to=lambda x: "EURUSD"),
                Mock(is_dir=lambda: True, relative_to=lambda x: "GBPUSD"),
                Mock(is_dir=lambda: False),  # Not a directory
            ]
            
            # Mock glob to return parquet files for directories
            def mock_glob(pattern):
                if pattern == "*.parquet":
                    return ["file.parquet"]  # Has parquet files
                return []
            
            for item in mock_items:
                item.glob = mock_glob
            
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.rglob', return_value=mock_items):
                
                pairs = loader.list_available_pairs()
                
                assert len(pairs) == 2
                assert "EURUSD" in pairs
                assert "GBPUSD" in pairs
                
                print(f"  ✓ Found {len(pairs)} available pairs: {pairs}")
            
            print("✓ Listing available pairs working correctly")
            return {"status": "success"}
            
    except Exception as e:
        print(f"✗ Listing available pairs test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_convenience_functions() -> Dict[str, Any]:
    """Test convenience functions."""
    try:
        print("Testing convenience functions...")
        
        # Test validate_data_path convenience function
        with patch('data_io.local_loader.LocalDataLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader.validate_data_path.return_value = True
            mock_loader_class.return_value = mock_loader
            
            result = validate_data_path("EURUSD")
            assert result is True
            mock_loader.validate_data_path.assert_called_once_with("EURUSD")
            
            print("  ✓ Convenience functions working correctly")
        
        print("✓ Convenience functions working correctly")
        return {"status": "success"}
        
    except Exception as e:
        print(f"✗ Convenience functions test failed: {e}")
        return {"status": "failed", "error": str(e)}


def main():
    """Run all LocalDataLoader tests."""
    print("=" * 60)
    print("Dynamic Stage 0 - LocalDataLoader Test")
    print("=" * 60)
    
    # Run tests
    tests = [
        ("LocalDataLoader Initialization", test_local_loader_initialization),
        ("Local Path Construction", test_local_path_construction),
        ("Data Path Validation", test_validate_data_path),
        ("Data Info Retrieval", test_get_data_info),
        ("Listing Available Pairs", test_list_available_pairs),
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
        else:
            print(f"✗ {test_name}: FAILED")
            if "error" in result:
                print(f"  Error: {result['error']}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All LocalDataLoader tests passed!")
        return 0
    else:
        print("❌ Some LocalDataLoader tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
