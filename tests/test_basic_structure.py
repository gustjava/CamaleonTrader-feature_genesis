#!/usr/bin/env python3
"""
Basic Structure Test for Dynamic Stage 0 Pipeline

This script tests the basic project structure and imports
without requiring GPU libraries.
"""

import sys
import os
import traceback
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_project_structure() -> Dict[str, Any]:
    """Test that all required directories exist."""
    try:
        print("Testing project structure...")
        
        required_dirs = [
            "config",
            "orchestration", 
            "data_io",
            "features",
            "utils",
            "docker",
            "tests"
        ]
        
        missing_dirs = []
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                missing_dirs.append(dir_name)
            else:
                print(f"  ✓ {dir_name}/")
        
        if missing_dirs:
            print(f"  ✗ Missing directories: {', '.join(missing_dirs)}")
            return {"status": "failed", "error": f"Missing directories: {missing_dirs}"}
        
        print("✓ All required directories exist")
        return {"status": "success"}
        
    except Exception as e:
        print(f"✗ Project structure test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_config_imports() -> Dict[str, Any]:
    """Test configuration module imports."""
    try:
        print("Testing configuration imports...")
        
        from config import Config, load_config, get_config
        from config.settings import Settings, get_settings
        
        print("  ✓ Config class imported")
        print("  ✓ load_config function imported")
        print("  ✓ get_config function imported")
        print("  ✓ Settings class imported")
        print("  ✓ get_settings function imported")
        
        return {"status": "success"}
        
    except Exception as e:
        print(f"✗ Configuration imports test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_config_loading() -> Dict[str, Any]:
    """Test configuration loading from YAML (without validation)."""
    try:
        print("Testing configuration loading...")
        
        from config import load_config
        
        # Load config without validation
        config = load_config()
        
        # Check that config has required sections
        required_sections = ['database', 'r2', 'dask', 'features']
        for section in required_sections:
            if hasattr(config, section):
                print(f"  ✓ {section} section present")
            else:
                print(f"  ✗ {section} section missing")
                return {"status": "failed", "error": f"Missing config section: {section}"}
        
        print("✓ Configuration loaded successfully")
        return {"status": "success"}
        
    except Exception as e:
        print(f"✗ Configuration loading test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_module_files_exist() -> Dict[str, Any]:
    """Test that module files exist (without importing)."""
    try:
        print("Testing module files exist...")
        
        required_files = [
            "data_io/__init__.py",
            "data_io/db_handler.py",
            "data_io/r2_loader.py",
            "orchestration/__init__.py",
            "orchestration/main.py",
            "features/__init__.py",
            "utils/__init__.py",
            "config/__init__.py",
            "config/config.py",
            "config/settings.py",
            "config/config.yaml"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
            else:
                print(f"  ✓ {file_path}")
        
        if missing_files:
            print(f"  ✗ Missing files: {', '.join(missing_files)}")
            return {"status": "failed", "error": f"Missing files: {missing_files}"}
        
        print("✓ All required module files exist")
        return {"status": "success"}
        
    except Exception as e:
        print(f"✗ Module files test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_yaml_config() -> Dict[str, Any]:
    """Test that config.yaml exists and is valid."""
    try:
        print("Testing config.yaml...")
        
        config_path = "config/config.yaml"
        if not os.path.exists(config_path):
            print(f"  ✗ {config_path} not found")
            return {"status": "failed", "error": f"Config file not found: {config_path}"}
        
        print(f"  ✓ {config_path} exists")
        
        # Try to load it
        from config import load_config
        config = load_config(config_path)
        print("  ✓ config.yaml is valid YAML")
        
        return {"status": "success"}
        
    except Exception as e:
        print(f"✗ YAML config test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_sql_schema() -> Dict[str, Any]:
    """Test that SQL schema file exists."""
    try:
        print("Testing SQL schema...")
        
        sql_path = "docker/init.sql"
        if not os.path.exists(sql_path):
            print(f"  ✗ {sql_path} not found")
            return {"status": "failed", "error": f"SQL schema not found: {sql_path}"}
        
        print(f"  ✓ {sql_path} exists")
        
        # Check that it contains required tables
        with open(sql_path, 'r') as f:
            content = f.read()
        
        required_tables = ['processing_tasks', 'feature_status']
        for table in required_tables:
            if table in content:
                print(f"  ✓ {table} table defined")
            else:
                print(f"  ✗ {table} table not found")
                return {"status": "failed", "error": f"Missing table definition: {table}"}
        
        print("✓ SQL schema is valid")
        return {"status": "success"}
        
    except Exception as e:
        print(f"✗ SQL schema test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_docker_files() -> Dict[str, Any]:
    """Test that Docker files exist."""
    try:
        print("Testing Docker files...")
        
        docker_files = [
            "Dockerfile",
            "docker-compose.yml",
            "environment.yml",
            "requirements.txt"
        ]
        
        missing_files = []
        for file_name in docker_files:
            if not os.path.exists(file_name):
                missing_files.append(file_name)
            else:
                print(f"  ✓ {file_name}")
        
        if missing_files:
            print(f"  ⚠ Missing Docker files: {', '.join(missing_files)}")
            print("  (These are optional for basic structure test)")
        
        print("✓ Docker files check completed")
        return {"status": "success"}
        
    except Exception as e:
        print(f"✗ Docker files test failed: {e}")
        return {"status": "failed", "error": str(e)}


def main():
    """Run all basic structure tests."""
    print("=" * 60)
    print("Dynamic Stage 0 - Basic Structure Test")
    print("=" * 60)
    
    # Run tests
    tests = [
        ("Project Structure", test_project_structure),
        ("Configuration Imports", test_config_imports),
        ("Configuration Loading", test_config_loading),
        ("Module Files", test_module_files_exist),
        ("YAML Config", test_yaml_config),
        ("SQL Schema", test_sql_schema),
        ("Docker Files", test_docker_files),
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
        print("🎉 All basic structure tests passed!")
        print("\nThe project structure is correctly set up.")
        print("Next steps:")
        print("1. Install GPU dependencies (RAPIDS, Dask-CUDA, etc.)")
        print("2. Configure database credentials")
        print("3. Run full test suite with: ./run_tests.sh")
        return 0
    else:
        print("❌ Some basic structure tests failed.")
        print("\nPlease fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
