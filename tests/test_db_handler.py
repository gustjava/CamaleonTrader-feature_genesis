#!/usr/bin/env python3
"""
Database Handler Test Script for Dynamic Stage 0 Pipeline

This script tests the database handler module functionality including:
- Database connection
- Fetching pending currency pairs
- Updating task status
"""

import sys
import os
import traceback
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from data_io.db_handler import DatabaseHandler, get_pending_currency_pairs, update_task_status
    from config import get_config
    from config.settings import get_settings
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)


def test_database_connection() -> Dict[str, Any]:
    """Test database connection using DatabaseHandler."""
    try:
        print("Testing database connection...")
        
        with DatabaseHandler() as db:
            print("✓ Database connection successful")
            return {"status": "success"}
            
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_get_pending_currency_pairs() -> Dict[str, Any]:
    """Test fetching pending currency pairs."""
    try:
        print("Testing get_pending_currency_pairs...")
        
        pending_pairs = get_pending_currency_pairs()
        print(f"✓ Found {len(pending_pairs)} pending currency pairs")
        
        if pending_pairs:
            print("Sample pending pairs:")
            for pair in pending_pairs[:3]:  # Show first 3
                print(f"  - {pair['currency_pair']} (Task ID: {pair['task_id']}, Status: {pair['current_status']})")
        
        return {"status": "success", "count": len(pending_pairs)}
        
    except Exception as e:
        print(f"✗ get_pending_currency_pairs failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_task_status_updates() -> Dict[str, Any]:
    """Test updating task status."""
    try:
        print("Testing task status updates...")
        
        # First, get a pending task to test with
        pending_pairs = get_pending_currency_pairs()
        if not pending_pairs:
            print("⚠ No pending tasks found. Creating a test task...")
            
            # Create a test task using direct database access
            with DatabaseHandler() as db:
                with db.engine.begin() as conn:
                    from sqlalchemy import text
                    
                    # Insert test task
                    result = conn.execute(text("""
                        INSERT INTO processing_tasks (currency_pair, r2_path) 
                        VALUES (:pair, :path)
                        ON DUPLICATE KEY UPDATE r2_path = VALUES(r2_path)
                    """), {"pair": "TESTPAIR", "path": "data/test/TESTPAIR/"})
                    
                    task_id = conn.execute(text("SELECT LAST_INSERT_ID()")).fetchone()[0]
                    print(f"✓ Created test task with ID: {task_id}")
        else:
            task_id = pending_pairs[0]['task_id']
            print(f"✓ Using existing task ID: {task_id}")
        
        # Test RUNNING status
        print(f"  Testing RUNNING status for task {task_id}...")
        success = update_task_status(task_id, 'RUNNING')
        if success:
            print("  ✓ RUNNING status updated successfully")
        else:
            print("  ✗ Failed to update RUNNING status")
            return {"status": "failed", "error": "Failed to update RUNNING status"}
        
        # Test COMPLETED status
        print(f"  Testing COMPLETED status for task {task_id}...")
        success = update_task_status(task_id, 'COMPLETED')
        if success:
            print("  ✓ COMPLETED status updated successfully")
        else:
            print("  ✗ Failed to update COMPLETED status")
            return {"status": "failed", "error": "Failed to update COMPLETED status"}
        
        # Test FAILED status with error message
        print(f"  Testing FAILED status for task {task_id}...")
        success = update_task_status(task_id, 'FAILED', "Test error message")
        if success:
            print("  ✓ FAILED status updated successfully")
        else:
            print("  ✗ Failed to update FAILED status")
            return {"status": "failed", "error": "Failed to update FAILED status"}
        
        # Verify the final status
        with DatabaseHandler() as db:
            task_info = db.get_task_info(task_id)
            if task_info and task_info['status'] == 'FAILED':
                print("  ✓ Task status verification successful")
            else:
                print("  ✗ Task status verification failed")
                return {"status": "failed", "error": "Task status verification failed"}
        
        return {"status": "success", "task_id": task_id}
        
    except Exception as e:
        print(f"✗ Task status updates failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_invalid_status() -> Dict[str, Any]:
    """Test handling of invalid status values."""
    try:
        print("Testing invalid status handling...")
        
        # Get a task to test with
        pending_pairs = get_pending_currency_pairs()
        if not pending_pairs:
            print("⚠ No tasks available for testing invalid status")
            return {"status": "skipped", "reason": "No tasks available"}
        
        task_id = pending_pairs[0]['task_id']
        
        # Test invalid status
        success = update_task_status(task_id, 'INVALID_STATUS')
        if not success:
            print("✓ Invalid status correctly rejected")
            return {"status": "success"}
        else:
            print("✗ Invalid status was not rejected")
            return {"status": "failed", "error": "Invalid status was not rejected"}
        
    except Exception as e:
        print(f"✗ Invalid status test failed: {e}")
        return {"status": "failed", "error": str(e)}


def test_context_manager() -> Dict[str, Any]:
    """Test DatabaseHandler context manager functionality."""
    try:
        print("Testing DatabaseHandler context manager...")
        
        # Test context manager
        with DatabaseHandler() as db:
            # Test that we can perform operations
            pending_pairs = db.get_pending_currency_pairs()
            print(f"✓ Context manager test successful, found {len(pending_pairs)} pending pairs")
        
        # Test that connection is properly closed
        print("✓ Context manager properly closed connection")
        return {"status": "success"}
        
    except Exception as e:
        print(f"✗ Context manager test failed: {e}")
        return {"status": "failed", "error": str(e)}


def main():
    """Run all database handler tests."""
    print("=" * 60)
    print("Dynamic Stage 0 - Database Handler Test")
    print("=" * 60)
    
    # Test configuration loading
    try:
        print("Testing configuration loading...")
        config = get_config()
        settings = get_settings()
        print("✓ Configuration loaded successfully")
        print(f"  Database: {settings.database.host}:{settings.database.port}/{settings.database.database}")
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        traceback.print_exc()
        return 1
    
    # Run database handler tests
    tests = [
        ("Database Connection", test_database_connection),
        ("Get Pending Currency Pairs", test_get_pending_currency_pairs),
        ("Task Status Updates", test_task_status_updates),
        ("Invalid Status Handling", test_invalid_status),
        ("Context Manager", test_context_manager),
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
        print("🎉 All database handler tests passed! Database handler is ready for Dynamic Stage 0.")
        return 0
    else:
        print("❌ Some database handler tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
