#!/usr/bin/env python3
"""
Database Test Script for Dynamic Stage 0 Pipeline

This script tests the MySQL database connection and verifies that the
required tables and schema are properly set up.
"""

import sys
import os
import traceback
from typing import Dict, Any, List, Tuple

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from config import get_config
    from config.settings import get_settings
except ImportError as e:
    print(f"Error importing configuration: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)


def test_database_connection() -> Dict[str, Any]:
    """Test database connection and basic operations."""
    try:
        from sqlalchemy import create_engine, text
        from sqlalchemy.exc import SQLAlchemyError
        
        config = get_config()
        settings = get_settings()
        
        print(f"Testing database connection to: {settings.database.host}:{settings.database.port}")
        
        # Create engine
        engine = create_engine(
            config.get_database_url(),
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.max_overflow,
            pool_timeout=settings.database.pool_timeout,
            pool_recycle=settings.database.pool_recycle
        )
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT VERSION()"))
            version = result.fetchone()[0]
            print(f"✓ Database connection successful")
            print(f"  MySQL version: {version}")
            
            # Test database exists
            result = conn.execute(text("SELECT DATABASE()"))
            current_db = result.fetchone()[0]
            print(f"  Current database: {current_db}")
            
            # Check if required tables exist
            result = conn.execute(text("""
                SELECT TABLE_NAME 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = :db_name
                AND TABLE_NAME IN ('processing_tasks', 'feature_status')
            """), {"db_name": current_db})
            
            existing_tables = [row[0] for row in result.fetchall()]
            print(f"  Existing tables: {existing_tables}")
            
            if len(existing_tables) == 2:
                print("✓ All required tables exist")
            else:
                missing_tables = set(['processing_tasks', 'feature_status']) - set(existing_tables)
                print(f"✗ Missing tables: {missing_tables}")
                return {"status": "failed", "error": f"Missing tables: {missing_tables}"}
        
        engine.dispose()
        return {"status": "success", "version": version, "database": current_db}
        
    except SQLAlchemyError as e:
        print(f"✗ Database connection failed: {e}")
        return {"status": "failed", "error": str(e)}
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return {"status": "failed", "error": str(e)}


def test_table_schema() -> Dict[str, Any]:
    """Test that table schemas match the expected structure."""
    try:
        from sqlalchemy import create_engine, text, inspect
        from sqlalchemy.exc import SQLAlchemyError
        
        config = get_config()
        engine = create_engine(config.get_database_url())
        inspector = inspect(engine)
        
        print("\n--- Testing Table Schemas ---")
        
        # Expected schema for processing_tasks
        expected_processing_tasks = {
            'task_id': {'type': 'INT', 'primary_key': True, 'nullable': False},
            'currency_pair': {'type': 'VARCHAR', 'primary_key': False, 'nullable': False},
            'r2_path': {'type': 'VARCHAR', 'primary_key': False, 'nullable': False},
            'added_timestamp': {'type': 'DATETIME', 'primary_key': False, 'nullable': False}
        }
        
        # Expected schema for feature_status
        expected_feature_status = {
            'status_id': {'type': 'INT', 'primary_key': True, 'nullable': False},
            'task_id': {'type': 'INT', 'primary_key': False, 'nullable': False},
            'status': {'type': 'ENUM', 'primary_key': False, 'nullable': False},
            'start_time': {'type': 'DATETIME', 'primary_key': False, 'nullable': True},
            'end_time': {'type': 'DATETIME', 'primary_key': False, 'nullable': True},
            'hostname': {'type': 'VARCHAR', 'primary_key': False, 'nullable': True},
            'error_message': {'type': 'TEXT', 'primary_key': False, 'nullable': True}
        }
        
        # Test processing_tasks table
        print("Testing processing_tasks table schema...")
        columns = inspector.get_columns('processing_tasks')
        column_info = {col['name']: col for col in columns}
        
        for expected_col, expected_props in expected_processing_tasks.items():
            if expected_col not in column_info:
                print(f"✗ Missing column: {expected_col}")
                return {"status": "failed", "error": f"Missing column: {expected_col}"}
            
            col = column_info[expected_col]
            print(f"  ✓ Column {expected_col}: {col['type']}")
        
        # Test feature_status table
        print("Testing feature_status table schema...")
        columns = inspector.get_columns('feature_status')
        column_info = {col['name']: col for col in columns}
        
        for expected_col, expected_props in expected_feature_status.items():
            if expected_col not in column_info:
                print(f"✗ Missing column: {expected_col}")
                return {"status": "failed", "error": f"Missing column: {expected_col}"}
            
            col = column_info[expected_col]
            print(f"  ✓ Column {expected_col}: {col['type']}")
        
        engine.dispose()
        print("✓ All table schemas are correct")
        return {"status": "success"}
        
    except SQLAlchemyError as e:
        print(f"✗ Schema test failed: {e}")
        return {"status": "failed", "error": str(e)}
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return {"status": "failed", "error": str(e)}


def test_sample_data_operations() -> Dict[str, Any]:
    """Test basic CRUD operations with sample data."""
    try:
        from sqlalchemy import create_engine, text
        from sqlalchemy.exc import SQLAlchemyError
        
        config = get_config()
        engine = create_engine(config.get_database_url())
        
        print("\n--- Testing Sample Data Operations ---")
        
        with engine.begin() as conn:
            # Insert sample task
            result = conn.execute(text("""
                INSERT INTO processing_tasks (currency_pair, r2_path) 
                VALUES (:pair, :path)
                ON DUPLICATE KEY UPDATE r2_path = VALUES(r2_path)
            """), {"pair": "TESTUSD", "path": "data/test/TESTUSD/"})
            
            task_id = conn.execute(text("SELECT LAST_INSERT_ID()")).fetchone()[0]
            print(f"✓ Inserted test task with ID: {task_id}")
            
            # Insert status
            conn.execute(text("""
                INSERT INTO feature_status (task_id, status, start_time, hostname)
                VALUES (:task_id, 'RUNNING', NOW(), 'test-host')
            """), {"task_id": task_id})
            
            print("✓ Inserted test status")
            
            # Query using view
            result = conn.execute(text("SELECT * FROM pending_tasks WHERE currency_pair = :pair"), 
                                {"pair": "TESTUSD"})
            rows = result.fetchall()
            print(f"✓ Query from pending_tasks view: {len(rows)} rows")
            
            # Update status to completed
            conn.execute(text("""
                UPDATE feature_status 
                SET status = 'COMPLETED', end_time = NOW()
                WHERE task_id = :task_id
            """), {"task_id": task_id})
            
            print("✓ Updated status to completed")
            
            # Clean up test data
            conn.execute(text("DELETE FROM feature_status WHERE task_id = :task_id"), 
                        {"task_id": task_id})
            conn.execute(text("DELETE FROM processing_tasks WHERE task_id = :task_id"), 
                        {"task_id": task_id})
            
            print("✓ Cleaned up test data")
        
        engine.dispose()
        return {"status": "success"}
        
    except SQLAlchemyError as e:
        print(f"✗ Data operations test failed: {e}")
        return {"status": "failed", "error": str(e)}
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return {"status": "failed", "error": str(e)}


def main():
    """Run all database tests."""
    print("=" * 60)
    print("Dynamic Stage 0 - Database Test")
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
    
    # Run database tests
    tests = [
        ("Database Connection", test_database_connection),
        ("Table Schema", test_table_schema),
        ("Sample Data Operations", test_sample_data_operations),
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
        print("🎉 All database tests passed! Database is ready for Dynamic Stage 0.")
        return 0
    else:
        print("❌ Some database tests failed. Please check the errors above.")
        print("\nTo initialize the database schema, run:")
        print("  mysql -h <host> -P <port> -u <user> -p < database/docker/init.sql")
        return 1


if __name__ == "__main__":
    sys.exit(main())
