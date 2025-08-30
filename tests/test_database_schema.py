#!/usr/bin/env python3
"""
Test script for database schema validation.

This script validates the SQL schema structure without requiring
a database connection.
"""

import sys
import os
import re
from typing import Dict, Any, List

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_sql_file_exists() -> Dict[str, Any]:
    """Test that the SQL schema file exists."""
    try:
        sql_file_path = "docker/init.sql"
        if os.path.exists(sql_file_path):
            print(f"✓ SQL schema file exists: {sql_file_path}")
            return {"status": "success"}
        else:
            print(f"✗ SQL schema file not found: {sql_file_path}")
            return {"status": "failed", "error": "SQL file not found"}
    except Exception as e:
        print(f"✗ Error checking SQL file: {e}")
        return {"status": "failed", "error": str(e)}


def test_sql_syntax() -> Dict[str, Any]:
    """Test basic SQL syntax structure."""
    try:
        sql_file_path = "docker/init.sql"
        with open(sql_file_path, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        # Check for required SQL statements
        required_statements = [
            r'CREATE DATABASE IF NOT EXISTS',
            r'USE dynamic_stage0_db',
            r'CREATE TABLE IF NOT EXISTS processing_tasks',
            r'CREATE TABLE IF NOT EXISTS feature_status',
            r'CREATE OR REPLACE VIEW pending_tasks',
            r'CREATE OR REPLACE VIEW completed_tasks'
        ]
        
        for statement in required_statements:
            if re.search(statement, sql_content, re.IGNORECASE):
                print(f"  ✓ Found: {statement}")
            else:
                print(f"  ✗ Missing: {statement}")
                return {"status": "failed", "error": f"Missing SQL statement: {statement}"}
        
        print("✓ All required SQL statements found")
        return {"status": "success"}
        
    except Exception as e:
        print(f"✗ Error reading SQL file: {e}")
        return {"status": "failed", "error": str(e)}


def test_table_structure() -> Dict[str, Any]:
    """Test table structure and constraints."""
    try:
        sql_file_path = "docker/init.sql"
        with open(sql_file_path, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        # Check processing_tasks table structure
        processing_tasks_patterns = [
            r'task_id INT AUTO_INCREMENT PRIMARY KEY',
            r'currency_pair VARCHAR\(16\) NOT NULL UNIQUE',
            r'r2_path VARCHAR\(1024\) NOT NULL',
            r'added_timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP'
        ]
        
        for pattern in processing_tasks_patterns:
            if re.search(pattern, sql_content, re.IGNORECASE):
                print(f"  ✓ processing_tasks: {pattern}")
            else:
                print(f"  ✗ processing_tasks missing: {pattern}")
                return {"status": "failed", "error": f"Missing column in processing_tasks: {pattern}"}
        
        # Check feature_status table structure
        feature_status_patterns = [
            r'status_id INT AUTO_INCREMENT PRIMARY KEY',
            r'task_id INT NOT NULL',
            r'status ENUM.*PENDING.*RUNNING.*COMPLETED.*FAILED.*NOT NULL',
            r'start_time DATETIME NULL',
            r'end_time DATETIME NULL',
            r'hostname VARCHAR\(255\) NULL',
            r'error_message TEXT NULL',
            r'UNIQUE KEY uq_task_id \(task_id\)',
            r'FOREIGN KEY \(task_id\) REFERENCES processing_tasks\(task_id\)'
        ]
        
        for pattern in feature_status_patterns:
            if re.search(pattern, sql_content, re.IGNORECASE):
                print(f"  ✓ feature_status: {pattern}")
            else:
                print(f"  ✗ feature_status missing: {pattern}")
                return {"status": "failed", "error": f"Missing column/constraint in feature_status: {pattern}"}
        
        print("✓ All table structures are correct")
        return {"status": "success"}
        
    except Exception as e:
        print(f"✗ Error checking table structure: {e}")
        return {"status": "failed", "error": str(e)}


def test_indexes() -> Dict[str, Any]:
    """Test that required indexes are present."""
    try:
        sql_file_path = "docker/init.sql"
        with open(sql_file_path, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        # Check for required indexes
        required_indexes = [
            r'INDEX idx_currency_pair \(currency_pair\)',
            r'INDEX idx_added_timestamp \(added_timestamp\)',
            r'INDEX idx_status \(status\)',
            r'INDEX idx_start_time \(start_time\)',
            r'INDEX idx_hostname \(hostname\)'
        ]
        
        for index in required_indexes:
            if re.search(index, sql_content, re.IGNORECASE):
                print(f"  ✓ Index found: {index}")
            else:
                print(f"  ✗ Index missing: {index}")
                return {"status": "failed", "error": f"Missing index: {index}"}
        
        print("✓ All required indexes are present")
        return {"status": "success"}
        
    except Exception as e:
        print(f"✗ Error checking indexes: {e}")
        return {"status": "failed", "error": str(e)}


def test_views() -> Dict[str, Any]:
    """Test that views are properly defined."""
    try:
        sql_file_path = "docker/init.sql"
        with open(sql_file_path, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        # Check for views
        view_patterns = [
            r'CREATE OR REPLACE VIEW pending_tasks AS',
            r'CREATE OR REPLACE VIEW completed_tasks AS',
            r'COALESCE\(fs\.status, .*PENDING.*\) as current_status',
            r'TIMESTAMPDIFF\(SECOND, fs\.start_time, fs\.end_time\) as processing_time_seconds'
        ]
        
        for pattern in view_patterns:
            if re.search(pattern, sql_content, re.IGNORECASE):
                print(f"  ✓ View pattern found: {pattern}")
            else:
                print(f"  ✗ View pattern missing: {pattern}")
                return {"status": "failed", "error": f"Missing view pattern: {pattern}"}
        
        print("✓ All views are properly defined")
        return {"status": "success"}
        
    except Exception as e:
        print(f"✗ Error checking views: {e}")
        return {"status": "failed", "error": str(e)}


def test_comments() -> Dict[str, Any]:
    """Test that tables and columns have proper comments."""
    try:
        sql_file_path = "docker/init.sql"
        with open(sql_file_path, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        # Check for table comments
        table_comments = [
            r'COMMENT.*Defines the list of currency pairs to be processed by the pipeline',
            r'COMMENT.*Tracks the execution status and history for each processing task'
        ]
        
        for comment in table_comments:
            if re.search(comment, sql_content, re.IGNORECASE):
                print(f"  ✓ Table comment found: {comment}")
            else:
                print(f"  ✗ Table comment missing: {comment}")
                return {"status": "failed", "error": f"Missing table comment: {comment}"}
        
        # Check for column comments
        column_comments = [
            r'COMMENT.*Unique identifier for the task',
            r'COMMENT.*Currency pair to be processed.*EURUSD',
            r'COMMENT.*Base path in R2 for this pair data',
            r'COMMENT.*When the task was added',
            r'COMMENT.*Current processing status',
            r'COMMENT.*Timestamp when processing for this pair started',
            r'COMMENT.*Timestamp when processing for this pair ended',
            r'COMMENT.*ID of the vast.*ai instance or hostname running the task',
            r'COMMENT.*Detailed error message and traceback if status is FAILED'
        ]
        
        for comment in column_comments:
            if re.search(comment, sql_content, re.IGNORECASE):
                print(f"  ✓ Column comment found: {comment}")
            else:
                print(f"  ✗ Column comment missing: {comment}")
                return {"status": "failed", "error": f"Missing column comment: {comment}"}
        
        print("✓ All comments are present")
        return {"status": "success"}
        
    except Exception as e:
        print(f"✗ Error checking comments: {e}")
        return {"status": "failed", "error": str(e)}


def main():
    """Run all database schema tests."""
    print("=" * 60)
    print("Dynamic Stage 0 - Database Schema Test")
    print("=" * 60)
    
    # Run tests
    tests = [
        ("SQL File Exists", test_sql_file_exists),
        ("SQL Syntax", test_sql_syntax),
        ("Table Structure", test_table_structure),
        ("Indexes", test_indexes),
        ("Views", test_views),
        ("Comments", test_comments),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
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
        print("🎉 All database schema tests passed!")
        return 0
    else:
        print("❌ Some database schema tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
