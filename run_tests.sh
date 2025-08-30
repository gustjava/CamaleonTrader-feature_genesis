#!/bin/bash

# Dynamic Stage 0 - Test Runner Script
# This script runs all tests for the pipeline components

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TESTS_DIR="tests"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_PATH="${PROJECT_ROOT}"

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Dynamic Stage 0 - Test Runner${NC}"
echo -e "${BLUE}================================${NC}"

# Function to run a test and capture its result
run_test() {
    local test_file="$1"
    local test_name="$2"
    
    echo -e "\n${YELLOW}--- Running ${test_name} ---${NC}"
    
    # Change to project root and run test
    cd "${PROJECT_ROOT}"
    
    if python "${test_file}"; then
        echo -e "${GREEN}✓ ${test_name} PASSED${NC}"
        return 0
    else
        echo -e "${RED}✗ ${test_name} FAILED${NC}"
        return 1
    fi
}

# Function to check if a test file exists
test_exists() {
    local test_file="$1"
    if [[ -f "${test_file}" ]]; then
        return 0
    else
        return 1
    fi
}

# Main test execution
echo -e "${BLUE}Starting test execution...${NC}"

# Initialize counters
total_tests=0
passed_tests=0
failed_tests=0

# Define test files and their descriptions
declare -A tests=(
    ["tests/test_basic_structure.py"]="Basic Project Structure"
    ["tests/test_r2_paths_simple.py"]="R2 Path Validation"
    ["tests/test_local_loader.py"]="Local Data Loader"
    ["tests/test_main_structure.py"]="Main Pipeline Structure"
    ["tests/test_database_schema.py"]="Database Schema Validation"
    ["tests/test_gpu_setup.py"]="GPU Environment Setup"
    ["tests/test_database.py"]="Database Connection and Schema"
    ["tests/test_db_handler.py"]="Database Handler Operations"
    ["tests/test_dask_cluster.py"]="Dask-CUDA Cluster Orchestration"
    ["tests/test_r2_loader.py"]="R2 Data Loader"
    ["tests/test_pipeline_integration.py"]="Pipeline Integration"
)

# Run each test
for test_file in "${!tests[@]}"; do
    test_name="${tests[$test_file]}"
    
    if test_exists "${test_file}"; then
        total_tests=$((total_tests + 1))
        
        if run_test "${test_file}" "${test_name}"; then
            passed_tests=$((passed_tests + 1))
        else
            failed_tests=$((failed_tests + 1))
        fi
    else
        echo -e "${YELLOW}⚠ Test file not found: ${test_file}${NC}"
    fi
done

# Summary
echo -e "\n${BLUE}================================${NC}"
echo -e "${BLUE}TEST EXECUTION SUMMARY${NC}"
echo -e "${BLUE}================================${NC}"

echo -e "Total tests run: ${total_tests}"
echo -e "${GREEN}Passed: ${passed_tests}${NC}"
echo -e "${RED}Failed: ${failed_tests}${NC}"

if [[ ${total_tests} -gt 0 ]]; then
    success_rate=$(echo "scale=1; ${passed_tests} * 100 / ${total_tests}" | bc -l 2>/dev/null || echo "0")
    echo -e "Success rate: ${success_rate}%"
fi

# Final result
if [[ ${failed_tests} -eq 0 && ${total_tests} -gt 0 ]]; then
    echo -e "\n${GREEN}🎉 All tests passed! Dynamic Stage 0 is ready for execution.${NC}"
    echo -e "\n${BLUE}Next steps:${NC}"
    echo -e "1. Configure environment variables if needed"
    echo -e "2. Initialize database schema: mysql -h <host> -P <port> -u <user> -p < docker/init.sql"
    echo -e "3. Run the pipeline: python orchestration/main.py"
    exit 0
else
    echo -e "\n${RED}❌ Some tests failed. Please check the errors above.${NC}"
    echo -e "\n${YELLOW}Troubleshooting tips:${NC}"
    echo -e "- Ensure all dependencies are installed"
    echo -e "- Check database connectivity"
    echo -e "- Verify R2 credentials are configured"
    echo -e "- Ensure GPU environment is properly set up"
    exit 1
fi
