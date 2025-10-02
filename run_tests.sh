#!/bin/bash
# Script to run all CLaSP tests

echo "Running CLaSP unit tests..."
echo "=========================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Counter for test results
PASSED=0
FAILED=0

# Function to run a test and check result
run_test() {
    echo -e "\n${NC}Running $1..."
    if python $1; then
        echo -e "${GREEN}✓ $1 passed${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗ $1 failed${NC}"
        ((FAILED++))
    fi
}

# Run all test files
run_test "tests/test_dataloaders.py"
run_test "tests/test_models.py"
run_test "tests/test_losses.py"

# Summary
echo -e "\n=========================="
echo -e "Test Summary:"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"

# Exit with appropriate code
if [ $FAILED -eq 0 ]; then
    echo -e "\n${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "\n${RED}Some tests failed!${NC}"
    exit 1
fi