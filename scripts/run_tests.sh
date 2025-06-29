#!/bin/bash

# Test runner script for mukh project
# This script provides convenient ways to run different types of tests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  all          Run all tests"
    echo "  unit         Run only unit tests"
    echo "  integration  Run only integration tests"
    echo "  face         Run face detection tests"
    echo "  deepfake     Run deepfake detection tests"
    echo "  coverage     Run tests with coverage report"
    echo "  fast         Run tests excluding slow tests"
    echo "  clean        Clean test artifacts"
    echo "  install      Install test dependencies"
    echo "  help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 all           # Run all tests"
    echo "  $0 unit          # Run only unit tests"
    echo "  $0 coverage      # Run with coverage"
    echo "  $0 face          # Run face detection tests only"
}

# Function to check if pytest is installed
check_pytest() {
    if ! command -v pytest &> /dev/null; then
        print_error "pytest is not installed. Run '$0 install' to install dependencies."
        exit 1
    fi
}

# Function to install test dependencies
install_deps() {
    print_status "Installing test dependencies..."
    pip install -r requirements_dev.txt
    print_success "Test dependencies installed successfully"
}

# Function to clean test artifacts
clean_artifacts() {
    print_status "Cleaning test artifacts..."
    
    # Remove coverage files
    rm -rf htmlcov/
    rm -f coverage.xml
    rm -f .coverage
    
    # Remove pytest cache
    rm -rf .pytest_cache/
    
    # Remove __pycache__ directories
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    
    # Remove .pyc files
    find . -name "*.pyc" -delete 2>/dev/null || true
    
    print_success "Test artifacts cleaned"
}

# Function to run all tests
run_all_tests() {
    print_status "Running all tests..."
    if ! pytest tests/ -v; then
        print_error "Tests failed"
        exit 1
    fi
    print_success "All tests passed"
}

run_parallel_tests() {
    print_status "Running tests in parallel..."
    pytest tests/ -v -n auto
}

# Function to run unit tests only
run_unit_tests() {
    print_status "Running unit tests..."
    pytest tests/ -v -m "unit"
}

# Function to run integration tests only
run_integration_tests() {
    print_status "Running integration tests..."
    pytest tests/ -v -m "integration"
}

# Function to run face detection tests
run_face_tests() {
    print_status "Running face detection tests..."
    pytest tests/face_detection/ -v
}

# Function to run deepfake detection tests
run_deepfake_tests() {
    print_status "Running deepfake detection tests..."
    pytest tests/deepfake_detection/ -v
}

# Function to run tests with coverage
run_coverage_tests() {
    print_status "Running tests with coverage..."
    pytest tests/ -v --cov=mukh --cov-report=html --cov-report=term-missing
    print_success "Coverage report generated in htmlcov/"
}

# Function to run fast tests (excluding slow ones)
run_fast_tests() {
    print_status "Running fast tests (excluding slow tests)..."
    pytest tests/ -v -m "not slow"
}

# Main script logic
case "${1:-help}" in
    all)
        check_pytest
        run_all_tests
        ;;
    unit)
        check_pytest
        run_unit_tests
        ;;
    integration)
        check_pytest
        run_integration_tests
        ;;
    face)
        check_pytest
        run_face_tests
        ;;
    deepfake)
        check_pytest
        run_deepfake_tests
        ;;
    coverage)
        check_pytest
        run_coverage_tests
        ;;
    fast)
        check_pytest
        run_fast_tests
        ;;
    clean)
        clean_artifacts
        ;;
    install)
        install_deps
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown option: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
