# Makefile for mukh project testing and development

.PHONY: help install test test-unit test-integration test-face test-deepfake test-coverage test-fast clean lint format type-check docs

# Default target
help:
	@echo "Available targets:"
	@echo "  install        Install development dependencies"
	@echo "  test           Run all tests"
	@echo "  test-unit      Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-face      Run face detection tests"
	@echo "  test-deepfake  Run deepfake detection tests"
	@echo "  test-coverage  Run tests with coverage report"
	@echo "  test-fast      Run fast tests (exclude slow ones)"
	@echo "  lint           Run linting checks"
	@echo "  format         Format code with black and isort"
	@echo "  type-check     Run type checking with mypy"
	@echo "  clean          Clean test artifacts and cache"
	@echo "  docs           Build documentation"

# Install development dependencies
install:
	pip install -r requirements_dev.txt
	pip install -e .

# Test targets
test:
	pytest tests/ -v

test-unit:
	pytest tests/ -v -m "unit"

test-integration:
	pytest tests/ -v -m "integration"

test-face:
	pytest tests/face_detection/ -v

test-deepfake:
	pytest tests/deepfake_detection/ -v

test-coverage:
	pytest tests/ -v --cov=mukh --cov-report=html --cov-report=term-missing

test-fast:
	pytest tests/ -v -m "not slow"

# Code quality targets
lint:
	flake8 mukh/ tests/
	black --check mukh/ tests/
	isort --check mukh/ tests/

format:
	black mukh/ tests/
	isort mukh/ tests/

type-check:
	mypy mukh/

# Cleanup
clean:
	rm -rf htmlcov/
	rm -f coverage.xml
	rm -f .coverage
	rm -rf .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

# Documentation
docs:
	mkdocs build

# Development workflow
dev-setup: install
	@echo "Development environment setup complete"

# Run all checks (useful for CI)
check-all: lint type-check test
	@echo "All checks passed!"
