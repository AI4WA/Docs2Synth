.PHONY: help install format lint type-check test clean pre-commit-install pre-commit-run

help:
	@echo "Available commands:"
	@echo "  make install              Install the package and dev dependencies"
	@echo "  make format               Format code with black and isort"
	@echo "  make lint                 Run flake8 linter"
	@echo "  make type-check           Run mypy type checker"
	@echo "  make test                 Run pytest tests"
	@echo "  make clean                Remove build artifacts and cache files"
	@echo "  make pre-commit-install   Install pre-commit hooks"
	@echo "  make pre-commit-run       Run pre-commit on all files"
	@echo "  make all                  Run format, lint, type-check, and test"

install:
	pip install -e ".[dev]"

format:
	@echo "Running isort..."
	isort .
	@echo "Running black..."
	black .

lint:
	@echo "Running flake8..."
	flake8 Docs2Synth tests

type-check:
	@echo "Running mypy..."
	mypy Docs2Synth

test:
	@echo "Running pytest..."
	pytest

clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build dist

pre-commit-install:
	@echo "Installing pre-commit hooks..."
	pre-commit install

pre-commit-run:
	@echo "Running pre-commit on all files..."
	pre-commit run --all-files

all: format lint type-check test
	@echo "All checks completed!"
