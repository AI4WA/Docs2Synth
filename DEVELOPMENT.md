# Development Guide

## Quick Start

```bash
# Install development dependencies
make install

# Install pre-commit hooks (one-time setup)
make pre-commit-install
```

## Running Checks Locally

### Simple workflow (recommended)
```bash
# Run all checks before pushing
make check
```

This will:
1. Format code with black and isort
2. Check linting with flake8
3. Run type checking with mypy
4. Run all tests with pytest

### Individual commands
```bash
make format       # Auto-format code (black + isort)
make lint         # Check code style (flake8)
make type-check   # Check types (mypy)
make test         # Run tests (pytest)
```

### Pre-commit hooks
Once installed, hooks run automatically on `git commit`:
- Checks formatting, linting, and types
- Fixes issues automatically when possible
- Prevents committing if there are errors

To manually run pre-commit on all files:
```bash
make pre-commit-run
```

## Tool Configuration

All tools are configured in `pyproject.toml` and `.flake8`:

- **Black**: Line length 88, Python 3.8-3.12
- **isort**: Black-compatible profile
- **flake8**: Relaxed rules, ignores docstring checks
- **mypy**: Type checking with reasonable defaults
- **pytest**: Tests in `tests/` directory

## CI/CD

GitHub Actions automatically runs all checks on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`

Make sure `make check` passes locally before pushing!

## Common Issues

### Black or isort errors
```bash
# Auto-fix formatting issues
make format
```

### Flake8 errors
Most common issues:
- Line too long (handled by black)
- Unused imports in `__init__.py` (ignored)
- Missing docstrings (ignored for now)

### Mypy errors
- Add `# type: ignore` for unavoidable issues
- Use `--ignore-missing-imports` in config for missing stubs

## Clean Up

```bash
make clean  # Remove cache files and build artifacts
```
