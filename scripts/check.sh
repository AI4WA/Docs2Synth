#!/bin/bash
# Run all code quality checks
# Usage: ./scripts/check.sh

set -e

echo "🔍 Running code quality checks..."
echo ""

echo "1️⃣  Sorting imports with isort..."
isort .
echo "✅ isort done"
echo ""

echo "2️⃣  Formatting with black..."
black .
echo "✅ black done"
echo ""

echo "3️⃣  Linting with flake8..."
flake8 Docs2Synth tests
echo "✅ flake8 passed"
echo ""

echo "4️⃣  Running tests with pytest..."
pytest
echo "✅ tests passed"
echo ""

echo "🎉 All checks passed! Ready to push."
