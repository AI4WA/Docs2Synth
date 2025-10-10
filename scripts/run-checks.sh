#!/bin/bash
# Run all code quality checks locally

set -e

echo "🔍 Running all code quality checks..."
echo ""

echo "1️⃣  Formatting with black..."
black .
echo "✅ Black formatting done"
echo ""

echo "2️⃣  Sorting imports with isort..."
isort .
echo "✅ Import sorting done"
echo ""

echo "3️⃣  Linting with flake8..."
flake8 Docs2Synth tests
echo "✅ Flake8 linting passed"
echo ""

echo "4️⃣  Type checking with mypy..."
mypy Docs2Synth
echo "✅ Mypy type checking passed"
echo ""

echo "5️⃣  Running tests with pytest..."
pytest
echo "✅ All tests passed"
echo ""

echo "🎉 All checks completed successfully!"
