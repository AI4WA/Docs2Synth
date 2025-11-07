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
flake8 docs2synth tests
echo "✅ flake8 passed"
echo ""

echo "4️⃣  Cleaning notebook outputs..."
if find notebooks -name "*.ipynb" -type f 2>/dev/null | grep -q .; then
    find notebooks -name "*.ipynb" -type f | while read -r notebook; do
        echo "  Cleaning: $notebook"
        jupyter nbconvert --clear-output --inplace "$notebook" 2>/dev/null || true
    done
    echo "✅ notebooks cleaned"
else
    echo "  No notebooks found, skipping"
fi
echo ""

echo "5️⃣  Running tests with pytest..."
pytest
echo "✅ tests passed"
echo ""

echo "🎉 All checks passed! Ready to push."
