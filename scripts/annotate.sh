#!/bin/bash
# Quick launcher for annotation tool

# Check if data directory is provided
DATA_DIR="${1:-./data/processed/dev}"

# Check if config exists
if [ -f "config.yml" ]; then
    echo "📝 Launching annotation tool with config.yml..."
    python -m docs2synth.cli annotate "$DATA_DIR"
else
    echo "📝 Launching annotation tool..."
    python -m docs2synth.cli annotate "$DATA_DIR"
fi
