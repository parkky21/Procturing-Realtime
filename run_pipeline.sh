#!/bin/bash
# Wrapper script to run python with correct library paths for torchcodec on macOS

# Add Homebrew lib path to DYLD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH

echo "Running with DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH"

SCRIPT=${1:-test.py}
uv run "$SCRIPT"
