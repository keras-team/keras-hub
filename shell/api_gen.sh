#!/bin/bash
set -Eeuo pipefail

base_dir=$(dirname $(dirname $0))

echo "Generating api directory with public APIs..."
# Generate API Files - try python3 first, fall back to python
if command -v python3 > /dev/null 2>&1; then
    python3 "${base_dir}"/api_gen.py
elif command -v python > /dev/null 2>&1; then
    python "${base_dir}"/api_gen.py
else
    echo "Error: Neither python3 nor python found"
    exit 1
fi

# Format code because `api_gen.py` might order
# imports differently.
echo "Formatting api directory..."
(SKIP=api-gen pre-commit run --files $(find "${base_dir}"/keras_hub/api -type f) --hook-stage pre-commit || true) > /dev/null
