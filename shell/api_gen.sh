#!/bin/bash
set -Eeuo pipefail

base_dir=$(dirname $(dirname $0))

if ! command -v pre-commit 2>&1 >/dev/null
then
    echo 'Please `pip install pre-commit` to run api_gen.sh.'
    exit 1
fi

echo "Generating api directory with public APIs..."
# Generate API Files
python3 "${base_dir}"/api_gen.py

# Format code because `api_gen.py` might order
# imports differently.
echo "Formatting api directory..."
(SKIP=api-gen pre-commit run --files $(find "${base_dir}"/keras_hub/api -type f) --hook-stage pre-commit || true) > /dev/null
