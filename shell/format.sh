#!/bin/bash
set -Euo pipefail

base_dir=$(dirname $(dirname $0))
cd $base_dir

ruff check --config "${base_dir}/pyproject.toml" --fix .
ruff format --config "${base_dir}/pyproject.toml" .
