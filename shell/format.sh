#!/bin/bash -e

base_dir=$(dirname $(dirname $0))
targets="${base_dir}"

isort --sp "${base_dir}/pyproject.toml" ${targets}
black --config "${base_dir}/pyproject.toml" ${targets}
flake8 --config "${base_dir}/setup.cfg" ${targets}
