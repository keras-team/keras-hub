#!/bin/bash -e

base_dir=$(dirname $(dirname $0))

isort --sl -c "${base_dir}" --skip venv
if ! [ $? -eq 0 ]; then
  echo "Please run \"./shell/format.sh\" to format the code."
  exit 1
fi
flake8 "${base_dir}" --exclude venv
if ! [ $? -eq 0 ]; then
  echo "Please fix the code style issue."
  exit 1
fi
black --check --line-length 80 "${base_dir}" --exclude venv
if ! [ $? -eq 0 ]; then
  echo "Please run \"./shell/format.sh\" to format the code."
    exit 1
fi
for i in $(find "${base_dir}" -name '*.py' -not -path "*/venv/*"); do
  if ! grep --exclude-dir=venv -q Copyright  $i; then
    echo "Please run \"./shell/format.sh\" to format the code."
    exit 1
  fi
done
