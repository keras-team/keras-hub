#!/bin/bash -e

base_dir=$(dirname $(dirname $0))

isort --sl ${base_dir} --skip venv
black --line-length 80 ${base_dir} --exclude venv

for i in $(find ${base_dir} -name '*.py' -not -path "*/venv/*"); do
  if ! grep -q Copyright $i; then
    echo $i
    cat shell/copyright.txt $i >$i.new && mv $i.new $i
  fi
done

flake8 ${base_dir} --exclude venv
