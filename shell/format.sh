#!/bin/bash -e

base_dir=$(dirname $(dirname $0))
targets="${base_dir}"

isort --sp "${base_dir}/pyproject.toml" ${targets}
black --config "${base_dir}/pyproject.toml" ${targets}

for i in $(find ${targets} -name '*.py'); do
  if ! grep -q Copyright $i; then
    echo $i
    cat shell/copyright.txt $i >$i.new && mv $i.new $i
  fi
done

flake8 --config "${base_dir}/setup.cfg" ${targets}
