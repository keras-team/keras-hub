#!/bin/bash -e

base_dir=$(dirname $(dirname $0))
targets="${base_dir}/*.py ${base_dir}/examples/ ${base_dir}/keras_nlp/"

isort --sp "${base_dir}/setup.cfg" --sl ${targets}
black --line-length 80 ${targets}

for i in $(find ${targets} -name '*.py'); do
  if ! grep -q Copyright $i; then
    echo $i
    cat shell/copyright.txt $i >$i.new && mv $i.new $i
  fi
done

flake8 --config "${base_dir}/setup.cfg" ${targets}
