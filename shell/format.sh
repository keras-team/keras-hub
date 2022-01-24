#!/bin/bash -e

base_dir=$(dirname $(dirname $0))

isort --sl ${base_dir}
black --line-length 80 ${base_dir}

for i in $(find ${base_dir} -name '*.py'); do
  if ! grep -q Copyright $i; then
    echo $i
    cat shell/copyright.txt $i >$i.new && mv $i.new $i
  fi
done

flake8 ${base_dir}
