#!/bin/bash -ex

base_dir=$(dirname $0)

for platform in "jax" "tensorflow" "torch"; do
    pushd "${base_dir}/${platform}" > /dev/null
    gcloud builds submit \
        --region=us-west1 \
        --project=keras-team-test \
        --tag "us-west1-docker.pkg.dev/keras-team-test/keras-nlp-test/keras-nlp-image-${platform}:deps" \
        --timeout=30m
    popd
done
