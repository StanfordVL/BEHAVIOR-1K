#!/usr/bin/env bash
set -e -o pipefail

docker build \
    -t stanfordvl/omnigibson:colab-docker \
    -f docker/prod.Dockerfile \
    .

docker build \
    -t stanfordvl/omnigibson-colab \
    -f docker/colab.Dockerfile \
    .