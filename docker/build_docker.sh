#!/usr/bin/env bash
set -e -o pipefail

docker build \
    -t stanfordvl/omnigibson:latest \
    -t stanfordvl/omnigibson:$(sed -ne "s/.*version= *['\"]\([^'\"]*\)['\"] *.*/\1/p" setup.py) \
    -f docker/prod.Dockerfile \
    .

docker build \
    -t stanfordvl/omnigibson-vscode:latest \
    -f docker/vscode.Dockerfile \
    .

docker build \
    -t stanfordvl/omnigibson-colab:latest \
    -f docker/colab.Dockerfile \
    .