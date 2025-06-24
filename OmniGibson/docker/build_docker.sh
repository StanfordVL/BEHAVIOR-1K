#!/usr/bin/env bash
set -e -o pipefail

docker build \
    -t stanfordvl/omnigibson:latest \
    -t stanfordvl/omnigibson:$(sed -ne "s/.*version= *['\"]\([^'\"]*\)['\"] *.*/\1/p" setup.py) \
    -f docker/prod.Dockerfile \
    .

# Pass the DEV_MODE=1 arg to the docker build command to build the development image
docker build \
    -t stanfordvl/omnigibson-dev:latest \
    -t stanfordvl/omnigibson-dev:$(sed -ne "s/.*version= *['\"]\([^'\"]*\)['\"] *.*/\1/p" setup.py) \
    -f docker/prod.Dockerfile \
    --build-arg DEV_MODE=1 \
    .

docker build \
    -t stanfordvl/omnigibson-vscode:latest \
    -f docker/vscode.Dockerfile \
    .

docker build \
    -t stanfordvl/omnigibson-colab:latest \
    -f docker/colab.Dockerfile \
    .