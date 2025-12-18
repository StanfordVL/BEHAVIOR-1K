#!/usr/bin/env bash
set -e -o pipefail

docker build \
    -t stanfordvl/behavior:latest \
    -t stanfordvl/behavior:$(sed -ne "s/.*version= *['\"]\([^'\"]*\)['\"] *.*/\1/p" OmniGibson/setup.py) \
    -f docker/Dockerfile \
    .

# Pass the DEV_MODE=1 arg to the docker build command to build the development image
docker build \
    -t stanfordvl/behavior-dev:latest \
    -t stanfordvl/behavior-dev:$(sed -ne "s/.*version= *['\"]\([^'\"]*\)['\"] *.*/\1/p" OmniGibson/setup.py) \
    -f docker/Dockerfile \
    --build-arg DEV_MODE=1 \
    .

docker build \
    -t stanfordvl/behavior-vscode:latest \
    -f docker/vscode.Dockerfile \
    .

docker build \
    -t stanfordvl/behavior-colab:latest \
    -f docker/colab.Dockerfile \
    .