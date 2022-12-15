#!/usr/bin/env bash

docker build \
    -t stanfordvl/omnigibson:latest \
    -t stanfordvl/omnigibson:$(sed -ne "s/.*version= *['\"]\([^'\"]*\)['\"] *.*/\1/p" setup.py) \
    -f docker/Dockerfile \
    .