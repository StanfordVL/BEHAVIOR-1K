#!/usr/bin/env bash
set -e -o pipefail

docker push stanfordvl/omnigibson:latest
docker push stanfordvl/omnigibson:$(sed -ne "s/.*version= *['\"]\([^'\"]*\)['\"] *.*/\1/p" setup.py)
docker push stanfordvl/omnigibson-dev:latest
