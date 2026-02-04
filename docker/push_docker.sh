#!/usr/bin/env bash
set -e -o pipefail

docker push stanfordvl/behavior:latest
docker push stanfordvl/behavior:$(sed -ne "s/.*version= *['\"]\([^'\"]*\)['\"] *.*/\1/p" OmniGibson/setup.py)
docker push stanfordvl/behavior-dev:latest
