#!/usr/bin/env bash

docker push stanfordvl/omnigibson:latest
docker push stanfordvl/omnigibson:$(sed -ne "s/.*version= *['\"]\([^'\"]*\)['\"] *.*/\1/p" setup.py)
docker push stanfordvl/omnigibson-dev:latest
