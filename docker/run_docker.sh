#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DEFAULT_DIR = "${SCRIPT_DIR}/omnigibson_data"
DATA_PATH=${1:-${DEFAULT_DIR}}

echo "The NVIDIA Omniverse License Agreement (EULA) must be accepted before"
echo "Omniverse Kit can start. The license terms for this product can be viewed at"
echo "https://docs.omniverse.nvidia.com/app_isaacsim/common/NVIDIA_Omniverse_License_Agreement.html"

while true; do
    read -p "Do you accept the Omniverse EULA? [y/n] " yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

docker run \
    --gpus all \
    --privileged \
    -e DISPLAY \
    -e OMNIGIBSON_HEADLESS=1 \
    -v $DATA_PATH:/data \
    --network=host --rm -it stanfordvl/omnigibson:latest
