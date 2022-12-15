#!/usr/bin/env bash

BYellow='\033[1;33m'
Color_Off='\033[0m'

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DEFAULT_DATA_DIR="$SCRIPT_DIR/omnigibson_data"
DATA_PATH=${1:-$DEFAULT_DATA_DIR}

echo -e "${BYellow}IMPORTANT: Saving OmniGibson assets at ${DATA_PATH}."
echo -e  "You can change this path by providing your desired path as an argument"
echo -e "to the run_docker script you are using.${Color_Off}"
echo ""

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
    --network=host --rm -it stanfordvl/omnigibson bash -c "source ~/.bashrc && jupyter lab --allow-root"
