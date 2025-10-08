#!/usr/bin/env bash
set -e -o pipefail

BYellow='\033[1;33m'
Color_Off='\033[0m'

# Parse the command line arguments.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DEFAULT_DATA_DIR="$SCRIPT_DIR/omnigibson_data"
DATA_PATH=$DEFAULT_DATA_DIR
GUI=true

# Parse command line arguments
while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
        -h|--headless)
        GUI=false
        shift
        ;;
        *)
        DATA_PATH="$1"
        shift
        ;;
    esac
done

echo -e "${BYellow}IMPORTANT: Saving OmniGibson assets at ${DATA_PATH}."
echo -e "You can change this path by providing your desired path as an argument"
echo -e "to the run_docker script you are using. Also note that Docker containers"
echo -e "are incompatible with AFS/NFS drives, so please make sure that this path"
echo -e "points to a local filesystem. ${Color_Off}"
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

docker pull stanfordvl/omnigibson:latest
DOCKER_DISPLAY=""
OMNIGIBSON_HEADLESS=1
if [ "$GUI" = true ] ; then
    xhost +local:root
    DOCKER_DISPLAY=$DISPLAY
    OMNIGIBSON_HEADLESS=0
fi
docker run \
    --gpus all \
    --privileged \
    -e DISPLAY=${DOCKER_DISPLAY} \
    -e OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS} \
    -v $DATA_PATH/datasets:/data \
    -v $DATA_PATH/isaac-sim/cache/kit:/isaac-sim/kit/cache/Kit:rw \
    -v $DATA_PATH/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v $DATA_PATH/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v $DATA_PATH/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v $DATA_PATH/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v $DATA_PATH/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v $DATA_PATH/isaac-sim/config:/root/.nvidia-omniverse/config:rw \
    -v $DATA_PATH/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v $DATA_PATH/isaac-sim/documents:/root/Documents:rw \
    --network=host --rm -it stanfordvl/omnigibson:latest
if [ "$GUI" = true ] ; then
    xhost -local:root
fi