#!/usr/bin/env bash

BYellow='\033[1;33m'
Color_Off='\033[0m'

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DEFAULT_DATA_DIR="$SCRIPT_DIR/omnigibson_data"
DATA_PATH=${1:-$DEFAULT_DATA_DIR}

ICD_PATH="/usr/share/vulkan/icd.d/nvidia_icd.json"
LAYERS_PATH="/usr/share/vulkan/icd.d/nvidia_layers.json"
EGL_VENDOR_PATH="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"

# Assert the presence of relevant Vulkan files
if [ ! -e "$ICD_PATH" ]; then
    echo "Missing ${ICD_PATH} file."
    echo "(default path: /usr/share/vulkan/icd.d/nvidia_icd.json)";
    echo "In some distributions this file will be at /etc/vulkan/icd.d/";
    echo "Consider updating your driver to 525 if you cannot find the file.";
    echo "To continue update the file path at the top of the run_docker.sh file and retry";
    exit;
fi 
if [ ! -e "$LAYERS_PATH" ]; then
    echo "Missing ${LAYERS_PATH} file."
    echo "(default path: /usr/share/vulkan/icd.d/nvidia_layers.json)";
    echo "In some distributions this file will be at /etc/vulkan/implicit_layer.d/";
    echo "Consider updating your driver to 525 if you cannot find the file.";
    echo "To continue update the file path at the top of the run_docker.sh file and retry";
    exit;
fi 
if [ ! -e "$EGL_VENDOR_PATH" ]; then
    echo "Missing ${EGL_VENDOR_PATH} file."
    echo "(default path: /usr/share/vulkan/icd.d/nvidia_icd.json)";
    echo "To continue update the file path at the top of the run_docker.sh file and retry";
    exit;
fi 

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
docker run \
    --gpus all \
    --privileged \
    -e DISPLAY \
    -e OMNIGIBSON_HEADLESS=1 \
    -v $DATA_PATH/datasets:/data \
    -v ${ICD_PATH}:/etc/vulkan/icd.d/nvidia_icd.json \
    -v ${LAYERS_PATH}:/etc/vulkan/implicit_layer.d/nvidia_layers.json \
    -v ${EGL_VENDOR_PATH}:/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    -v $DATA_PATH/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v $DATA_PATH/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v $DATA_PATH/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v $DATA_PATH/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v $DATA_PATH/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v $DATA_PATH/isaac-sim/config:/root/.nvidia-omniverse/config:rw \
    -v $DATA_PATH/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v $DATA_PATH/isaac-sim/documents:/root/Documents:rw \
    --network=host --rm -it stanfordvl/omnigibson:latest
