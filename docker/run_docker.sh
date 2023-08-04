#!/usr/bin/env bash

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

ICD_PATH_1="/usr/share/vulkan/icd.d/nvidia_icd.json"
ICD_PATH_2="/etc/vulkan/icd.d/nvidia_icd.json"
LAYERS_PATH_1="/usr/share/vulkan/icd.d/nvidia_layers.json"
LAYERS_PATH_2="/usr/share/vulkan/implicit_layer.d/nvidia_layers.json"
LAYERS_PATH_3="/etc/vulkan/implicit_layer.d/nvidia_layers.json"
EGL_VENDOR_PATH="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"

# Find the ICD file
if [ -e "$ICD_PATH_1" ]; then
    ICD_PATH=$ICD_PATH_1
elif [ -e "$ICD_PATH_2" ]; then
    ICD_PATH=$ICD_PATH_2
else
    echo "Missing nvidia_icd.json file.";
    echo "Typical paths:";
    echo "- /usr/share/vulkan/icd.d/nvidia_icd.json or";
    echo "- /etc/vulkan/icd.d/nvidia_icd.json";
    echo "You can google nvidia_icd.json for your distro to find the correct path.";
    echo "Consider updating your driver to 525 if you cannot find the file.";
    echo "To continue update the ICD_PATH_1 at the top of the run_docker.sh file and retry";
    exit;
fi

# Find the layers file
if [ -e "$LAYERS_PATH_1" ]; then
    LAYERS_PATH=$LAYERS_PATH_1
elif [ -e "$LAYERS_PATH_2" ]; then
    LAYERS_PATH=$LAYERS_PATH_2
elif [ -e "$LAYERS_PATH_3" ]; then
    LAYERS_PATH=$LAYERS_PATH_3
else
    echo "Missing nvidia_layers.json file."
    echo "Typical paths:";
    echo "- /usr/share/vulkan/icd.d/nvidia_layers.json";
    echo "- /usr/share/vulkan/implicit_layer.d/nvidia_layers.json";
    echo "- /etc/vulkan/implicit_layer.d/nvidia_layers.json";
    echo "You can google nvidia_layers.json for your distro to find the correct path.";
    echo "Consider updating your driver to 525 if you cannot find the file.";
    echo "To continue update the LAYERS_PATH_1 at the top of the run_docker.sh file and retry";
    exit;
fi 

if [ ! -e "$EGL_VENDOR_PATH" ]; then
    echo "Missing ${EGL_VENDOR_PATH} file."
    echo "(default path: /usr/share/vulkan/icd.d/nvidia_icd.json)";
    echo "To continue update the EGL_VENDOR_PATH at the top of the run_docker.sh file and retry";
    exit;
fi 

# Move directories from their legacy paths.
if [ -e "${DATA_PATH}/og_dataset" ]; then
    mv "${DATA_PATH}/og_dataset" "${DATA_PATH}/datasets/og_dataset"
fi
if [ -e "${DATA_PATH}/assets" ]; then
    mv "${DATA_PATH}/assets" "${DATA_PATH}/datasets/assets"
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
    -v ${ICD_PATH}:/etc/vulkan/icd.d/nvidia_icd.json \
    -v ${LAYERS_PATH}:/etc/vulkan/implicit_layer.d/nvidia_layers.json \
    -v ${EGL_VENDOR_PATH}:/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
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