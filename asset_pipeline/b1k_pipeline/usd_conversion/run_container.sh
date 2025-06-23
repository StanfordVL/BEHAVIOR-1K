#!/usr/bin/env bash
set -e

BYellow='\033[1;33m'
Color_Off='\033[0m'

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DEFAULT_DATA_DIR="$SCRIPT_DIR/omnigibson_data"
PIPELINE_PATH=$(readlink -f "$SCRIPT_DIR/../../")

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
    exit 1;
fi 
if [ ! -e "$LAYERS_PATH" ]; then
    echo "Missing ${LAYERS_PATH} file."
    echo "(default path: /usr/share/vulkan/icd.d/nvidia_layers.json)";
    echo "In some distributions this file will be at /etc/vulkan/implicit_layer.d/";
    echo "Consider updating your driver to 525 if you cannot find the file.";
    echo "To continue update the file path at the top of the run_docker.sh file and retry";
    exit 1;
fi 
if [ ! -e "$EGL_VENDOR_PATH" ]; then
    echo "Missing ${EGL_VENDOR_PATH} file."
    echo "(default path: /usr/share/vulkan/icd.d/nvidia_icd.json)";
    echo "To continue update the file path at the top of the run_docker.sh file and retry";
    exit 1;
fi 

# docker pull stanfordvl/omnigibson:latest
docker run \
    --gpus all \
    --privileged \
    -e DISPLAY \
    -e OMNIGIBSON_HEADLESS=1 \
    -v $DEFAULT_DATA_DIR/datasets:/data \
    -v $PIPELINE_PATH:/ig_pipeline \
    -v ${ICD_PATH}:/etc/vulkan/icd.d/nvidia_icd.json \
    -v ${LAYERS_PATH}:/etc/vulkan/implicit_layer.d/nvidia_layers.json \
    -v ${EGL_VENDOR_PATH}:/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    -v $DEFAULT_DATA_DIR/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v $DEFAULT_DATA_DIR/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v $DEFAULT_DATA_DIR/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v $DEFAULT_DATA_DIR/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v $DEFAULT_DATA_DIR/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v $DEFAULT_DATA_DIR/isaac-sim/config:/root/.nvidia-omniverse/config:rw \
    -v $DEFAULT_DATA_DIR/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v $DEFAULT_DATA_DIR/isaac-sim/documents:/root/Documents:rw \
    --network=host --rm -it stanfordvl/omnigibson:latest \
    /bin/bash -i /ig_pipeline/b1k_pipeline/usd_conversion/run_in_container.sh
