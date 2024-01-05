#!/usr/bin/env bash

IMAGE_PATH="/cvgl2/u/cgokmen/omnigibson.sqsh"
ISAAC_CACHE_PATH="/scr/og-docker-data/isaac_sim"

JOB_ID=$2
GPU_ID=$(( JOB_ID % 2))

# Define env kwargs to pass
declare -A ENVS=(
    [NVIDIA_DRIVER_CAPABILITIES]=all
    [NVIDIA_VISIBLE_DEVICES]=$GPU_ID
    [DISPLAY]=""
    [OMNIGIBSON_HEADLESS]=1
)
for env_var in "${!ENVS[@]}"; do
    # Add to env kwargs we'll pass to enroot command later
    ENV_KWARGS="${ENV_KWARGS} --env ${env_var}=${ENVS[${env_var}]}"
done

# Define mounts to create (maps local directory to container directory)
declare -A MOUNTS=(
    [${ISAAC_CACHE_PATH}/kit/cache/Kit]=/isaac-sim/kit/cache/Kit
    [${ISAAC_CACHE_PATH}/cache/ov]=/root/.cache/ov
    [${ISAAC_CACHE_PATH}/cache/pip]=/root/.cache/pip
    [${ISAAC_CACHE_PATH}/cache/glcache]=/root/.cache/nvidia/GLCache
    [${ISAAC_CACHE_PATH}/cache/computecache]=/root/.nv/ComputeCache
    [${ISAAC_CACHE_PATH}/logs]=/root/.nvidia-omniverse/logs
    [${ISAAC_CACHE_PATH}/config]=/root/.nvidia-omniverse/config
    [${ISAAC_CACHE_PATH}/data]=/root/.local/share/ov/data
    [${ISAAC_CACHE_PATH}/documents]=/root/Documents
    [/scr/og-docker-data/datasets]=/data
    [/cvgl2/u/cgokmen/OmniGibson]=/omnigibson-src
)

MOUNT_KWARGS=""
for mount in "${!MOUNTS[@]}"; do
    # Verify mount path in local directory exists, otherwise, create it
    if [ ! -e "$mount" ]; then
        mkdir -p ${mount}
    fi
    # Add to mount kwargs we'll pass to enroot command later
    MOUNT_KWARGS="${MOUNT_KWARGS} --mount ${mount}:${MOUNTS[${mount}]}"
done

# Create the image if it doesn't already exist
CONTAINER_NAME=omnigibson_${JOB_ID}
enroot create --force --name ${CONTAINER_NAME} ${IMAGE_PATH}

# Remove leading space in string
ENV_KWARGS="${ENV_KWARGS:1}"
MOUNT_KWARGS="${MOUNT_KWARGS:1}"

# Pick a port using the array index
BASE_PORT=50100
WORKER_PORT=$((BASE_PORT + JOB_ID))

# The last line here is the command you want to run inside the container.
# Here I'm running some unit tests.
ENROOT_MOUNT_HOME=no enroot start \
    --root \
    --rw \
    ${ENV_KWARGS} \
    ${MOUNT_KWARGS} \
    ${CONTAINER_NAME} \
    micromamba run -n omnigibson /bin/bash --login -c "source /isaac-sim/setup_conda_env.sh && pip install gymnasium grpcio grpcio-tools stable_baselines3 && cd /omnigibson-src/rl/service && python -u omni_grpc_worker.py $1 ${WORKER_PORT}"

# Clean up the image if possible.
enroot remove -f ${CONTAINER_NAME}
