#!/bin/bash

# VALUES TO SET #######################################
DOCKER_IMAGE="stanfordvl/ig_pipeline:latest"     # Can also use, e.g.: stanfordvl/omnigibson:latest
SCR_DIR="/scr"
CVGL2_DIR="/cvgl2"
#######################################################
# YOU SHOULD NOT HAVE TO TOUCH ANYTHING BELOW HERE :) #


BYellow='\033[1;33m'
Color_Off='\033[0m'

# Parse the command line arguments.
ENV_KWARGS=""
case $1 in
    -n|--no-omniverse)
    ENV_KWARGS="${ENV_KWARGS} --env OMNIGIBSON_NO_OMNIVERSE=1"
    shift
    ;;
esac


SCRIPT_DIR="/scr/BEHAVIOR-1K/asset_pipeline/b1k_pipeline/docker"
DATA_PATH="/scr/og-docker-data/datasets"
SQSH_SOURCE="${SCRIPT_DIR}/ig_pipeline.sqsh"

# Define env kwargs to pass
declare -A ENVS=(
    [NVIDIA_DRIVER_CAPABILITIES]=all
    [DISPLAY]=""
    [OMNIGIBSON_HEADLESS]=1
)
for env_var in "${!ENVS[@]}"; do
    # Add to env kwargs we'll pass to enroot command later
    ENV_KWARGS="${ENV_KWARGS} --env ${env_var}=${ENVS[${env_var}]}"
done

# Define mounts to create (maps local directory to container directory)
declare -A MOUNTS=(
    [${SCR_DIR}]=/scr
    [${DATA_PATH}]=/data
    ["/scr/OmniGibson_old"]=/omnigibson-src
    ["/scr/BEHAVIOR-1K/bddl"]=/bddl-src
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

# Remove leading space in string
ENV_KWARGS="${ENV_KWARGS:1}"
MOUNT_KWARGS="${MOUNT_KWARGS:1}"


# Create the image if it doesn't already exist
WORKER_CNT=$1; shift
for ((i = 1 ; i <= $WORKER_CNT ; i++));
do
    CONTAINER_NAME=ig_pipeline_${i}
    echo "Creating container ${CONTAINER_NAME}..."
    # enroot create --name ${CONTAINER_NAME} ${SQSH_SOURCE}

    if [ `expr $i % 2` == 0 ]
    then
        GPU=0
    else
        GPU=1
    fi

    echo "Launching job"
    export ENROOT_RESTRICT_DEV=y
    enroot start \
        --root \
        --rw \
        ${ENV_KWARGS} \
        --env OMNIGIBSON_GPU_ID=${GPU} \
        ${MOUNT_KWARGS} \
        ${CONTAINER_NAME} $@ &> /dev/null &
done

# Uncomment this to remove the image after
# for i in {1..8}
# do
#     enroot remove -f ig_pipeline_${i}
# done

