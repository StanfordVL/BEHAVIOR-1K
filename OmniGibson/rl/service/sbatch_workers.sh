#!/usr/bin/env bash
#SBATCH --account=cvgl
#SBATCH --partition=svl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=100G
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --array=0-1

# Wait for learner
current_dir=$(pwd)
exit_code=1
cd /cvgl2/u/${USER}/OmniGibson/rl/service/telegym
while [ $exit_code -eq 1 ]; do
    python3 -c "import grpc; from protos import environment_pb2, environment_pb2_grpc; channel = grpc.insecure_channel('${1}'); stub = environment_pb2_grpc.EnvironmentRegistrationServiceStub(channel); request = environment_pb2.Empty(); stub.RegisterEnvironmentAvailable(request)"
    exit_code=$?
    sleep 60
done
cd $current_dir

IMAGE_PATH="/cvgl2/u/cgokmen/omnigibson.sqsh"
GPU_ID=$(nvidia-smi -L | grep -oP '(?<=GPU-)[a-fA-F0-9\-]+' | head -n 1)
ISAAC_CACHE_PATH="/scr-ssd/${SLURM_JOB_USER}/isaac_cache_${GPU_ID}"

# Pick a port using the array index
BASE_PORT=$2
for i in {0..2}; do
    # Define env kwargs to pass
    declare -A ENVS=(
        [NVIDIA_DRIVER_CAPABILITIES]=all
        [NVIDIA_VISIBLE_DEVICES]=0
        [DISPLAY]=""
        [OMNIGIBSON_HEADLESS]=1
    )
    ENV_KWARGS=""
    for env_var in "${!ENVS[@]}"; do
        # Add to env kwargs we'll pass to enroot command later
        ENV_KWARGS="${ENV_KWARGS} --env ${env_var}=${ENVS[${env_var}]}"
    done

    # Define mounts to create (maps local directory to container directory)
    declare -A MOUNTS=(
        [/scr-ssd/og-data-0-2-1]=/data
        [${ISAAC_CACHE_PATH}/isaac-sim/kit/cache/Kit]=/isaac-sim/kit/cache/Kit
        [${ISAAC_CACHE_PATH}/isaac-sim/cache/ov]=/root/.cache/ov
        [${ISAAC_CACHE_PATH}/isaac-sim/cache/pip]=/root/.cache/pip
        [${ISAAC_CACHE_PATH}/isaac-sim/cache/glcache]=/root/.cache/nvidia/GLCache
        [${ISAAC_CACHE_PATH}/isaac-sim/cache/computecache]=/root/.nv/ComputeCache
        [${ISAAC_CACHE_PATH}/isaac-sim/logs]=/root/.nvidia-omniverse/logs
        [${ISAAC_CACHE_PATH}/isaac-sim/config]=/root/.nvidia-omniverse/config
        [${ISAAC_CACHE_PATH}/isaac-sim/data]=/root/.local/share/ov/data
        [${ISAAC_CACHE_PATH}/isaac-sim/documents]=/root/Documents
        # Feel free to include lines like the below to mount a workspace or a custom OG version
        [/cvgl2/u/${USER}/OmniGibson]=/omnigibson-src
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
    CONTAINER_NAME=omnigibson_${GPU_ID}_${i}
    enroot create --force --name ${CONTAINER_NAME} ${IMAGE_PATH}

    # Remove leading space in string
    ENV_KWARGS="${ENV_KWARGS:1}"
    MOUNT_KWARGS="${MOUNT_KWARGS:1}"


    # The last line here is the command you want to run inside the container.
    # Here I'm running some unit tests.
    ENROOT_MOUNT_HOME=no enroot start \
        --root \
        --rw \
        ${ENV_KWARGS} \
        ${MOUNT_KWARGS} \
        ${CONTAINER_NAME} \
        micromamba run -n omnigibson /bin/bash --login -c "source /isaac-sim/setup_conda_env.sh && pip install gymnasium grpcio grpcio-tools stable_baselines3 wandb && cd /omnigibson-src/rl/service && WANDB_API_KEY=$2 python -u omni_grpc_worker.py $1" \
        > "output_${SLURM_ARRAY_TASK_ID}_${i}.txt" 2>&1 &
done

wait

for i in {0..2}; do
    CONTAINER_NAME=omnigibson_${GPU_ID}_${i}
    # Clean up the image if possible.
    enroot remove -f ${CONTAINER_NAME}
done
