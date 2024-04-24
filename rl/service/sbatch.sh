#!/usr/bin/env bash
#SBATCH --account=cvgl
#SBATCH --partition=svl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --gres=gpu:titanrtx:2

IMAGE_PATH="/cvgl2/u/cgokmen/omnigibson.sqsh"
GPU_ID=$(nvidia-smi -L | grep -oP '(?<=GPU-)[a-fA-F0-9\-]+' | head -n 1)
ISAAC_CACHE_PATH="/scr-ssd/${SLURM_JOB_USER}/isaac_cache_${GPU_ID}"

if netstat -tuln | grep ":$2" > /dev/null; then
    echo "Port $2 is in use."
    exit 1
else
    echo "Using unused port $2."
fi

if netstat -tuln | grep ":$3" > /dev/null; then
    echo "Port $3 is in use."
    exit 1
else
    echo "Using unused port $3."
fi

for i in {0..1}; do
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

    if [ $i -eq 0 ]; then
        enroot start \
        --root \
        --rw \
        ${ENV_KWARGS} \
        ${MOUNT_KWARGS} \
        ${CONTAINER_NAME} \
        micromamba run -n omnigibson /bin/bash --login -c "source /isaac-sim/setup_conda_env.sh && pip install gymnasium grpcio grpcio-tools stable_baselines3 wandb tensorboard moviepy && cd /omnigibson-src/workspace && WANDB_API_KEY=$4 python -u /omnigibson-src/rl/service/omni_grpc_learner.py --n_envs $1 --port $2 --eval_port $3 --sweep_id $5" > "output_learner.txt" 2>&1 &
    else
        enroot start \
        --root \
        --rw \
        ${ENV_KWARGS} \
        ${MOUNT_KWARGS} \
        ${CONTAINER_NAME} \
        micromamba run -n omnigibson /bin/bash --login -c "source /isaac-sim/setup_conda_env.sh && pip install gymnasium grpcio grpcio-tools stable_baselines3 wandb tensorboard moviepy && cd /omnigibson-src/workspace && WANDB_API_KEY=$4 python -u /omnigibson-src/rl/service/omni_grpc_worker.py 0.0.0.0:$3 --render" > "output_eval.txt" 2>&1 &
    fi
    
done

wait

for i in {0..1}; do
    CONTAINER_NAME=omnigibson_${GPU_ID}_${i}

    # Clean up the image if possible.
    enroot remove -f ${CONTAINER_NAME}
done
