# :material-server-network: **Running on a Compute Cluster**

_This documentation is a work in progress._

**OmniGibson** can be run on a compute cluster. Currently, only a SLURM-based cluster server is supported in our documentation, but a similar approach can be followed on other cluster systems that support running Docker containers.

## Running on a SLURM Cluster

We assume the SLURM cluster using the _enroot_ container software, which is a replacement for Docker that allows containers to be run as the current user rather than as root. _enroot_ needs to be installed on your SLURM cluster by an administrator.

**All of the below commands assume version 1.0.0 of OmniGibson and will not be updated. Please update the paths to match the version you are installing in order to ensure you use the correct version of the dataset going forward.**

With enroot installed, you can follow the below steps to run OmniGibson on SLURM: 

1. Download the dataset to a location that is accessible by cluster nodes. To do this, you can use the download_datasets.py script inside OmniGibson's scripts directory, and move it to the right spot later. In the below example, /cvgl/ is a networked drive that is accessible by the cluster nodes. **For Stanford users, this step is already done for SVL and Viscam nodes**
```{.shell .annotate}
OMNIGIBSON_NO_OMNIVERSE=1 python omnigibson/download_datasets.py
mv omnigibson/data /cvgl/group/Gibson/og-data-1-0-0
```

2. (Optional) Distribute the dataset to the individual nodes.
This will make load times much better than reading from a network drive. To do this, run the below command on your SLURM head node (replace `svl` with your partition name and `cvgl` with your account name, as well as the paths with the respective network and local paths). Confirm via `squeue -u $USER` that all jobs have finished. **This step is already done for SVL and Viscam nodes**
```{.shell .annotate}
sinfo -p svl -o "%N,%n" -h | \
  sed s/,.*//g | \
  xargs -L1 -I{} \
    sbatch \
      --account=cvgl --partition=svl --nodelist={} --mem=8G --cpus-per-task=4 \
      --wrap 'cp -R /cvgl/group/Gibson/og-data-1-0-0 /scr-ssd/og-data-1-0-0'
```

3. Download your desired image to a location that is accessible by the cluster nodes. (Replace the path with your own path, and feel free to replace `latest` with your desired branch tag). You have the option to mount code (meaning you don't need the container to come with all the code you want to run, just the right dependencies / environment setup)
```{.shell .annotate}
enroot import --output /cvgl2/u/cgokmen/omnigibson.sqsh docker://stanfordvl/omnigibson:action-primitives
```

4. (Optional) If you intend to mount code onto the container, make it available at a location that is accessible by the cluster nodes. You can mount arbitrary code, and you can also mount a custom version of OmniGibson (for the latter, you need to make sure you mount your copy of OmniGibson at /omnigibson-src inside the container). For example:
```{.shell .annotate}
git clone https://github.com/StanfordVL/OmniGibson.git /cvgl2/u/cgokmen/OmniGibson
```

5. Create your launch script. You can start with a copy of the script below. If you want to launch multiple workers, increase the job array option. You should keep the setting at at least 1 GPU per node, but can feel free to edit other settings. You can mount any additional code as you'd like, and you can change the entrypoint such that the container runs your mounted code upon launch. See the mounts section for an example. A copy of this script can be found in docker/sbatch_example.sh
```{.shell .annotate}
#!/usr/bin/env bash
#SBATCH --account=cvgl
#SBATCH --partition=svl --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --gres=gpu:2080ti:1

IMAGE_PATH="/cvgl2/u/cgokmen/omnigibson.sqsh"
GPU_ID=$(nvidia-smi -L | grep -oP '(?<=GPU-)[a-fA-F0-9\-]+' | head -n 1)
ISAAC_CACHE_PATH="/scr-ssd/${SLURM_JOB_USER}/isaac_cache_${GPU_ID}"

# Define env kwargs to pass
declare -A ENVS=(
    [NVIDIA_DRIVER_CAPABILITIES]=all
    [NVIDIA_VISIBLE_DEVICES]=0
    [DISPLAY]=""
    [OMNIGIBSON_HEADLESS]=1
)
for env_var in "${!ENVS[@]}"; do
    # Add to env kwargs we'll pass to enroot command later
    ENV_KWARGS="${ENV_KWARGS} --env ${env_var}=${ENVS[${env_var}]}"
done

# Define mounts to create (maps local directory to container directory)
declare -A MOUNTS=(
    [/scr-ssd/og-data-1-0-0]=/data
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
    # [/cvgl2/u/cgokmen/OmniGibson]=/omnigibson-src
    # [/cvgl2/u/cgokmen/my-project]=/my-project
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
CONTAINER_NAME=omnigibson_${GPU_ID}
enroot create --force --name ${CONTAINER_NAME} ${IMAGE_PATH}

# Remove leading space in string
ENV_KWARGS="${ENV_KWARGS:1}"
MOUNT_KWARGS="${MOUNT_KWARGS:1}"

# The last line here is the command you want to run inside the container.
# Here I'm running some unit tests.
enroot start \
    --root \
    --rw \
    ${ENV_KWARGS} \
    ${MOUNT_KWARGS} \
    ${CONTAINER_NAME} \
    source /isaac-sim/setup_conda_env.sh && pytest tests/test_object_states.py

# Clean up the image if possible.
enroot remove -f ${CONTAINER_NAME}
```

6. Launch your job using `sbatch your_script.sh` - and profit!