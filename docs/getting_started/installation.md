# :material-tools: **Installation**

## :material-note-text: **Requirements**

Please make sure your system meets the following specs:

- [x] **OS:** Ubuntu 20.04+ / Windows 10+
- [x] **RAM:** 32GB+
- [x] **GPU:** NVIDIA RTX 2070+
- [x] **VRAM:** 8GB+

??? question "Why these specs?"
    
    The core simulator behind BEHAVIOR, **`OmniGibson`**, is built upon NVIDIA's [Omniverse](https://www.nvidia.com/en-us/omniverse/) and [Isaac Sim](https://developer.nvidia.com/isaac-sim) platforms, so we inherit their dependencies. For more information, please see [Isaac Sim's Requirements](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/requirements.html).

## :material-laptop: **Setup**

Choose your installation method:

- **🔧 Development Setup (Linux / Windows, Recommended)**: Use our unified setup script for editable installs and development.
<!-- - **🐳 Docker Install (Linux only)**: Quick setup using our pre-built docker image. -->

!!! note "Installation Options"
    - **PyPI packages** (`omnigibson`, `bddl`, `joylo`) are coming soon for easier production deployments
    - **Docker installation** is temporarily unavailable while we update it for the monorepo structure
    
    For now, please use the development setup which installs from source.

!!! tip ""
    === "🔧 Development Setup (Linux / Windows)"

        **Use this method if you want to modify the code or need the latest development features.**

        1. Clone the BEHAVIOR-1K repository:
            
            Clone the latest stable release (recommended):

            ```shell
            git clone -b v3.7.2 https://github.com/StanfordVL/BEHAVIOR-1K.git
            cd BEHAVIOR-1K
            ```
            
            If you want the latest development features and updates (potentially less stable), directly clone the main branch instead:
            
            ```shell
            git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
            cd BEHAVIOR-1K
            ```
                
        2. Run the unified setup script:

            === "Linux"

                ```shell
                # Full installation with all components
                ./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval --primitives

                # Core components only (without evaluation and teleoperation)
                ./setup.sh --new-env --omnigibson --bddl --dataset

                # Headless/automated installation (auto-accepts Conda TOS, NVIDIA Isaac Sim EULA, and BEHAVIOR Dataset License)
                ./setup.sh --new-env --omnigibson --bddl --dataset \
                           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos

                # See all options
                ./setup.sh --help
                ```

            === "Windows"

                ```powershell
                # Full installation with all components
                .\setup.ps1 -NewEnv -OmniGibson -BDDL -JoyLo -Dataset -Eval -Primitives

                # Core components only (without evaluation and teleoperation)
                .\setup.ps1 -NewEnv -OmniGibson -BDDL -Dataset

                # Headless/automated installation (auto-accepts Conda TOS, NVIDIA Isaac Sim EULA, and BEHAVIOR Dataset License)
                .\setup.ps1 -NewEnv -OmniGibson -BDDL -Dataset `
                            -AcceptCondaTos -AcceptNvidiaEula -AcceptDatasetTos

                # See all options
                .\setup.ps1 -Help
                ```

        3. Activate the environment:

            ```shell
            conda activate behavior
            ```


        !!! info "What does the setup script do?"
            
            - Creates a new conda environment named `behavior` (when using `--new-env`)
            - Installs all selected components in editable mode
            - Downloads Isaac Sim and BEHAVIOR datasets (if requested)

        !!! tip "Using an existing Python environment"
            
            You can omit `--new-env` to use your current Python environment instead of creating a new conda environment:
            ```shell
            # Linux: Install in current environment
            ./setup.sh --omnigibson --bddl --joylo --dataset
            
            # Skip confirmation prompt with --confirm-no-conda
            ./setup.sh --omnigibson --bddl --confirm-no-conda
            ```
            The script will prompt for confirmation if not in a conda environment.

        !!! note "Windows Note"
            
            Run PowerShell as Administrator and set execution policy if needed:
            ```powershell
            Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
            ```
<!--
    === "🐳 Install with Docker (Linux only)"

        Install **`OmniGibson`** with Docker is supported for **🐧 Linux** only.

        ??? info "Need to install docker or NVIDIA docker?"
            
            ```{.shell .annotate}
            # Install docker
            curl https://get.docker.com | sh && sudo systemctl --now enable docker

            # Install nvidia-docker runtime
            distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
                && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
                sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
                && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
                sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
            sudo apt-get update
            sudo apt-get install -y nvidia-docker2 # install
            sudo systemctl restart docker # restart docker engine
            ```

        1. Install our docker launching scripts:
            ```shell
            curl -LJO https://raw.githubusercontent.com/StanfordVL/OmniGibson/main/docker/run_docker.sh
            chmod a+x run_docker.sh
            ```

            ??? question annotate "What is being installed?"

                Our docker image automatically ships with a pre-configured conda virtual environment named `omnigibson` with Isaac Sim and **`OmniGibson`** pre-installed. Upon running the first time, our scene and object assets will automatically be downloaded as well.

        2. Then, simply launch the shell script:

            === "Headless"

                ```{.shell .annotate}
                sudo ./run_docker.sh -h <ABS_DATA_PATH> # (1)!
                ```

                1.  `<ABS_DATA_PATH>` specifies the **absolute** path data will be stored on your machine (if no `<ABS_DATA_PATH>` is specified, it defaults to `./omnigibson_data`). This needs to be called each time the docker container is run!


            === "GUI"

                ```{.shell .annotate}
                sudo ./run_docker.sh <ABS_DATA_PATH> # (1)!
                ```

                1.  `<ABS_DATA_PATH>` specifies the **absolute** path data will be stored on your machine (if no `<ABS_DATA_PATH>` is specified, it defaults to `./omnigibson_data`). This needs to be called each time the docker container is run!

            
            ??? warning annotate "Are you using NFS or AFS?"

                Docker containers are unable to access NFS or AFS drives, so if `run_docker.sh` are located on an NFS / AFS partition, please set `<DATA_PATH>` to an alternative data directory located on a non-NFS / AFS partition.
-->


## :material-earth: **Explore `BEHAVIOR`!**

!!! warning annotate "Expect slowdown during first execution"

    Omniverse requires some one-time startup setup when **`OmniGibson`** is imported for the first time.
    
    The process could take up to 5 minutes. This is expected behavior, and should only occur once!

**`BEHAVIOR`** is now successfully installed! You can try teleoperating one of our robots:

```{.shell .annotate}
python -m omnigibson.examples.robots.robot_control_example --quickstart # (1)!
```

1. This demo lets you choose a scene, robot, and set of controllers, and then teleoperate the robot using your keyboard.
    The `--quickstart` flag will automatically select the scene and robot for you - remove that if you want to change
    the scene or robot.


You can also try exploring some of our new scenes interactively:

```{.shell .annotate}
python -m omnigibson.examples.scenes.scene_selector # (1)!
```

1. This demo lets you choose a scene and interactively move around using your keyboard and mouse. Hold down **`Shift`** and then **`Left-click + Drag`** an object to apply forces!

***

**Next:** Get quickly familiarized with **`BEHAVIOR`** from our [Quickstart Guide](./quickstart.md)!


## :material-fire-extinguisher: **Troubleshooting**

??? question "CuRobo installation failed"

    This is likely due to incorrect configurations of cuda toolkit. If you don't plan to use system-installed cuda toolkit, here is an alternate way to get it work all in conda:

    1. Install behavior without primitives support first: `./setup.sh --new-env --bddl --omnigibson [OTHER FLAGS]`
    2. Activate conda environment: `conda activate behavior`
    3. Install cuda (we use 12.4 here as an example): `conda install "cuda-compiler=12.4" "cuda-toolkit=12.4" -c conda-forge`
    4. Set environment variables:
        ```
        export CUDA_HOME="$CONDA_PREFIX"
        export CUDACXX="$CONDA_PREFIX/bin/nvcc"
        export CUDA_PATH="$CUDA_HOME"

        export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH"
        export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$LIBRARY_PATH"
        export LD_LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
        ```

    5. Install primitive support: `./setup.sh --bddl --omnigibson --primitives`

    Note that with this method you need to set the environment variables (step 4) the first time curobo performs JIT compilation. 

??? question "OmniGibson is stuck at `HydraEngine rtx failed creating scene renderer.`"

    `OmniGibson` is likely using an unsupported GPU (default is id 0). Run `nvidia-smi` to see the active list of GPUs, and select an NVIDIA-supported GPU and set its corresponding ID when running `OmniGibson` with `export OMNIGIBSON_GPU_ID=<ID NUMBER>`.
