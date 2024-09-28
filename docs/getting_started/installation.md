---
icon: material/hammer-wrench
---

# 🛠️ **Installation**

## 🗒️ **Requirements**

Please make sure your system meets the following specs:

- [x] **OS:** Ubuntu 20.04+ / Windows 10+
- [x] **RAM:** 32GB+
- [x] **GPU:** NVIDIA RTX 2070+
- [x] **VRAM:** 8GB+

??? question "Why these specs?"
    
    **`OmniGibson`** is built upon NVIDIA's [Omniverse](https://www.nvidia.com/en-us/omniverse/) and [Isaac Sim](https://developer.nvidia.com/isaac-sim) platforms, so we inherit their dependencies. For more information, please see [Isaac Sim's Requirements](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/requirements.html).

## 💻 **Setup**

There are three ways to setup **`OmniGibson`**, all built upon different ways of installing NVIDIA Isaac Sim:

- **🐍 Install with pip (Linux / Windows, Recommended)**: You can install **`Omnigibson`** and automatically install Isaac Sim through pip for the fastest startup.
- **🐳 Install with Docker (Linux only)**: You can quickly get **`OmniGibson`** immediately up and running from our pre-built docker image that includes Isaac Sim.
- **🧪 Install with Omniverse Launcher (Linux / Windows)**: You can install Isaac Sim via the Omniverse launcher and hook **`OmniGibson`** up to it.

!!! tip ""
    === "🐍 Install with pip (Linux / Windows)"

        <div class="annotate" markdown>

        1. Create a conda environment with Python version **`3.10`** and numpy and PyTorch:

            ```shell
            conda create -n omnigibson python=3.10 pytorch torchvision torchaudio pytorch-cuda=12.1 "numpy<2" -c pytorch -c nvidia
            conda activate omnigibson
            ```

        2. Install OmniGibson:

            <div class="grid" markdown>

            !!! note "Install from PyPI (source not editable)"

                ```shell
                pip install omnigibson
                ```

            !!! example "Install from GitHub (source editable)"

                ```shell
                git clone https://github.com/StanfordVL/OmniGibson.git
                cd OmniGibson
                pip install -e .
                ```

            </div>

            !!! note "Nightly build"

                The main branch contains the stable version of **`OmniGibson`**. For our latest developed (yet not fully tested) features and bug fixes, please clone from the `og-develop` branch.

        3. Run the installation script to  install Isaac Sim as well as **`OmniGibson`** dataset and assets:

            ```shell
            python -m omnigibson.install
            ```

            If this step fails, we recommend considering the [source installation](#-install-from-source-linux--windows) method.

        </div>

        !!! note "More information"
            For more details on installing Isaac Sim via pip, please refer to the [official Isaac Sim documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_python.html).

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

    === "🧪 Install with Omniverse Launcher (Linux / Windows)"

        Install **`OmniGibson`** with Omniverse Launcher is supported for both **🐧 Linux (bash)** and **📁 Windows (powershell/cmd)**.
            
        <div class="annotate" markdown>

        1. Install [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and NVIDIA's [Omniverse Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_workstation.html) 

            !!! warning "Please make sure you have the currently supported version of Isaac Sim (4.1.0) installed."

            For Ubuntu 22.04, you need to [install FUSE](https://github.com/AppImage/AppImageKit/wiki/FUSE) to run the Omniverse Launcher AppImage.

        2. Create a conda environment with Python version **`3.10`**:

            ```shell
            conda create -n omnigibson python=3.10 pytorch torchvision torchaudio pytorch-cuda=12.1 "numpy<2" -c pytorch -c nvidia
            conda activate omnigibson
            ```

        3. Install OmniGibson:

            <div class="grid" markdown>

            !!! note "Install from PyPI (source not editable)"

                ```shell
                pip install omnigibson
                ```

            !!! example "Install from GitHub (source editable)"

                ```shell
                git clone https://github.com/StanfordVL/OmniGibson.git
                cd OmniGibson
                pip install -e .
                ```

            </div>

            !!! note "Nightly build"

                The main branch contains the stable version of **`OmniGibson`**. For our latest developed (yet not fully tested) features and bug fixes, please clone from the `og-develop` branch.

        4. Run the installation script to hook the environment up to Isaac Sim as well as **`OmniGibson`** dataset and assets:

            ```shell
            python -m omnigibson.install --launcher-install
            ```

            !!! note "What does this do?"

                When you install OmniGibson this way, it will modify your conda environment setup to hook it up to the launcher-installed Isaac Sim.

        5. Deactivate and reactivate the conda environment:

            Because the environment was modified by the installer to hook it up to the launcher-installed Isaac Sim, you need to reactivate it.

            ```shell
            conda deactivate
            conda activate omnigibson
            ```

        </div>



## 🌎 **Explore `OmniGibson`!**

!!! warning annotate "Expect slowdown during first execution"

    Omniverse requires some one-time startup setup when **`OmniGibson`** is imported for the first time.
    
    The process could take up to 5 minutes. This is expected behavior, and should only occur once!

**`OmniGibson`** is now successfully installed! You can try teleoperating one of our robots:

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

**Next:** Get quickly familiarized with **`OmniGibson`** from our [Quickstart Guide](./quickstart.md)!


## 🧯 **Troubleshooting**

??? question "I cannot open Omniverse Launcher AppImage on Linux"

    You probably need to [install FUSE](https://github.com/AppImage/AppImageKit/wiki/FUSE) to run the Omniverse Launcher AppImage.

??? question "OmniGibson is stuck at `HydraEngine rtx failed creating scene renderer.`"

    `OmniGibson` is likely using an unsupported GPU (default is id 0). Run `nvidia-smi` to see the active list of GPUs, and select an NVIDIA-supported GPU and set its corresponding ID when running `OmniGibson` with `export OMNIGIBSON_GPU_ID=<ID NUMBER>`.