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

- **🐍 Pip Install (Linux / Windows, Recommended)**: You can clone **`Omnigibson`** and automatically install Isaac Sim through pip for the fastest startup.
- **🐳 Install with Docker (Linux only)**: You can quickly get **`OmniGibson`** immediately up and running from our pre-built docker image that includes Isaac Sim.
- **🧪 Install with Omniverse Launcher (Linux / Windows)**: You can install Isaac Sim via the Omniverse launcher and hook **`OmniGibson`** up to it.

!!! tip ""
    === "🐍 Pip Install (Linux / Windows)"

        <div class="annotate" markdown>

        1. Create a conda environment with Python version **`3.10`**:

            ```shell
            conda create -n omnigibson python=3.10
            conda activate omnigibson
            ```

        2. Install OmniGibson with the optional Isaac Sim dependency:

            ```shell
            git clone https://github.com/StanfordVL/OmniGibson.git
            cd OmniGibson
            pip install --no-cache-dir -e .[isaac]
            ```

            If this step fails, we recommend trying the [source installation](#-install-from-source-linux--windows) method.

        4. Run Isaac Sim to accept the EULA:

            ```shell
            isaacsim
            ```

            !!! important "EULA Acceptance"
                It is necessary to accept the Omniverse License Agreement (EULA) in order to use Isaac Sim.
                The first time `isaacsim` is imported, you will be prompted to accept the EULA:

                ```
                By installing or using Omniverse Kit, I agree to the terms of NVIDIA OMNIVERSE LICENSE AGREEMENT (EULA)
                in https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html
                Do you accept the EULA? (Yes/No)
                ```

                You must respond with 'Yes' to proceed. Once the EULA is accepted, it should not appear on subsequent Isaac Sim calls. If the EULA is not accepted, the execution will be terminated.

                **You might get some error dialogs that are safe to ignore. After accepting the EULA, you can close Isaac Sim.**

        5. Download **`OmniGibson`** dataset and assets:

            ```shell
            python scripts/download_datasets.py
            ```

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

    === "🧪 Install from source (Linux / Windows)"

        Install **`OmniGibson`** from source is supported for both **🐧 Linux (bash)** and **📁 Windows (powershell/cmd)**.
        !!! example ""
            === "🐧 Linux (bash)"
            
                <div class="annotate" markdown>

                1. Install [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and NVIDIA's [Omniverse Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_workstation.html) 

                    !!! warning "Please make sure you have the latest version of Isaac Sim (2023.1.1) installed."

                    For Ubuntu 22.04, you need to [install FUSE](https://github.com/AppImage/AppImageKit/wiki/FUSE) to run the Omniverse Launcher AppImage.

                2. Clone [**`OmniGibson`**](https://github.com/StanfordVL/OmniGibson) and move into the directory:

                    ```shell
                    git clone https://github.com/StanfordVL/OmniGibson.git
                    cd OmniGibson
                    ```
                
                    ??? note "Nightly build"

                        The main branch contains the stable version of **`OmniGibson`**. For our latest developed (yet not fully tested) features and bug fixes, please clone from the `og-develop` branch.

                3. Setup a virtual conda environment to run **`OmniGibson`**:

                    ```{.shell .annotate}
                    ./scripts/setup.sh # (1)!
                    ```

                    1. The script will ask you which Isaac Sim to use. If you installed it in the default location, it should be `~/.local/share/ov/pkg/isaac_sim-2023.1.1`

                    This will create a conda env with `omnigibson` installed. Simply call `conda activate` to activate it.

                4. Download **`OmniGibson`** dataset (within the conda env):

                    ```shell
                    python scripts/download_datasets.py
                    ```

                </div>

                

            === "📁 Windows (powershell/cmd)"

                <div class="annotate" markdown>

                1. Install [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and NVIDIA's [Omniverse Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_workstation.html)

                    !!! warning "Please make sure you have the latest version of Isaac Sim (2023.1.1) installed."

                2. Clone [**`OmniGibson`**](https://github.com/StanfordVL/OmniGibson) and move into the directory:

                    ```shell
                    git clone https://github.com/StanfordVL/OmniGibson.git
                    cd OmniGibson
                    ```

                    ??? note "Nightly build"

                        The main branch contains the stable version of **`OmniGibson`**. For our latest developed (yet not fully tested) features and bug fixes, please clone from the `og-develop` branch.

                3. Setup a virtual conda environment to run **`OmniGibson`**:

                    ```{.powershell .annotate}
                    .\scripts\setup.bat # (1)!
                    ```

                    1. The script will ask you which Isaac Sim to use. If you installed it in the default location, it should be `C:\Users\<USER_NAME>\AppData\Local\ov\pkg\isaac_sim-2023.1.1`

                    This will create a conda env with `omnigibson` installed. Simply call `conda activate` to activate it.

                4. Download **`OmniGibson`** dataset (within the conda env):

                    ```powershell
                    python scripts\download_datasets.py
                    ```

                </div>


## 🌎 **Explore `OmniGibson`!**

!!! warning annotate "Expect slowdown during first execution"

    Omniverse requires some one-time startup setup when **`OmniGibson`** is imported for the first time.
    
    The process could take up to 5 minutes. This is expected behavior, and should only occur once!

**`OmniGibson`** is now successfully installed! Try exploring some of our new scenes interactively:

```{.shell .annotate}
python -m omnigibson.examples.scenes.scene_selector # (1)!
```

1. This demo lets you choose a scene and interactively move around using your keyboard and mouse. Hold down **`Shift`** and then **`Left-click + Drag`** an object to apply forces!

You can also try teleoperating one of our robots:

```{.shell .annotate}
python -m omnigibson.examples.robots.robot_control_example # (1)!
```

1. This demo lets you choose a scene, robot, and set of controllers, and then teleoperate the robot using your keyboard.

***

**Next:** Get quickly familiarized with **`OmniGibson`** from our [Quickstart Guide](./quickstart.md)!


## 🧯 **Troubleshooting**

??? question "I cannot open Omniverse Launcher AppImage on Linux"

    You probably need to [install FUSE](https://github.com/AppImage/AppImageKit/wiki/FUSE) to run the Omniverse Launcher AppImage.

??? question "OmniGibson is stuck at `HydraEngine rtx failed creating scene renderer.`"

    `OmniGibson` is likely using an unsupported GPU (default is id 0). Run `nvidia-smi` to see the active list of GPUs, and select an NVIDIA-supported GPU and set its corresponding ID when running `OmniGibson` with `export OMNIGIBSON_GPU_ID=<ID NUMBER>`.