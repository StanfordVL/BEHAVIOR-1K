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

There are two ways to setup **`OmniGibson`**:

- **Install with Docker (Linux Only)**: You can quickly get **`OmniGibson`** immediately up and running from our pre-built 🐳 docker image.
- **Install from source**: This method is recommended for deeper users looking to develop upon **`OmniGibson`** or use it extensively for research. 

=== "Docker (Linux only)"

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

    Install our docker launching scripts:
    ```shell
    curl -LJO https://raw.githubusercontent.com/StanfordVL/OmniGibson/main/docker/run_docker.sh
    chmod a+x run_docker.sh
    ```

    ??? question annotate "What is being installed?"

        Our docker image automatically ships with a pre-configured conda virtual environment named `omnigibson` with Isaac Sim and **`OmniGibson`** pre-installed. Upon running the first time, our scene and object assets will automatically be downloaded as well. (1)

    1.  📊 **Worried about dataset size?** We will ask whether you want to install our small demo dataset or full dataset of assets!

    Then, simply launch the shell script:

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

    The `-h --headless` argument is used to switch between gui and headless mode.
    
    ??? warning annotate "Are you using NFS or AFS?"

        Docker containers are unable to access NFS or AFS drives, so if `run_docker.sh` are located on an NFS / AFS partition, please set `<DATA_PATH>` to an alternative data directory located on a non-NFS / AFS partition.

=== "Source"

    Install **`OmniGibson`** from source is supported for both Linux and Windows.
    
    === "Linux (bash)"
    
        <div class="annotate" markdown>

        1. Install [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

        2. Install NVIDIA's [Isaac Sim platform](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_workstation.html) (1)

            !!! warning "The latest version of Isaac Sim (2022.2.1) has known issues when loading large **`OmniGibson`** scenes. Please install 2022.2.0 instead."

            !!! note "Depending on the OS, you might need to [install FUSE](https://github.com/AppImage/AppImageKit/wiki/FUSE) to run the Omniverse Launcher AppImage."

        3. Export IsaacSim directory path as an environment variable: (2)

            ```shell
            export ISAAC_SIM_PATH=<YOUR_PATH_TO_ISAAC_SIM>
            ```

        4. Clone [**`OmniGibson`**](https://github.com/StanfordVL/OmniGibson) and move into the directory:

            ```shell
            git clone https://github.com/StanfordVL/OmniGibson.git
            cd OmniGibson
            ```

        5. Setup a virtual conda environment to run **`OmniGibson`**:

            ```shell
            source setup_conda_env.sh
            ```

            This will automatically create an dump you into a conda env called `omnigibson`. If you need to activate this environment later, simply call:

            ```shell
            conda activate omnigibson
            ```

            ??? info "Note for zsh users"
                bash is **strongly** recommended on Linux. If you are using zsh, you need to change `${BASH_SOURCE[0]}` and `${BASH_SOURCE}` to `$0` in the first line of `<ISAAC_SIM_PATH>/setup_conda_env.sh` and `<ISAAC_SIM_PATH>/setup_python_env.sh` respectively in order for **`OmniGibson`** to work properly.

        7. Download **`OmniGibson`** assets and datasets:

            ```shell
            OMNIGIBSON_NO_OMNIVERSE=1 python omnigibson/scripts/setup.py
            ```

        8. 🎉 Congrats! You installed **`OmniGibson`** successfully.  

        </div>

        1. Be sure keep track of where you choose Omniverse to write package files! By default this should be `~/.local/share/ov/pkg`

        2. If you installed Isaac Sim to the default location, this is `~/.local/share/ov/pkg/isaac_sim-2022.2.0`

    === "Windows (cmd)"

        <div class="annotate" markdown>

        1. Install [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

        2. Install NVIDIA's [Isaac Sim platform](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_workstation.html) (1)

            !!! warning "The latest version of Isaac Sim (2022.2.1) has known issues when loading large **`OmniGibson`** scenes. Please install 2022.2.0 instead."

        3. Export IsaacSim directory path as an environment variable: (2)

            ```shell
            set ISAAC_SIM_PATH=<YOUR_PATH_TO_ISAAC_SIM>
            ```

        4. Clone [**`OmniGibson`**](https://github.com/StanfordVL/OmniGibson) and move into the directory:

            ```shell
            git clone https://github.com/StanfordVL/OmniGibson.git
            cd OmniGibson
            ```

        5. Setup a virtual conda environment to run **`OmniGibson`**:

            ```shell
            setup_conda_env.bat
            ```
            
            This will automatically create an dump you into a conda env called `omnigibson`. If you need to activate this environment later, simply call:

            ```shell
            conda activate omnigibson
            ```

        6. Download **`OmniGibson`** assets and datasets:

            ```shell
            set OMNIGIBSON_NO_OMNIVERSE=1&&python omnigibson/scripts/setup.py&&set OMNIGIBSON_NO_OMNIVERSE=
            ```

        7. 🎉 Congrats! You installed **`OmniGibson`** successfully.

        </div>
        
        1. Be sure keep track of where you choose Omniverse to write package files! By default this should be `C:\Users\<USER_NAME>\AppData\Local\ov\pkg`

        2. If you installed Isaac Sim to the default location, this is `C:\Users\<USER_NAME>\AppData\Local\ov\pkg\isaac_sim-2022.2.0`

    

## 🌎 **Explore `OmniGibson`!**

!!! warning annotate "Expect slowdown during first execution"

    Omniverse requires some one-time startup setup (up to ~5 minutes) when **`OmniGibson`** is imported for the first time. This is expected behavior, and should only occur once!

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
