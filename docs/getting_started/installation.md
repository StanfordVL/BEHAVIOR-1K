---
icon: material/hammer-wrench
---

# 🛠️ **Installation**

## 🗒️ **Requirements**

Please make sure your system meets the following specs:

- [x] **OS:** Ubuntu 18.04+
- [x] **RAM:** 32GB+
- [x] **GPU:** NVIDIA RTX 2070+
- [x] **VRAM:** 8GB+

??? question "Why these specs?"
    
    **`OmniGibson`** is built upon NVIDIA's [Omniverse](https://www.nvidia.com/en-us/omniverse/) and [Isaac Sim](https://developer.nvidia.com/isaac-sim) platforms, so we inherit their dependencies. For more information, please see [Isaac Sim's Requirements](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/requirements.html).

## 💻 **Setup**

You can quickly get **`OmniGibson`** immediately up and running from our pre-built 🐳 docker image:

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
curl -LJO https://raw.githubusercontent.com/StanfordVL/OmniGibson/main/docker/run_docker_gui.sh
chmod a+x run_docker.sh
chmod a+x run_docker_gui.sh
```

??? question annotate "What is being installed?"

    Our docker image automatically ships with a pre-configured conda virtual environment named `omnigibson` with Isaac Sim and **`OmniGibson`** pre-installed. Upon running the first time, our scene and object assets will automatically be downloaded as well. (1)

1.  📊 **Worried about dataset size?** We will ask whether you want to install our small demo dataset or full dataset of assets!


Then, simply launch the desired script:

=== "Headless"

    ```{.shell .annotate}
    sudo ./run_docker.sh <DATA_PATH> # (1)!
    ```

    1.  `<DATA_PATH>` specifies where data will be stored on your machine. (1) This needs to be called each time the docker container is run!
        {.annotate}

        1. If no `<DATA_PATH>` is specified, it defaults to `~/omnigibson-data` 

=== "GUI"

    ```{.shell .annotate}
    sudo ./run_docker_gui.sh <DATA_PATH> # (1)!
    ```

    1.  `<DATA_PATH>` specifies where data will be stored on your machine. (1) This needs to be called each time the docker container is run!
        {.annotate}

        1. If no `<DATA_PATH>` is specified, it defaults to `~/omnigibson-data` 

??? example annotate "Advanced: Installing from Source"

    This method is recommended for deeper users looking to develop upon **`OmniGibson`** or use it extensively for research. 

    
    1. Install [Conda](https://www.google.com/search?q=install+conda&rlz=1C5GCEA_enUS978US978&oq=install+conda&aqs=chrome..69i57l2j69i59l2j0i271j69i60l3.922j0j7&sourceid=chrome&ie=UTF-8)

    1. Install NVIDIA's [Isaac Sim platform](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_basic.html) (1)

    2. Export IsaacSim directory path as an environment variable: (2)

        ```shell
        export ISAAC_SIM_PATH = <YOUR_PATH_TO_ISAAC_SIM>
        ```

    3. Clone [**`OmniGibson`**](https://github.com/StanfordVL/OmniGibson) and move into the directory:

        ```shell
        git clone https://github.com/StanfordVL/OmniGibson.git
        cd OmniGibson
        ```

    4. Run the command to setup a virtual conda environment to run **`OmniGibson`**:

        ```shell
        chmod +x setup_conda_env.sh
        ./setup_conda_env.sh
        ```

    5. This will automatically create an dump you into a conda env called `omnigibson`. If you need to activate this environment later, simply call:

        ```shell
        conda activate omnigibson
        ```

    6. 🎉 Congrats! You installed **`OmniGibson`** successfully.  

1. Be sure keep track of where you choose Omniverse to write package files! By default this should be `~/.local/share/ov/pkg`

2. If you installed Isaac Sim to the default location, this is `~/.local/share/ov/pkg/isaac_sim-2022.1.1`


## 🌎 **Explore `OmniGibson`!**

??? warning annotate "Expect slowdown during first execution"

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
