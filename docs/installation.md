# Installation
There are two steps to install OmniGibson, the Interactive Gibson Environment, on your computer.

First, you need to install the simulation environment. This may require installing additional dependencies. Then, you need to download the assets: models of the robotic agents, the interactive objects and 3D reconstructed real-world large environments for your agents to train.

## Installing the Environment

### System Requirements

The minimum system requirements are the following:

- Linux
    - Ubuntu 16.04
    - Nvidia GPU with VRAM > 6.0GB
    - Nvidia driver >= 384
    - CUDA >= 9.0, CuDNN >= v7
    - CMake >= 2.8.12 (can install with `pip install cmake`)
    - g++ (GNU C++ compiler)
    - libegl-dev (Debian/Ubuntu: vendor neutral GL dispatch library -- EGL support)
- Windows
    - Windows 10
    - Nvidia GPU with VRAM > 6.0GB
    - Nvidia driver >= 384
    - CUDA >= 9.0, CuDNN >= v7
    - CMake >= 2.8.12 (can install with `pip install cmake`)
    - Microsoft Visual Studio 2017 with visual C++ tool and latest Windows 10 SDK
- Mac OS X
    - Tested on 10.15
    - PBR features not supported
    - CMake >= 2.8.12 (can install with `pip install cmake`)

Other system configurations may work, but we haven't tested them extensively and we probably won't be able to provide as much support as we want.

### Installing dependencies in Linux machines

As most Python packages, we recommend to install OmniGibson in a virtual environment. 
We suggest to use Conda instead of a standard virtual environment. 
To setup anaconda with the right dependencies, run the following as your user account (**not as root/superuser**):

```bash
# Install miniconda
curl -LO http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
bash Miniconda-latest-Linux-x86_64.sh
rm Miniconda-latest-Linux-x86_64.sh

# Add conda to your PATH
echo "export PATH=$HOME/.miniconda/bin:$PATH" >> .bashrc 

# Update conda and create a virtual environment for OmniGibson
conda update -y conda
conda create -y -n omnigibson python=3.8
conda activate omnigibson
```

<div class="admonition important">
<p class="admonition-title">Careful, this may change your GPU drivers!</p>
There are several system dependencies to correctly run OmniGibson on Linux, mostly related to Nvidia drivers and Cuda.
In case your system is a clean Ubuntu 20.04, you can run the following commands as root/superuser to install all required dependencies:

<details>
  <summary>Click to expand the code to install the dependencies including Nvidia drivers in headless mode to use for example in a cluster:</summary>
<p>

```bash
# Add the nvidia ubuntu repositories
apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

# The following cuda libraries are required to compile omnigibson
apt-get update && apt-get update && apt-get install -y --no-install-recommends \
    nvidia-headless-470 \
    cuda-cudart-11-1=11.1.74-1 \
    cuda-compat-11-1 \
    cuda-command-line-tools-11-1=11.1.1-1 \
    cuda-libraries-dev-11-1=11.1.1-1 \

# For building and running omnigibson
apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    git \
    g++ \
    libegl-dev
```
</p>
</details>
<details>
  <summary>Click to expand the code to install the dependencies including Nvidia drivers to render on screen for example on a desktop computer:</summary>
<p>

```bash
# Add the nvidia ubuntu repositories
apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

# The following cuda libraries are required to compile omnigibson
apt-get update && apt-get update && apt-get install -y --no-install-recommends \
    xserver-xorg-video-nvidia-470 \
    cuda-cudart-11-1=11.1.74-1 \
    cuda-compat-11-1 \
    cuda-command-line-tools-11-1=11.1.1-1 \
    cuda-libraries-dev-11-1=11.1.1-1 \

# For building and running omnigibson
apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    git \
    g++ \
    libegl-dev
```

</p>
</details>

</div>

By default, OmniGibson builds with CUDA support which requires that `nvcc` is on your path (or CUDA 11 is symlinked to `/usr/local/cuda` from `/usr/local/cuda-11.1`). Cmake uses `nvcc` to find the CUDA libraries and headers when building OmniGibson. Add the following to your shell rc (`.bashrc`, `.zshrc`, etc.) and re-login to your shell (`exec bash`, `exec zsh`, etc.):
```bash
export PATH=/usr/local/cuda-11.1/bin:$PATH
```

Then verify nvcc is on your PATH:
```bash
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Built on Mon_Oct_12_20:09:46_PDT_2020
Cuda compilation tools, release 11.1, V11.1.105
Build cuda_11.1.TC455_06.29190527_0
```

In case you want to build without CUDA support (used for the "rendering to GPU tensor" feature), you will have to set `USE_CUDA` to `False` in `OmniGibson/omnigibson/render/CMakeLists.txt`.

### Installing OmniGibson

We provide 3 methods to install the simulator.

#### 1. pip

OmniGibson's simulator can be installed as a python package using pip:

```bash
pip install omnigibson  # This step takes about 4 minutes
# run the demo
python -m omnigibson.examples.environments.env_nonint_example
```

#### 2. Docker image

Docker provides an easy way to reproduce the development environment across platforms without manually installing the software dependencies. We have prepared docker images that contain everything you need to get started with OmniGibson.  

First, install Docker from the [official website](https://www.docker.com/). Please make sure that the docker version is at least v19.0 to enable native GPU support.

Next, download our pre-built images with the script in the `OmniGibson` repo:

```
cd OmniGibson
./docker/pull-images.sh
```

Two images will be downloaded:
* `omnigibson/omnigibson:latest`: smaller image, but does not support GUI. 
* `omnigibson/omnigibson-gui:latest`: supports GUI and remote desktop access via VNC.

We also provide scripts to build the images from scratch:
```
# image without GUI:
cd OmniGibson/docker/base
./build.sh

# image with GUI and VNC:
cd OmniGibson/docker/headless-gui
./build.sh
```


#### 3. Compile from source

Alternatively, OmniGibson can be compiled from source: [OmniGibson GitHub Repo](https://github.com/StanfordVL/OmniGibson). First, you need to install anaconda following the guide on [their website](https://www.anaconda.com/). 

```bash
git clone https://github.com/StanfordVL/OmniGibson --recursive
cd OmniGibson

# if you didn't create the conda environment before:
conda create -y -n omnigibson python=3.8
conda activate omnigibson

pip install -e . # This step takes about 4 minutes
```

We recommend the third method if you plan to modify OmniGibson in your project. If you plan to use it as it is to train navigation and manipulation agents, the pip installation or docker image should meet your requirements.

Note: If you are not using conda, you will need the system packages python3-dev (header files to build Python extensions) and python3-opencv (provides opencv and its dependencies).

### The SVL pybullet fork

To optimize and accelerate physics simulation, we use a custom version of pybullet in OmniGibson. This is installed automatically if you install OmniGibson in a fresh conda environment, but if you already have a regular pybullet, you should manually remove it and install our fork as follows (otherwise your 'pip install -e .' will fail):

```bash
pip uninstall pybullet
pip install pybullet-svl
```

## Downloading the Assets and Datasets of Scenes and Objects

Once the environment has been installed, we need to download the assets to enable the simulation including models of the robotic agents, objects, 3D scenes, etc. This process requires three simple steps.
 
First, we need to configure where the OmniGibson's assets are going to be stored. The desired path should be indicated in `your_installation_path/omnigibson/global_config.yaml`. The default place to store the data is:
```bash
assets_path: your_installation_path/omnigibson/data/assets 
g_dataset_path: your_installation_path/omnigibson/data/g_dataset
og_dataset_path: your_installation_path/omnigibson/data/og_dataset
threedfront_dataset_path: your_installation_path/omnigibson/data/threedfront_dataset 
cubicasa_dataset_path: your_installation_path/omnigibson/data/assetscubicasa_dataset 
```

In case you prefer to store the assets in a different location, you can run the command:
```bash
python -m omnigibson.utils.assets_utils --change_data_path
```

Second, we need to download the robot models and some small objects from the assets bundle [here](https://storage.googleapis.com/gibson_scenes/assets_omnigibson.tar.gz) and unpack it in the assets folder. More easily, this process can be automatically done by executing the command:

```bash
python -m omnigibson.utils.assets_utils --download_assets
```

Third, we need to download datasets of scenes (Gibson or OmniGibson), and, possibly, the BEHAVIOR Datasets of Object Models. 
For interactive tasks, you need to download OmniGibson 2.0 Scenes and the BEHAVIOR Dataset of Objects, or OmniGibson 1.0 Scenes. They include several fully interactive scenes and hundreds of 3D objects to use with our simulator.
For navigation tasks, you could use the interactive scenes, but we also provide back-compatibility to the Gibson Dataset, Stanford 2D-3D-Semantics Dataset, and Matterport3D Dataset that include non-interactive scene models.
Follow the detailed instructions [here](dataset.md) to download the aforementioned datasets.

Alternatively, to avoid downloading an entire dataset before you can test OmniGibson's functionalities, we provide a single [high quality small non-interactive scene (R's)](https://storage.googleapis.com/gibson_scenes/Rs.tar.gz) for demo and testing purposes.
To download this demo data, run:

```bash
python -m omnigibson.utils.assets_utils --download_demo_data
```

## Examples

We provide multiple examples to get you started!
Check the folder [omnigibson/examples](https://github.com/StanfordVL/OmniGibson/tree/master/omnigibson/examples) and the description [here](examples.md).

After installing the code and downloading the demo data, you should be able to try out a simple robot navigation demo executing:

```bash
python -m omnigibson.examples.environments.env_nonint_example
```

## Testing

To test OmniGibson installation, you can run 
```bash
python
>> import omnigibson
```

For a full suite of tests and benchmarks, you can refer to [tests](tests.md) for more details. 
(For Mac users) Some tests will fail as they require a Nvidia GPU.

## Uninstalling
Uninstalling OmniGibson is easy: `pip uninstall omnigibson`
