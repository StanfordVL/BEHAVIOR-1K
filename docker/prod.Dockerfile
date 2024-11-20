FROM nvcr.io/nvidia/isaac-sim:4.1.0

# Set up all the prerequisites.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  curl git wget \
  g++ cmake pkg-config libeigen3-dev wget libyaml-cpp-dev castxml pypy3 \
  && rm -rf /var/lib/apt/lists/*

RUN rm -rf /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/gym*
RUN rm -rf /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch*
RUN rm -rf /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/functorch*
RUN rm -rf /isaac-sim/kit/extscore/omni.kit.pip_archive/pip_prebundle/numpy*
RUN /isaac-sim/python.sh -m pip install click~=8.1.3

# Mount the data directory
VOLUME ["/data"]
ENV OMNIGIBSON_DATASET_PATH /data/og_dataset
ENV OMNIGIBSON_ASSET_PATH /data/assets
ENV GIBSON_DATASET_PATH /data/g_dataset
ENV OMNIGIBSON_KEY_PATH /data/omnigibson.key

# Install Mamba (light conda alternative)
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C / bin/micromamba
ENV MAMBA_ROOT_PREFIX /micromamba
RUN micromamba create -n omnigibson -c conda-forge python=3.10
RUN micromamba shell init --shell=bash

# Install evdev, which is a dependency of telemoma. It cannot be
# installed afterwards because it depends on some C compilation that
# fails if kernel headers ("sysroot") is present inside the conda env.
# The CUDA installation will add the kernel headers to the conda env.
# So we install evdev before installing CUDA.
RUN micromamba run -n omnigibson pip install evdev

# Install CUDA and torch
RUN micromamba run -n omnigibson micromamba install \
  pytorch torchvision pytorch-cuda=11.8 cuda=11.8.0 \
  -c pytorch -c nvidia -c conda-forge

# Install curobo. This can normally be installed when OmniGibson is pip
# installed, but we need to install it beforehand here so that it doesn't
# have to happen on every time a CI action is run (otherwise it's just
# very slow)
# Here we also compile this such that it is compatible with GPU architectures
# Turing, Ampere, and Ada; which correspond to 20, 30, and 40 series GPUs.
# TORCH_CUDA_ARCH_LIST='7.5;8.0;8.6;8.7;8.9;7.5+PTX;8.0+PTX;8.6+PTX;8.7+PTX;8.9+PTX'
RUN TORCH_CUDA_ARCH_LIST='7.5,8.0;8.6;8.7;8.9+PTX' \
  micromamba run -n omnigibson pip install git+https://github.com/StanfordVL/curobo@06d8c79b660db60c2881e9319e60899cbde5c5b5#egg=nvidia_curobo --no-build-isolation

# Make sure isaac gets properly sourced every time omnigibson gets called
ARG CONDA_ACT_FILE="/micromamba/envs/omnigibson/etc/conda/activate.d/env_vars.sh"
RUN mkdir -p "/micromamba/envs/omnigibson/etc/conda/activate.d"
RUN touch $CONDA_ACT_FILE

RUN echo '#!/bin/sh' > $CONDA_ACT_FILE
RUN echo "source /isaac-sim/setup_conda_env.sh" >> $CONDA_ACT_FILE

RUN echo "micromamba activate omnigibson" >> /root/.bashrc

# Copy over omnigibson source
ADD . /omnigibson-src
WORKDIR /omnigibson-src

# Set the shell
SHELL ["micromamba", "run", "-n", "omnigibson", "/bin/bash", "--login", "-c"]

# Optionally install OmniGibson (e.g. unless the DEV_MODE flag is set) or
# remove the OmniGibson source code if we are in dev mode and change the workdir
ARG DEV_MODE
ENV DEV_MODE=${DEV_MODE}
ARG WORKDIR_PATH=/omnigibson-src
RUN if [ "$DEV_MODE" != "1" ]; then \
      echo "OMNIGIBSON_NO_OMNIVERSE=1 python omnigibson/download_datasets.py" >> /root/.bashrc; \
      micromamba run -n omnigibson pip install -e .[dev,primitives]; \
    else \
      WORKDIR_PATH=/; \
      cd / && rm -rf /omnigibson-src; \
    fi

# Reset the WORKDIR based on whether or not we are in dev mode
WORKDIR ${WORKDIR_PATH}

ENTRYPOINT ["micromamba", "run", "-n", "omnigibson"]
CMD ["/bin/bash"]