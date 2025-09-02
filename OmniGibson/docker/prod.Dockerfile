FROM nvcr.io/nvidia/isaac-sim:4.5.0

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
ENV OMNIGIBSON_DATA_PATH /data

# Install Mamba (light conda alternative)
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C / bin/micromamba
ENV MAMBA_ROOT_PREFIX /micromamba
RUN micromamba create -n omnigibson -c conda-forge python=3.10
RUN micromamba shell init --shell=bash

# Install torch
RUN micromamba run -n omnigibson micromamba install \
  pytorch torchvision pytorch-cuda=11.8 \
  -c pytorch -c nvidia -c conda-forge

# Install curobo. This can normally be installed when OmniGibson is pip
# installed, but we need to install it beforehand here so that it doesn't
# have to happen on every time a CI action is run (otherwise it's just
# very slow).
# This also allows us to uninstall the cuda toolkit after curobo is built
# to save space (meaning curobo will not be able to be rebuilt at runtime).
# Here we also compile this such that it is compatible with GPU architectures
# Turing, Ampere, and Ada; which correspond to 20, 30, and 40 series GPUs.
# We also suppress the output of the installation to avoid the log limit.
RUN wget --no-verbose -O /cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \ 
  dpkg -i /cuda-keyring.deb && rm /cuda-keyring.deb && apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-11-8 && \
  TORCH_CUDA_ARCH_LIST='7.5;8.0;8.6+PTX' PATH=/usr/local/cuda-11.8/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH \
    micromamba run -n omnigibson pip install \
    git+https://github.com/StanfordVL/curobo@cbaf7d32436160956dad190a9465360fad6aba73#egg=nvidia_curobo \
    --no-build-isolation > /dev/null && \
  apt-get remove -y cuda-toolkit-11-8 && apt-get autoremove -y && apt-get autoclean -y && rm -rf /var/lib/apt/lists/*

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