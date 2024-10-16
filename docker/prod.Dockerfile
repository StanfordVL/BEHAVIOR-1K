FROM nvcr.io/nvidia/isaac-sim:4.1.0

# Set up all the prerequisites.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  curl git wget \
  g++ cmake pkg-config libeigen3-dev wget libyaml-cpp-dev castxml pypy3 \
  && rm -rf /var/lib/apt/lists/*

RUN rm -rf /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/gym*
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

# Make sure isaac gets properly sourced every time omnigibson gets called
ARG CONDA_ACT_FILE="/micromamba/envs/omnigibson/etc/conda/activate.d/env_vars.sh"
RUN mkdir -p "/micromamba/envs/omnigibson/etc/conda/activate.d"
RUN touch $CONDA_ACT_FILE

RUN echo '#!/bin/sh' > $CONDA_ACT_FILE
RUN echo "source /isaac-sim/setup_conda_env.sh" >> $CONDA_ACT_FILE

RUN echo "micromamba activate omnigibson" >> /root/.bashrc

# Prepare to build OMPL
ENV CXX="g++"
ENV MAKEFLAGS="-j `nproc`"
RUN micromamba run -n omnigibson micromamba install -c conda-forge boost && \
    micromamba run -n omnigibson pip install pyplusplus && \
    git clone https://github.com/ompl/ompl.git /ompl && \
    mkdir -p /ompl/build/Release && \
    sed -i "s/find_program(PYPY/# find_program(PYPY/g" /ompl/CMakeModules/Findpypy.cmake

# Build and install OMPL 
RUN micromamba run -n omnigibson /bin/bash --login -c 'source /isaac-sim/setup_conda_env.sh && (which python > /root/PYTHON_EXEC) && (echo $PYTHONPATH > /root/PYTHONPATH)' && \
    cd /ompl/build/Release && \
    micromamba run -n omnigibson cmake ../.. \
      -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
      -DBOOST_ROOT="$CONDA_PREFIX" \
      -DPYTHON_EXEC=$(cat /root/PYTHON_EXEC) \
      -DPYTHONPATH=$(cat /root/PYTHONPATH) && \
    micromamba run -n omnigibson make -j 4 update_bindings && \
    micromamba run -n omnigibson make -j 4 && \
    cd py-bindings && \
    micromamba run -n omnigibson make install

# Test OMPL
RUN micromamba run -n omnigibson python -c "from ompl import base"

# Add setup to be executed on bash launch
RUN 

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
      micromamba run -n omnigibson pip install -e .[dev]; \
    else \
      WORKDIR_PATH=/; \
      cd / && rm -rf /omnigibson-src; \
    fi

# Reset the WORKDIR based on whether or not we are in dev mode
WORKDIR ${WORKDIR_PATH}

ENTRYPOINT ["micromamba", "run", "-n", "omnigibson"]
CMD ["/bin/bash"]