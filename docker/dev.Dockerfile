FROM nvcr.io/nvidia/isaac-sim:2022.2.0

# Set up all the prerequisites.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  curl git wget \
  g++ cmake pkg-config libeigen3-dev wget libyaml-cpp-dev castxml pypy3 \
  && rm -rf /var/lib/apt/lists/*

RUN rm -rf /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/gym*
RUN rm -rf /isaac-sim/kit/extscore/omni.kit.pip_archive/pip_prebundle/numpy*

# Mount the data directory
VOLUME ["/data"]
ENV OMNIGIBSON_DATASET_PATH /data/og_dataset
ENV OMNIGIBSON_ASSET_PATH /data/assets
ENV GIBSON_DATASET_PATH /data/g_dataset
ENV OMNIGIBSON_KEY_PATH /data/omnigibson.key

# Install Mamba (light conda alternative)
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C / bin/micromamba
ENV MAMBA_ROOT_PREFIX /micromamba
RUN micromamba create -n omnigibson -c conda-forge python=3.7
RUN micromamba shell init --shell=bash --prefix=/micromamba

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
    mkdir -p /ompl/build/Release

# Build and install OMPL
RUN cd /ompl/build/Release && \
    micromamba run -n omnigibson cmake ../.. \
      -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
      -DBOOST_ROOT="$CONDA_PREFIX" \
      -DPYTHON_EXEC=/micromamba/envs/omnigibson/bin/python3.7 \
      -DPYTHONPATH=/micromamba/envs/omnigibson/lib/python3.7/site-packages && \
    micromamba run -n omnigibson make -j 4 update_bindings && \
    micromamba run -n omnigibson make -j 4 && \
    cd py-bindings && \
    micromamba run -n omnigibson make install

# Test OMPL
RUN micromamba run -n omnigibson python -c "import ompl"

ENTRYPOINT ["micromamba", "run", "-n", "omnigibson"]

CMD ["/bin/bash"]
