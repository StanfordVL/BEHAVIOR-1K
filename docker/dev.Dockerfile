FROM nvcr.io/nvidia/isaac-sim:2022.2.0

# Set up all the prerequisites.
RUN apt-get update && apt-get install -y \
  curl \
  git \
  wget \
  && rm -rf /var/lib/apt/lists/*

RUN rm -rf /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/gym
RUN rm -rf /isaac-sim/kit/extscore/omni.kit.pip_archive/pip_prebundle/numpy

# Mount the data directory
VOLUME ["/data"]
ENV OMNIGIBSON_DATASET_PATH /data/og_dataset
ENV OMNIGIBSON_ASSETS_PATH /data/assets
ENV GIBSON_DATASET_PATH /data/g_dataset
ENV OMNIGIBSON_KEY_PATH /data/omnigibson.key

# Install Mamba (light conda alternative)
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C / bin/micromamba
ENV MAMBA_ROOT_PREFIX /micromamba
RUN micromamba create -n omnigibson -c conda-forge python=3.7
RUN micromamba shell init --shell=bash --prefix=/micromamba
RUN echo "micromamba activate omnigibson" >> /root/.bashrc
RUN echo "source /isaac-sim/setup_conda_env.sh" >> /root/.bashrc

# Install some additional niceties for working with notebooks
SHELL ["micromamba", "run", "-n", "omnigibson", "/bin/bash", "--login", "-c"]
RUN micromamba install -c conda-forge opencv

ENTRYPOINT []

CMD ["/bin/bash"]
