FROM nvcr.io/nvidia/isaac-sim:2022.1.1

RUN echo "bash /isaac-sim/vulkan_check.sh" >> /root/.bashrc
RUN rm -rf /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/gym

RUN apt-get update && apt-get install -y \
  curl \
  git \
  && rm -rf /var/lib/apt/lists/*

RUN  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C / bin/micromamba

ENV MAMBA_ROOT_PREFIX /micromamba
RUN micromamba create -n omnigibson -c conda-forge python=3.7
RUN micromamba shell init --shell=bash --prefix=/micromamba
ENV PATH /micromamba/envs/omnigibson/bin:$PATH
RUN echo "source /isaac-sim/setup_conda_env.sh" >> /root/.bashrc

WORKDIR /
ENV GITHUB_USER mjlbach
ENV GITHUB_TOKEN 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
ENV GITHUB_REPOSITORY stanfordvl/OmniGibson3.git
RUN git clone https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}

ENV GITHUB_REPOSITORY stanfordvl/bddl-dev.git
ENV GITHUB_TOKEN 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
RUN git clone https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}

WORKDIR /OmniGibson3

RUN pip install -e .
RUN micromamba install -n omnigibson -c conda-forge opencv

WORKDIR /bddl-dev
RUN pip install -e .

WORKDIR /OmniGibson3/omnigibson/examples/

RUN sed -i "s/gm.HEADLESS = False/gm.HEADLESS = True/" /OmniGibson3/omnigibson/macros.py

ENTRYPOINT bash --login
