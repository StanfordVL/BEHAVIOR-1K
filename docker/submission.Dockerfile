FROM continuumio/miniconda3

# Set working directory
WORKDIR /workspace

# Install system dependencies including build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    libudev-dev \
    libinput-dev \
    linux-libc-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a new conda environment
RUN conda create -n behavior python=3.10 -y -c conda-forge
SHELL ["conda", "run", "-n", "behavior", "/bin/bash", "-c"]

# Install additional packages
RUN pip install "numpy<2" "setuptools<=79"
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Copy over omnigibson source
ADD . /b1k-src
WORKDIR /b1k-src

# Install bddl (editable)
RUN pip install -e bddl
# Install omnigibson (editable)
RUN pip install -e OmniGibson[eval]

ENV PATH=/opt/conda/envs/behavior/bin:$PATH
ENV CONDA_DEFAULT_ENV=behavior

CMD ["python", "-u", "-c", "from omnigibson.learning.utils.network_utils import WebsocketPolicyServer; from omnigibson.learning.policies import LocalPolicy; server = WebsocketPolicyServer(LocalPolicy(action_dim=23)); server.serve_forever()"]
