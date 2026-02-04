# --- START ISAAC SIM DOCKERFILE COPY ---
FROM nvcr.io/nvidia/base/ubuntu:jammy-20251013

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all
# ENV OMNI_SERVER=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1
ENV MIN_DRIVER_VERSION=570.169

# Disable hardware-specific library optimizations in dynamic linker for wider compatibility (legacy Debian/Ubuntu)
RUN touch /etc/ld.so.nohwcap

# Install dependencies
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl \
        libatomic1 \
        libegl1 \
        libgl1 \
        libglu1-mesa \
        libglx0 \
        libgomp1 \
        libsm6 \
        libxi6 \
        libxrandr2 \
        libxt6 \
        unzip \
        ca-certificates \
        libglib2.0-0 \
        libnghttp2-14 \
        # --- NOTE THAT THIS IS NOT PART OF THE COPY ---
        # From this point on these are BEHAVIOR dependencies.
        git \
        g++ \
        cmake \
        pkg-config \
        libeigen3-dev \
        wget \
        libyaml-cpp-dev \
        castxml \
    && apt-get -y autoremove \
    && apt-get clean autoclean \
    && rm -rf /var/lib/apt/lists/*

# Install runtime configuration files
RUN mkdir -p /usr/share/glvnd/egl_vendor.d /etc/vulkan/icd.d /etc/vulkan/implicit_layer.d
RUN printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_nvidia.so.0"\n    }\n}\n' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
RUN printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_mesa.so.0"\n    }\n}\n' > /usr/share/glvnd/egl_vendor.d/50_mesa.json
RUN printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libGLX_nvidia.so.0",\n        "api_version" : "1.3.194"\n    }\n}\n' > /etc/vulkan/icd.d/nvidia_icd.json
RUN mkdir -p /etc/vulkan/implicit_layer.d && printf '{\n    "file_format_version" : "1.0.0",\n    "layer": {\n        "name": "VK_LAYER_NV_optimus",\n        "type": "INSTANCE",\n        "library_path": "libGLX_nvidia.so.0",\n        "api_version" : "1.3.194",\n        "implementation_version" : "1",\n        "description" : "NVIDIA Optimus layer",\n        "functions": {\n            "vkGetInstanceProcAddr": "vk_optimusGetInstanceProcAddr",\n            "vkGetDeviceProcAddr": "vk_optimusGetDeviceProcAddr"\n        },\n        "enable_environment": {\n            "__NV_PRIME_RENDER_OFFLOAD": "1"\n        },\n        "disable_environment": {\n            "DISABLE_LAYER_NV_OPTIMUS_1": ""\n        }\n    }\n}\n' > /etc/vulkan/implicit_layer.d/nvidia_layers.json
ENV VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json

# Open ports for live streaming
EXPOSE 47998/udp \
       49100/tcp

# --- END ISAAC SIM DOCKERFILE COPY ---

# --- COPIED FROM CONDA DOCKERFILE ---
ENV PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ARG INSTALLER_URL_LINUX64=https://repo.anaconda.com/miniconda/Miniconda3-py313_25.3.1-1-Linux-x86_64.sh
ARG SHA256SUM_LINUX64=53a86109463cfd70ba7acab396d416e623012914eee004729e1ecd6fe94e8c69
ARG INSTALLER_URL_AARCH64=https://repo.anaconda.com/miniconda/Miniconda3-py313_25.3.1-1-Linux-aarch64.sh
ARG SHA256SUM_AARCH64=4caa0c266ab726b440ccad40a74774167494e001da5de281b74f2d5673e4ace9
RUN /bin/sh -c set -x && UNAME_M="$(uname -m)" && if [ "${UNAME_M}" = "x86_64" ]; then INSTALLER_URL="${INSTALLER_URL_LINUX64}"; SHA256SUM="${SHA256SUM_LINUX64}"; elif [ "${UNAME_M}" = "aarch64" ]; then INSTALLER_URL="${INSTALLER_URL_AARCH64}"; SHA256SUM="${SHA256SUM_AARCH64}"; fi && wget "${INSTALLER_URL}" -O miniconda.sh -q && echo "${SHA256SUM} miniconda.sh" > shasum && sha256sum --check --status shasum && mkdir -p /opt && bash miniconda.sh -b -p /opt/conda && rm miniconda.sh shasum && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && echo "conda activate" >> ~/.bashrc && find /opt/conda/ -follow -type f -name '*.a' -delete && find /opt/conda/ -follow -type f -name '*.js.map' -delete && /opt/conda/bin/conda clean -afy
# --- END COPIED FROM CONDA DOCKERFILE ---

# Mount the data directory
VOLUME ["/data"]
ENV OMNIGIBSON_DATA_PATH /data

# Create a cache directory also
VOLUME ["/cache"]
ENV OMNIGIBSON_APPDATA_PATH /cache/appdata

# Copy over BEHAVIOR source
ADD . /behavior-src
WORKDIR /behavior-src

# Install the BEHAVIOR environment.
# We wrap this in a CUDA install-and-uninstall block to save space.
# (This means curobo will not be able to be rebuilt at runtime).
# Here we also compile this such that it is compatible with GPU architectures
# Turing, Ampere, and Ada; which correspond to 20, 30, and 40 series GPUs.
# We also suppress the output of the installation to avoid the log limit.
# TODO: Add support for aarch64.
RUN wget --no-verbose -O /cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \ 
  dpkg -i /cuda-keyring.deb && rm /cuda-keyring.deb && apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-12-4 && \
  TORCH_CUDA_ARCH_LIST='7.5;8.0;8.6+PTX' PATH=/usr/local/cuda-12.4/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH \
    ./setup.sh --new-env --omnigibson --bddl --joylo --primitives --eval --asset-pipeline --dev --cuda-version 12.4 --accept-conda-tos --accept-nvidia-eula && \
  apt-get remove -y cuda-toolkit-12-4 && apt-get autoremove -y && apt-get autoclean -y && rm -rf /var/lib/apt/lists/*

# Replace the "conda activate" line in bashrc with "conda activate behavior"
RUN sed -i 's/conda activate/conda activate behavior/' /root/.bashrc

# Optionally remove the BEHAVIOR source code if we are in dev mode and change the workdir
ARG DEV_MODE
ENV DEV_MODE=${DEV_MODE}
WORKDIR /behavior-src
RUN if [ "$DEV_MODE" == "1" ]; then \
      cd / && rm -rf /behavior-src; \
    fi

# Add before ENTRYPOINT
RUN echo '#!/bin/bash\n\
source /opt/conda/etc/profile.d/conda.sh\n\
conda activate behavior\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]