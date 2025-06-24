FROM stanfordvl/omnigibson:og-develop

ADD environment.yml /tmp/environment.yml

SHELL ["/bin/bash", "-c"]

# Create the pipeline environment
RUN micromamba env create --file /tmp/environment.yml

# Install CoACD, open3d
RUN micromamba run -n pipeline pip install coacd open3d

# Install V-HACD
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y make build-essential cmake && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/kmammou/v-hacd.git /root/vhacd && \
    cd /root/vhacd/app && \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build && \
    mv build/TestVHACD /bin/TestVHACD

# Install shapely in omnigibson env
RUN micromamba run -n omnigibson pip install shapely

# Clone BDDL and install it in omnigibson env. We do this to allow external mounting.
RUN git clone --branch develop https://github.com/StanfordVL/bddl.git /bddl-src && \
    micromamba run -n omnigibson pip install -e /bddl-src

# Launch a dask worker - the first argument needs to be the scheduler.
ENTRYPOINT ["micromamba", "run", "-n", "pipeline", "dask-worker", "--nworkers=1", "--nthreads=1"]
