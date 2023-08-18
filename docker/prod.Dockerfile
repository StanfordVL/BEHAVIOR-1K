FROM stanfordvl/omnigibson-dev

# Copy over omnigibson source
ADD . /omnigibson-src
WORKDIR /omnigibson-src

SHELL ["micromamba", "run", "-n", "omnigibson", "/bin/bash", "--login", "-c"]

# Install OmniGibson
RUN micromamba run -n omnigibson pip install -e .

# Add setup to be executed on bash launch
RUN echo "OMNIGIBSON_NO_OMNIVERSE=1 python scripts/download_datasets.py" >> /root/.bashrc
