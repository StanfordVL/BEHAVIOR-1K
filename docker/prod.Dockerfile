FROM stanfordvl/omnigibson-dev

# Copy over omnigibson source
ADD . /omnigibson-src
WORKDIR /omnigibson-src

# Install OmniGibson
RUN pip install -e .

# Add setup to be executed on bash launch
RUN echo "OMNIGIBSON_NO_OMNIVERSE=1 python -m omnigibson.scripts.setup" >> /root/.bashrc
