FROM stanfordvl/omnigibson-dev

# Copy over omnigibson source
ADD . /omnigibson-src
WORKDIR /omnigibson-src

# Install OmniGibson
RUN pip install -e .

# Add setup to be executed on bash launch
RUN echo "OMNIGIBSON_NO_OMNIVERSE=1 python -m omnigibson.scripts.setup" >> /root/.bashrc

# Always assume a command is being called from within the conda virtual env
ENTRYPOINT ["micromamba", "run", "-n", "omnigibson"]

CMD ["/bin/bash"]
