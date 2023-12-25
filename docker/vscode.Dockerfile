FROM stanfordvl/omnigibson:latest

# environment settings
ARG DEBIAN_FRONTEND="noninteractive"
ENV HOME="/config"
ENV OMNIGIBSON_REMOTE_STREAMING="webrtc"

RUN \
  echo "**** install runtime dependencies ****" && \
  apt-get update && \
  apt-get install -y \
    git \
    jq \
    libatomic1 \
    nano \
    net-tools \
    netcat \
    sudo && \
  echo "**** install code-server ****" && \
  if [ -z ${CODE_RELEASE+x} ]; then \
    CODE_RELEASE=$(curl -sX GET https://api.github.com/repos/coder/code-server/releases/latest \
      | awk '/tag_name/{print $4;exit}' FS='[""]' | sed 's|^v||'); \
  fi && \
  mkdir -p /app/code-server && \
  curl -o \
    /tmp/code-server.tar.gz -L \
    "https://github.com/coder/code-server/releases/download/v${CODE_RELEASE}/code-server-${CODE_RELEASE}-linux-amd64.tar.gz" && \
  tar xf /tmp/code-server.tar.gz -C \
    /app/code-server --strip-components=1 && \
  echo "**** clean up ****" && \
  apt-get clean && \
  rm -rf \
    /config/* \
    /tmp/* \
    /var/lib/apt/lists/* \
    /var/tmp/*

# Remove the omnigibson source code
RUN rm -rf /omnigibson-src

# run command
CMD /app/code-server/bin/code-server \
  --bind-addr 0.0.0.0:${OMNIGIBSON_VSCODE_PORT} \
  --user-data-dir /workspace/vscode-config/data \
  --extensions-dir /workspace/vscode-config/extensions \
  --disable-telemetry \
  --auth password \
  /workspace