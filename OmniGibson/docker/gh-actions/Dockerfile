FROM stanfordvl/omnigibson-dev:og-develop

ARG DUMB_INIT_VERSION="1.2.2"
ARG GIT_CORE_PPA_KEY="A1715D88E1DF1F24"

ENV GIT_LFS_VERSION="3.2.0"
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
# hadolint ignore=SC2086,DL3015,DL3008,DL3013,SC2015
RUN echo en_US.UTF-8 UTF-8 >> /etc/locale.gen \
  && apt-get update \
  && apt-get install -y --no-install-recommends gnupg \
  && ( \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys ${GIT_CORE_PPA_KEY} \
      || apt-key adv --keyserver pgp.mit.edu --recv-keys ${GIT_CORE_PPA_KEY} \
      || apt-key adv --keyserver keyserver.pgp.com --recv-keys ${GIT_CORE_PPA_KEY} \
  ) \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
    gnupg \
    lsb-release \
    curl \
    tar \
    unzip \
    zip \
    apt-transport-https \
    ca-certificates \
    sudo \
    gpg-agent \
    software-properties-common \
    build-essential \
    zlib1g-dev \
    zstd \
    gettext \
    libcurl4-openssl-dev \
    inetutils-ping \
    jq \
    wget \
    dirmngr \
    openssh-client \
    locales \
    python3-pip \
    python3-setuptools \
    python3-venv \
    python3 \
    dumb-init \
    nodejs \
    rsync \
    libpq-dev \
    gosu \
    pkg-config \
    graphviz \
  && DPKG_ARCH="$(dpkg --print-architecture)" \
  && LSB_RELEASE_CODENAME="$(lsb_release --codename | cut -f2)" \
  && sed -e 's/Defaults.*env_reset/Defaults env_keep = "HTTP_PROXY HTTPS_PROXY NO_PROXY FTP_PROXY http_proxy https_proxy no_proxy ftp_proxy"/' -i /etc/sudoers \
  && echo deb http://ppa.launchpad.net/git-core/ppa/ubuntu $([[ $(grep -E '^ID=' /etc/os-release | sed 's/.*=//g') == "ubuntu" ]] && (grep VERSION_CODENAME /etc/os-release | sed 's/.*=//g') || echo bionic) main>/etc/apt/sources.list.d/git-core.list \
  && apt-get update \
  && ( apt-get install -y --no-install-recommends git || apt-get install -t stable -y --no-install-recommends git || apt-get install -y --no-install-recommends git=1:2.33.1-0ppa1~ubuntu18.04.1 git-man=1:2.33.1-0ppa1~ubuntu18.04.1 ) \
  && ( [[ $(apt-cache search -n liblttng-ust0 | awk '{print $1}') == "liblttng-ust0" ]] && apt-get install -y --no-install-recommends liblttng-ust0 || : ) \
  && ( [[ $(apt-cache search -n liblttng-ust1 | awk '{print $1}') == "liblttng-ust1" ]] && apt-get install -y --no-install-recommends liblttng-ust1 || : ) \
  && ( ( curl "https://awscli.amazonaws.com/awscli-exe-linux-$(uname -m).zip" -o "awscliv2.zip" && unzip awscliv2.zip -d /tmp/ && /tmp/aws/install && rm awscliv2.zip) || pip3 install --no-cache-dir awscli ) \
  && ( curl -s "https://github.com/git-lfs/git-lfs/releases/download/v${GIT_LFS_VERSION}/git-lfs-linux-${DPKG_ARCH}-v${GIT_LFS_VERSION}.tar.gz" -L -o /tmp/lfs.tar.gz && tar -xzf /tmp/lfs.tar.gz -C /tmp && /tmp/git-lfs-${GIT_LFS_VERSION}/install.sh && rm -rf /tmp/lfs.tar.gz  /tmp/git-lfs-${GIT_LFS_VERSION}) \
  && distro=$(lsb_release -is | awk '{print tolower($0)}') \
  && VERSION_ID=$(lsb_release -r | cut -f2) \
  && echo "deb http://download.opensuse.org/repositories/devel:/kubic:/libcontainers:/stable/xUbuntu_${VERSION_ID}/ /" | sudo tee /etc/apt/sources.list.d/devel-kubic-libcontainers-stable.list \
  && curl -Ls https://download.opensuse.org/repositories/devel:kubic:libcontainers:stable/xUbuntu_$VERSION_ID/Release.key | apt-key add - \
  && apt-get update \
  && apt-get install buildah podman fuse-overlayfs -y \
  && sed -i 's/^\[machine\]$/#\[machine\]/' /usr/share/containers/containers.conf \
  # && sed -i 's/#mount_program/mount_program/g' /etc/containers/storage.conf \
  && mkdir -p /etc/apt/keyrings \
  && ( curl -fsSL https://download.docker.com/linux/${distro}/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg ) \
  && version=$(lsb_release -cs | sed 's/trixie\|n\/a/bookworm/g') \
  && ( echo "deb [arch=${DPKG_ARCH} signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/${distro} ${version} stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null ) \
  && apt-get update \
  && apt-get install -y docker-ce docker-ce-cli docker-buildx-plugin containerd.io docker-compose-plugin --no-install-recommends --allow-unauthenticated \
  && echo -e '#!/bin/sh\ndocker compose --compatibility "$@"' > /usr/local/bin/docker-compose && chmod +x /usr/local/bin/docker-compose \
  && ( [[ "${LSB_RELEASE_CODENAME}" == "focal" ]] && ( echo "deb https://download.opensuse.org/repositories/devel:/kubic:/libcontainers:/stable/xUbuntu_20.04/ /" | tee /etc/apt/sources.list.d/devel:kubic:libcontainers:stable.list && curl -L "https://build.opensuse.org/projects/devel:kubic/public_key" | apt-key add -; echo "deb https://download.opensuse.org/repositories/devel:/kubic:/libcontainers:/stable/xUbuntu_20.04/ /" | tee /etc/apt/sources.list.d/devel:kubic:libcontainers:stable.list && curl -L "https://build.opensuse.org/projects/devel:kubic/public_key" | apt-key add - && apt-get update) || : ) \
  && ( [[ "${LSB_RELEASE_CODENAME}" == "focal" || "${LSB_RELEASE_CODENAME}" == "jammy" || "${LSB_RELEASE_CODENAME}" == "sid" || "${LSB_RELEASE_CODENAME}" == "bullseye" ]] && apt-get install -y --no-install-recommends podman buildah skopeo || : ) \
  && ( [[ "${LSB_RELEASE_CODENAME}" == "jammy" ]] && echo "Ubuntu Jammy is marked as beta. Please see https://github.com/actions/virtual-environments/issues/5490" || : ) \
  && GH_CLI_VERSION=$(curl -sL -H "Accept: application/vnd.github+json"   https://api.github.com/repos/cli/cli/releases/latest | jq -r '.tag_name' | sed 's/^v//g') \
  && GH_CLI_DOWNLOAD_URL=$(curl -sL -H "Accept: application/vnd.github+json"   https://api.github.com/repos/cli/cli/releases/latest | jq ".assets[] | select(.name == \"gh_${GH_CLI_VERSION}_linux_${DPKG_ARCH}.deb\")" | jq -r '.browser_download_url') \
  && curl -sSLo /tmp/ghcli.deb ${GH_CLI_DOWNLOAD_URL} && apt-get -y install /tmp/ghcli.deb && rm /tmp/ghcli.deb \
  && YQ_VERSION=$(curl -sL -H "Accept: application/vnd.github+json" https://api.github.com/repos/mikefarah/yq/releases/latest | jq -r '.tag_name' | sed 's/^v//g') \
  && YQ_DOWNLOAD_URL=$(curl -sL -H "Accept: application/vnd.github+json" https://api.github.com/repos/mikefarah/yq/releases/latest | jq ".assets[] | select(.name == \"yq_linux_${DPKG_ARCH}.tar.gz\")" | jq -r '.browser_download_url') \
  && ( curl -s ${YQ_DOWNLOAD_URL} -L -o /tmp/yq.tar.gz && tar -xzf /tmp/yq.tar.gz -C /tmp && mv /tmp/yq_linux_${DPKG_ARCH} /usr/local/bin/yq) \
  && rm -rf /var/lib/apt/lists/* \
  && rm -rf /tmp/* \
  && groupadd -g 121 runner \
  && useradd -mr -d /home/runner -u 1001 -g 121 runner \
  && usermod -aG sudo runner \
  && usermod -aG docker runner \
  && echo '%sudo ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers \
  && ( [[ -f /etc/apt/sources.list.d/devel:kubic:libcontainers:stable.list ]] && rm /etc/apt/sources.list.d/devel:kubic:libcontainers:stable.list || : )

ENV AGENT_TOOLSDIRECTORY=/opt/hostedtoolcache
RUN mkdir -p /opt/hostedtoolcache

ARG GH_RUNNER_VERSION="2.320.0"

ARG TARGETPLATFORM

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /actions-runner
COPY install_actions.sh /actions-runner

RUN chmod +x /actions-runner/install_actions.sh \
  && /actions-runner/install_actions.sh ${GH_RUNNER_VERSION} ${TARGETPLATFORM} \
  && rm /actions-runner/install_actions.sh \
  && chown runner /_work /actions-runner /opt/hostedtoolcache

COPY token.sh entrypoint.sh app_token.sh /
RUN chmod +x /token.sh /entrypoint.sh /app_token.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["./bin/Runner.Listener", "run", "--startuptype", "service"]