FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

WORKDIR /gello

# Set environment variables first (less likely to change)
ENV PYTHONPATH=/gello:/gello/third_party/oculus_reader/

# Group apt updates and installs together
RUN apt update && apt install -y \
    libhidapi-dev \
    python3-pip \
    android-tools-adb \
    libegl1-mesa-dev && \
    rm -rf /var/lib/apt/lists/* 


# Python alias setup
RUN echo "alias python=python3" >> ~/.bashrc

# Install Python dependencies
COPY requirements.txt /gello
RUN pip install -r requirements.txt