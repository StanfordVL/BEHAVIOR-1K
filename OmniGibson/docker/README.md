# Requirements

- Modern Linux distribution (Ubuntu 20.04, Fedora 36, etc.)
- RTX capable Nvidia graphics card (20 series or newer,)
- Up-to-date NVIDIA drivers

# Usage

**The below instructions concern the usage of OmniGibson containers with self-built images. Please see the BEHAVIOR-1K docs for instructions on how to pull and run a cloud image.**

1. Set up the NVIDIA Docker Runtime and login to the NVIDIA Container Registry
See [here](https://www.pugetsystems.com/labs/hpc/how-to-setup-nvidia-docker-and-ngc-registry-on-your-workstation-part-4-accessing-the-ngc-registry-1115/) for details.

2. Build the container. **From the OmniGibson root**, run: `./docker/build_docker.sh`

3. Run the container
* To get a shell inside a container with GUI: `sudo ./docker/run_docker_gui.sh`
* To get a jupyter notebook: `sudo ./docker/run_docker_notebook.sh`
* To get access to a shell inside a headless container `sudo ./docker/run_docker.sh`

# Development
To push a Docker container, run: `sudo ./docker/push_docker.sh`