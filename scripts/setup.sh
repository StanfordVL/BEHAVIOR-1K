#!/usr/bin/env bash
set -eo &> /dev/null

# Choose venv name 
echo "The new conda environment will be named [4momnigibson[0m by default."
read -p "If you want to use a different name, please type in here (press enter to skip) >>> " conda_name
conda_name=${conda_name:-omnigibson}
echo -e "\nUsing [4m$conda_name[0m as the conda environment name\n"

# Ask the user for Omniverse installation method
echo "Please choose your installation method for Isaac-Sim:"
echo "1. Install Isaac-Sim from Python pip install (default, recommended)"
echo "2. Install Isaac-Sim from the Omniverse Launcher"

while true; do
    read -p "Please enter your choice (1 or 2) >>> " isaac_sim_installation_method
    
    # Set default to 1 if no input is provided
    isaac_sim_installation_method=${isaac_sim_installation_method:-1}
    
    # Validate input
    if [[ "$isaac_sim_installation_method" == "1" || "$isaac_sim_installation_method" == "2" ]]; then
        break
    else
        echo "Invalid input. Please enter 1 or 2."
    fi
done

# If the user chooses to install Isaac-Sim from pip
if [ $isaac_sim_installation_method -eq 1 ]; then

    # Get Python version from Isaac Sim (currently supporting Python 3.10 only)
    echo Installing Isaac Sim from pip install...
    ISAAC_PYTHON_VERSION=3.10
    echo Using Python version [4m$ISAAC_PYTHON_VERSION[0m matching your current Isaac Sim version
fi

# If the user chooses to install Isaac-Sim from the Omniverse Launcher
if [ $isaac_sim_installation_method -eq 2 ]; then
    
    # Make sure that the ISAAC_SIM_PATH variable is set correctly
    if [[ -d ~/.local/share/ov/pkg ]] && [[ $(ls ~/.local/share/ov/pkg | grep isaac_sim) ]]; 
    then
    FOUND_ISAAC_SIM_PATH=$(ls -d ~/.local/share/ov/pkg/* | grep isaac_sim | tail -n 1) 
    echo "We found Isaac Sim installed at [4m$FOUND_ISAAC_SIM_PATH[0m. OmniGibson will use it by default."
    read -p "If you want to use a different one, please type in the path containing isaac-sim.sh here (press enter to skip) >>> " ISAAC_SIM_PATH
    ISAAC_SIM_PATH=${ISAAC_SIM_PATH:-$FOUND_ISAAC_SIM_PATH}
    else
    echo "We did not find Isaac Sim under ~/.local/share/ov/pkg."
    echo "If you haven't installed Isaac Sim yet, please do so before running this setup script."
    read -p "If you have already installed it in a custom location, please type in the path containing isaac-sim.sh here >>> " ISAAC_SIM_PATH
    fi

    while [[ ! -f "${ISAAC_SIM_PATH}/isaac-sim.sh" ]]; do
    read -p "isaac-sim.sh not found in [4m$ISAAC_SIM_PATH[0m! Make sure you have entered the correct path >>> " ISAAC_SIM_PATH
    done
    echo -e "\nUsing Isaac Sim at [4m$ISAAC_SIM_PATH[0m\n"

    # Get Python version from Isaac Sim
    ISAAC_PYTHON_VERSION=$(${ISAAC_SIM_PATH}/python.sh -c "import platform; print(platform.python_version())")
    ISAAC_PYTHON_VERSION="${ISAAC_PYTHON_VERSION##*$'\n'}" # get rid of conda activation warnings
    echo Using Python version [4m$ISAAC_PYTHON_VERSION[0m matching your current Isaac Sim version

fi


# Create a conda environment with the appropriate python version
source $(conda info --base)/etc/profile.d/conda.sh
conda create -y -n $conda_name python=${ISAAC_PYTHON_VERSION}

# Now activate the omnigibson environment
conda activate $conda_name

# Install omnigibson!
pip install -e .

# # Install Isaac Sim using pip if the user chooses to install it from pip
# if [ $isaac_sim_installation_method -eq 1 ]; then
#     echo -e "\nInstalling Isaac Sim using pip...\n"


mkdir -p ${CONDA_PREFIX}/etc/conda/activate.d
mkdir -p ${CONDA_PREFIX}/etc/conda/deactivate.d
touch ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh
touch ${CONDA_PREFIX}/etc/conda/deactivate.d/env_vars.sh
# We add some preprocessing information so that the Isaac Sim paths are linked to this environment upon startup
# See https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#macos-and-linux for reference
CONDA_ACT_FILE="${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh"
echo '#!/bin/sh' > ${CONDA_ACT_FILE}
echo "export LD_LIBRARY_PATH_OLD=\$LD_LIBRARY_PATH" >> ${CONDA_ACT_FILE}
echo "export PYTHONPATH_OLD=\$PYTHONPATH" >> ${CONDA_ACT_FILE}

CONDA_DEACT_FILE="${CONDA_PREFIX}/etc/conda/deactivate.d/env_vars.sh"
echo '#!/bin/sh' > ${CONDA_DEACT_FILE}
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH_OLD" >> ${CONDA_DEACT_FILE}
echo "export PYTHONPATH=\$PYTHONPATH_OLD" >> ${CONDA_DEACT_FILE}
echo "unset ISAAC_PATH" >> ${CONDA_DEACT_FILE}
echo "unset CARB_APP_PATH" >> ${CONDA_DEACT_FILE}
echo "unset LD_LIBRARY_PATH_OLD" >> ${CONDA_DEACT_FILE}
echo "unset PYTHONPATH_OLD" >> ${CONDA_DEACT_FILE}
echo "unset EXP_PATH" >> ${CONDA_DEACT_FILE}

# # Add the Isaac Sim paths to the conda environment for pip install 
# if [ $isaac_sim_installation_method -eq 1 ]; then
#     echo "export ISAAC_PATH=$(python -c 'import site; print(site.getsitepackages()[0] + "/isaacsim")')" >> ${CONDA_ACT_FILE}
#     echo "export EXP_PATH=\$ISAAC_PATH/apps" >> ${CONDA_ACT_FILE}
# fi

# Add the Isaac Sim paths to the conda environment for Omniverse Launcher install
if [ $isaac_sim_installation_method -eq 2 ]; then
    echo "source ${ISAAC_SIM_PATH}/setup_conda_env.sh" >> ${CONDA_ACT_FILE}
fi


echo -e "\nPerforming clean up...\n"

# Cycle conda environment so that all dependencies are propagated
conda deactivate

# # If using pip install, remove the downloaded whl files
# if [ $isaac_sim_installation_method -eq 1 ]; then
#     rm omniverse_kit-106.0.0-cp310-none-manylinux_2_31_x86_64.whl
#     rm isaacsim_kernel-4.0.0.0-cp310-none-manylinux_2_31_x86_64.whl
#     rm isaacsim_app-4.0.0.0-cp310-none-manylinux_2_31_x86_64.whl
#     rm isaacsim_core-4.0.0.0-cp310-none-manylinux_2_31_x86_64.whl
#     rm isaacsim_gui-4.0.0.0-cp310-none-manylinux_2_31_x86_64.whl
#     rm isaacsim_utils-4.0.0.0-cp310-none-manylinux_2_31_x86_64.whl
#     rm isaacsim_storage-4.0.0.0-cp310-none-manylinux_2_31_x86_64.whl
#     rm isaacsim_asset-4.0.0.0-cp310-none-manylinux_2_31_x86_64.whl
#     rm isaacsim_sensor-4.0.0.0-cp310-none-manylinux_2_31_x86_64.whl
#     rm isaacsim_robot_motion-4.0.0.0-cp310-none-manylinux_2_31_x86_64.whl
#     rm isaacsim_robot-4.0.0.0-cp310-none-manylinux_2_31_x86_64.whl
#     rm isaacsim_benchmark-4.0.0.0-cp310-none-manylinux_2_31_x86_64.whl
#     rm isaacsim_code_editor-4.0.0.0-cp310-none-manylinux_2_31_x86_64.whl
#     rm isaacsim_ros1-4.0.0.0-cp310-none-manylinux_2_31_x86_64.whl
#     rm isaacsim_cortex-4.0.0.0-cp310-none-manylinux_2_31_x86_64.whl
#     rm isaacsim_example-4.0.0.0-cp310-none-manylinux_2_31_x86_64.whl
#     rm isaacsim_replicator-4.0.0.0-cp310-none-manylinux_2_31_x86_64.whl
#     rm isaacsim_rl-4.0.0.0-cp310-none-manylinux_2_31_x86_64.whl
#     rm isaacsim_robot_setup-4.0.0.0-cp310-none-manylinux_2_31_x86_64.whl
#     rm isaacsim_ros2-4.0.0.0-cp310-none-manylinux_2_31_x86_64.whl
#     rm isaacsim_template-4.0.0.0-cp310-none-manylinux_2_31_x86_64.whl
#     rm isaacsim_test-4.0.0.0-cp310-none-manylinux_2_31_x86_64.whl
#     rm isaacsim-4.0.0.0-cp310-none-manylinux_2_31_x86_64.whl
#     rm isaacsim_extscache_physics-4.0.0.0-cp310-none-manylinux_2_31_x86_64.whl
#     rm isaacsim_extscache_kit-4.0.0.0-cp310-none-manylinux_2_31_x86_64.whl
#     rm isaacsim_extscache_kit_sdk-4.0.0.0-cp310-none-manylinux_2_31_x86_64.whl
# fi

echo -e "\nOmniGibson successfully installed! Please run [4mconda activate $conda_name[0m to activate the environment.\n"
