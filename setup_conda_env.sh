#!/usr/bin/env bash

# Make sure that the ISAAC_SIM_PATH variable is set
if [ x"${ISAAC_SIM_PATH}" == "x" ]; then
  echo "Please set ISAAC_SIM_PATH!"
  exit
fi

# First, we prune some packages from Isaac Sim that are outdated relative to what we need in OmniGibson
rm -r "${ISAAC_SIM_PATH}/exts/omni.isaac.ml_archive/pip_prebundle/gym*"

# Create a conda environment with python 3.7
conda create -y -n omnigibson python=3.7

# Now activate the omnigibson environment
conda activate omnigibson

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
echo "source ${ISAAC_SIM_PATH}/setup_conda_env.sh" >> ${CONDA_ACT_FILE}

CONDA_DEACT_FILE="${CONDA_PREFIX}/etc/conda/deactivate.d/env_vars.sh"
echo '#!/bin/sh' > ${CONDA_DEACT_FILE}
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH_OLD" >> ${CONDA_DEACT_FILE}
echo "export PYTHONPATH=\$PYTHONPATH_OLD" >> ${CONDA_DEACT_FILE}
echo "unset ISAAC_PATH" >> ${CONDA_DEACT_FILE}
echo "unset CARB_APP_PATH" >> ${CONDA_DEACT_FILE}
echo "unset LD_LIBRARY_PATH_OLD" >> ${CONDA_DEACT_FILE}
echo "unset PYTHONPATH_OLD" >> ${CONDA_DEACT_FILE}

# Install omnigibson!
pip install -e . 

