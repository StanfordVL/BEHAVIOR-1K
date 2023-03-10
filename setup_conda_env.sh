#!/usr/bin/env bash

# Helper function to check whether the script is soruced
is_sourced() {
  if [ -n "$ZSH_VERSION" ]; then 
    case $ZSH_EVAL_CONTEXT in *:file:*) return 0;; esac
  else
    case ${0##*/} in dash|-dash|bash|-bash|ksh|-ksh|sh|-sh) return 0;; esac
  fi
  return 1
}
is_sourced && sourced=1 || sourced=0

if [[ $sourced == 0 ]]; then
  echo "Please source the script to make sure conda env is created successfully!"
  exit
fi

# Make sure that the ISAAC_SIM_PATH variable is set
if [[ x"${ISAAC_SIM_PATH}" == "x" ]]; then
  echo "Please set ISAAC_SIM_PATH!"
  return
fi

# We search to see if we've already modified this config; if not, we add a couple additional dependencies
# Ideally this would be a lot cleaner by having a separate config altogether, but isaac sim complains if we
# don't use this EXACT file in this EXACT location
CFG_PATH="${ISAAC_SIM_PATH}/apps/omni.isaac.sim.python.kit"
if grep -qxF '# OmniGibson additional dependencies' ${CFG_PATH}; then
  echo 'Already found OmniGibson additional dependencies; skipping modifying isaac-sim python config!'
else
  echo '' >> ${CFG_PATH}
  echo '# OmniGibson additional dependencies' >> ${CFG_PATH}
  echo '"omni.flowusd" = {}' >> ${CFG_PATH}
  echo '"omni.particle.system.bundle" = {}' >> ${CFG_PATH}
fi

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

# Cycle conda environment so that all dependencies are propagated
conda deactivate
conda activate omnigibson

echo "OmniGibson successfully installed!"
