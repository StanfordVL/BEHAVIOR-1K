#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=import_spoc_scene_structures
#SBATCH --output=/home/cgokmen/projects/BEHAVIOR-1K/slurm/logs/import_spoc_scene_structures-%A_%a.log
#SBATCH --error=/home/cgokmen/projects/BEHAVIOR-1K/slurm/logs/import_spoc_scene_structures-%A_%a.log
#SBATCH --array=0-127

# This script launches a configurable number of concurrent python processes.
# Each process is managed by a separate function call running in the background.
# The script will re-launch a process if it terminates, until a success file is found.

# --- Configuration ---
SCRIPT_NAME="import_spoc_scene_structures"
NUM_JOBS=${1:-8}
TOTAL_JOBS_IN_ARRAY=$((NUM_JOBS * SLURM_ARRAY_TASK_COUNT))
DATASET_ROOT="/home/cgokmen/projects/BEHAVIOR-1K/datasets"
SUCCESS_DIR="${DATASET_ROOT}/spoc/jobs"

# --- Sanity Check ---
if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
  echo "Error: SLURM_ARRAY_TASK_ID is not set. Exiting." >&2
  exit 1
fi

# Create success directory if it doesn't exist
mkdir -p "${SUCCESS_DIR}"
mkdir -p "/home/cgokmen/projects/BEHAVIOR-1K/slurm/logs"

# --- Process Management Function ---
manage_process() {
  local process_id=$1
  # Namespace success file by script name and SLURM array job ID
  local success_file="${SUCCESS_DIR}/${SCRIPT_NAME}_${SLURM_ARRAY_JOB_ID}_${process_id}.success"
  local log_file="/home/cgokmen/projects/BEHAVIOR-1K/slurm/logs/${SCRIPT_NAME}_${SLURM_ARRAY_JOB_ID}_${process_id}.log"

  # Remove stale files from previous runs
  if [ -f "${success_file}" ]; then
    rm -f "${success_file}"
  fi
  if [ -f "${log_file}" ]; then
    rm -f "${log_file}"
  fi

  # Loop until success file exists
  while [ ! -f "${success_file}" ]; do
    echo "[$(date)] Launching process for ID: ${process_id} / ${TOTAL_JOBS_IN_ARRAY}"
    
    cd /home/cgokmen/projects/BEHAVIOR-1K
    python -u -m omnigibson.examples.scenes.import_spoc_scene_structures \
      "${process_id}" "${TOTAL_JOBS_IN_ARRAY}" \
      --success-prefix "${SCRIPT_NAME}_${SLURM_ARRAY_JOB_ID}" \
      >> "${log_file}" 2>&1
  done
  
  echo "[$(date)] Success file found for ID: ${process_id}. Process complete."
}

# --- Main Execution ---
echo "Starting process manager for SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID} with ${NUM_JOBS} jobs"

for (( i=0; i<NUM_JOBS; i++ )); do
  task_id=$((SLURM_ARRAY_TASK_ID * NUM_JOBS + i))
  manage_process "${task_id}" &
done

echo "Waiting for all ${NUM_JOBS} background processes to complete..."
wait
echo "All processes finished successfully. Exiting."
