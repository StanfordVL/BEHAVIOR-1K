#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=titanrtx:1
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=svl
#SBATCH --account=cvgl
#SBATCH --job-name=import_vid2room_scene_structures
#SBATCH --output=/cvgl2/u/cgokmen/BEHAVIOR-1K/slurm/logs/import_vid2room_scene_structures-%A_%a.log
#SBATCH --error=/cvgl2/u/cgokmen/BEHAVIOR-1K/slurm/logs/import_vid2room_scene_structures-%A_%a.log
#SBATCH --array=0-23

# This script launches a configurable number of concurrent python processes.
# Each process is managed by a separate function call running in the background.
# The script will re-launch a process if it terminates, until a success file is found.

# --- Configuration ---
SCRIPT_NAME="import_vid2room_scene_structures"

# Paths - adjust these as needed
VID2ROOM_ROOT="/vision/group/vid2room/raw/RealEstate10K"
DATASET_NAME="vid2room"
DATASET_ROOT="/cvgl2/u/cgokmen/BEHAVIOR-1K/datasets"
SUCCESS_DIR="${DATASET_ROOT}/${DATASET_NAME}/jobs"

# --- Sanity Check ---
if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
  echo "Error: SLURM_ARRAY_TASK_ID is not set. Exiting." >&2
  exit 1
fi

# Create success directory if it doesn't exist
mkdir -p "${SUCCESS_DIR}"

# Call NVIDIA-SMI to get the GPU ID
GPU_ID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tail -n 1)

# --- Process Management Function ---
# Namespace success file by script name and SLURM array job ID
SUCCESS_FILE="${SUCCESS_DIR}/${SCRIPT_NAME}_${SLURM_ARRAY_TASK_ID}.success"

# Remove stale files from previous runs
if [ -f "${SUCCESS_FILE}" ]; then
  rm -f "${SUCCESS_FILE}"
fi

# Loop until success file exists
while [ ! -f "${SUCCESS_FILE}" ]; do
  echo "[$(date)] Launching process for ID: ${SLURM_ARRAY_TASK_ID} / ${SLURM_ARRAY_TASK_COUNT}"
  
  cd /cvgl2/u/cgokmen/BEHAVIOR-1K
  OMNIGIBSON_APPDATA_PATH=/scr/cgokmen/omnigibson_${GPU_ID}_cache python -u -m omnigibson.examples.scenes.import_vid2room_scene_structures \
    "${SLURM_ARRAY_TASK_ID}" "${SLURM_ARRAY_TASK_COUNT}" \
    --dataset-name "${DATASET_NAME}" \
    --success-file "${SUCCESS_FILE}"
done

echo "[$(date)] Success file found for ID: ${SLURM_ARRAY_TASK_ID}. Process complete."