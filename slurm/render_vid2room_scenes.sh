#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=titanrtx:1
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=svl
#SBATCH --account=cvgl
#SBATCH --job-name=render_vid2room_scenes
#SBATCH --output=/cvgl2/u/cgokmen/BEHAVIOR-1K/slurm/logs/render_vid2room_scenes-%A_%a.log
#SBATCH --error=/cvgl2/u/cgokmen/BEHAVIOR-1K/slurm/logs/render_vid2room_scenes-%A_%a.log
#SBATCH --array=0-7

# --- Configuration ---
SCRIPT_NAME="render_vid2room_scenes"
DATASET_ROOT="/cvgl2/u/cgokmen/BEHAVIOR-1K/datasets"
OUTPUT_ROOT="${DATASET_ROOT}/vid2room/renders"
SUCCESS_DIR="${DATASET_ROOT}/vid2room/jobs"

# --- Sanity Check ---
if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
  echo "Error: SLURM_ARRAY_TASK_ID is not set. Exiting." >&2
  exit 1
fi

# Create directories if they don't exist
mkdir -p "${SUCCESS_DIR}"
mkdir -p "${OUTPUT_ROOT}"

# Call NVIDIA-SMI to get the GPU ID
GPU_ID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tail -n 1)

# --- Process Management ---
SUCCESS_FILE="${SUCCESS_DIR}/${SCRIPT_NAME}_${SLURM_ARRAY_TASK_ID}.success"

# Remove stale files from previous runs
if [ -f "${SUCCESS_FILE}" ]; then
  rm -f "${SUCCESS_FILE}"
fi

# Loop until success file exists (handles restarts within the Python script)
while [ ! -f "${SUCCESS_FILE}" ]; do
  echo "[$(date)] Launching render process for ID: ${SLURM_ARRAY_TASK_ID} / ${SLURM_ARRAY_TASK_COUNT}"

  cd /cvgl2/u/cgokmen/BEHAVIOR-1K
  OMNIGIBSON_APPDATA_PATH=/scr/cgokmen/omnigibson_${GPU_ID}_cache python -u -m omnigibson.examples.scenes.render_vid2room_scenes \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --total-tasks "${SLURM_ARRAY_TASK_COUNT}" \
    --output-root "${OUTPUT_ROOT}" \
    --success-file "${SUCCESS_FILE}"
done

echo "[$(date)] Rendering complete for ID: ${SLURM_ARRAY_TASK_ID}."
