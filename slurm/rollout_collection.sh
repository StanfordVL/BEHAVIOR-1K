#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=titanrtx:1
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=svl
#SBATCH --account=cvgl
#SBATCH --job-name=rollout_collection
#SBATCH --output=/cvgl2/u/cgokmen/BEHAVIOR-1K/slurm/logs/rollout_collection-%A_%a.log
#SBATCH --error=/cvgl2/u/cgokmen/BEHAVIOR-1K/slurm/logs/rollout_collection-%A_%a.log
#SBATCH --array=0-47

# --- Configuration ---
SCRIPT_NAME="rollout_collection"
START_FROM=${1:-0}
TOTAL_JOBS=329

# Paths - adjust these as needed
PROJECT_ROOT="/cvgl2/u/cgokmen/BEHAVIOR-1K"
ROLLOUT_JOBS_DIR="${PROJECT_ROOT}/slurm/rollout_jobs"
ROLLOUT_SUCCESSES_DIR="${PROJECT_ROOT}/slurm/rollout_successes"
OUTPUT_BASE="/vision/group/vid2room/rollouts"
GPU_ID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tail -n 1)
mkdir -p ${OUTPUT_BASE}

PROCESS_ID=$((${SLURM_ARRAY_TASK_ID} + ${START_FROM}))

while [ ${PROCESS_ID} -lt ${TOTAL_JOBS} ]; do
  # --- Process Management ---
  JOB_FILE="${ROLLOUT_JOBS_DIR}/${PROCESS_ID}.csv"
  SUCCESS_FILE="${ROLLOUT_SUCCESSES_DIR}/${PROCESS_ID}.success"

  # Remove stale files from previous runs
  if [ -f "${SUCCESS_FILE}" ]; then
    rm -f "${SUCCESS_FILE}"
  fi

  # Check if job file exists
  if [ ! -f "${JOB_FILE}" ]; then
    echo "[$(date)] No job file found for process ID: ${PROCESS_ID}. Skipping."
    PROCESS_ID=$((${PROCESS_ID} + ${SLURM_ARRAY_TASK_COUNT}))
    continue
  fi

  echo "[$(date)] Processing job file for ID: ${PROCESS_ID}"

  IFS=, read -r dataset split scene count < "$JOB_FILE"

  # Trim whitespace using xargs
  dataset=$(echo "${dataset}" | xargs)
  split=$(echo "${split}" | xargs)
  scene=$(echo "${scene}" | xargs)
  count=$(echo "${count}" | xargs)

  # FIX: Moved the header/empty check outside the while loop to prevent an infinite loop
  if [ -z "${dataset}" ] || [ "${dataset}" = "dataset" ]; then
    echo "[$(date)] Job file ${PROCESS_ID} is empty or a header. Exiting cleanly."
    PROCESS_ID=$((${PROCESS_ID} + ${SLURM_ARRAY_TASK_COUNT}))
    continue
  fi

  attempt=0
  max_attempts=3

  # Loop until success file exists or max attempts reached
  while [ ! -f "${SUCCESS_FILE}" ] && [ "$attempt" -lt "$max_attempts" ]; do
    ((attempt++))

    # Generate a random UUID
    uuid=$(tr -dc 'a-z0-9' < /dev/urandom | head -c 6)

    # Form output path using dataset/split
    output_path="${OUTPUT_BASE}/${dataset}-${split}"
    mkdir -p "${output_path}"

    echo "[$(date)] Running collection (Attempt ${attempt}/${max_attempts}): dataset=${dataset}, split=${split}, scene=${scene}, count=${count}"

    cd "${PROJECT_ROOT}/vid2room_policy"
    OMNIGIBSON_HEADLESS=1 OMNIGIBSON_APPDATA_PATH=/scr/cgokmen/omnigibson_${GPU_ID}_cache python -u collect_data.py \
      --scene_dataset "${dataset}" \
      --scene_name "${scene}" \
      --num_episodes "${count}" \
      --output "${output_path}" \
      --repo-id "${scene}-${uuid}" \
      --success-file "${SUCCESS_FILE}"
      
  done

  if [ ! -f "${SUCCESS_FILE}" ]; then
    echo "[$(date)] FAILED: Exceeded ${max_attempts} attempts for ID: ${PROCESS_ID}"
    PROCESS_ID=$((${PROCESS_ID} + ${SLURM_ARRAY_TASK_COUNT}))
    continue
  fi

  echo "[$(date)] Finished processing job file for ID: ${PROCESS_ID}"

  PROCESS_ID=$((${PROCESS_ID} + ${SLURM_ARRAY_TASK_COUNT}))
done

echo "[$(date)] All jobs completed."