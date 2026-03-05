#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=titanrtx:1
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=svl
#SBATCH --account=cvgl
#SBATCH --job-name=run_eval
#SBATCH --output=/cvgl2/u/cgokmen/BEHAVIOR-1K/slurm/logs/run_eval-%A_%a.log
#SBATCH --error=/cvgl2/u/cgokmen/BEHAVIOR-1K/slurm/logs/run_eval-%A_%a.log
#SBATCH --array=0-47

# --- Configuration ---
SCRIPT_NAME="run_eval"
TOTAL_JOBS=351

# Paths - adjust these as needed
PROJECT_ROOT="/cvgl2/u/cgokmen/BEHAVIOR-1K"
EVAL_JOBS_DIR="${PROJECT_ROOT}/slurm/eval_jobs"
EVAL_SUCCESSES_DIR="${PROJECT_ROOT}/slurm/eval_successes_shorter"
OUTPUT_BASE="/vision/group/vid2room/eval_results_shorter"
GPU_ID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tail -n 1)
mkdir -p ${OUTPUT_BASE}
mkdir -p ${EVAL_SUCCESSES_DIR}

PROCESS_ID=${SLURM_ARRAY_TASK_ID}

while [ ${PROCESS_ID} -lt ${TOTAL_JOBS} ]; do
  # --- Process Management ---
  JOB_FILE="${EVAL_JOBS_DIR}/${PROCESS_ID}.json"
  SUCCESS_FILE="${EVAL_SUCCESSES_DIR}/${PROCESS_ID}.success"

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

  attempt=0
  max_attempts=3

  # Loop until success file exists or max attempts reached
  while [ ! -f "${SUCCESS_FILE}" ] && [ "$attempt" -lt "$max_attempts" ]; do
    ((attempt++))

    # Generate a random UUID
    uuid=$(tr -dc 'a-z0-9' < /dev/urandom | head -c 6)

    # Form output path using dataset/split
    JOB_OUTPUT_PATH="${OUTPUT_BASE}/${PROCESS_ID}"
    mkdir -p "${JOB_OUTPUT_PATH}"

    echo "[$(date)] Running evaluation (Attempt ${attempt}/${max_attempts}): job=${PROCESS_ID}"

    cd "${PROJECT_ROOT}/vid2room_policy"
    OMNIGIBSON_HEADLESS=1 OMNIGIBSON_APPDATA_PATH=/scr/cgokmen/omnigibson_${GPU_ID}_cache python -u -m vid2scene_policy.policy_evaluation.eval \
      --episodes-file "${JOB_FILE}" \
      --output-dir "${JOB_OUTPUT_PATH}" \
      --success-file "${SUCCESS_FILE}" \
      --max-steps-multiplier 1.1
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