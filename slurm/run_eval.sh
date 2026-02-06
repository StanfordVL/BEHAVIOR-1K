#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=h200_core_shared
#SBATCH --account=clear
#SBATCH --job-name=run_eval
#SBATCH --output=/home/cgokmen/projects/BEHAVIOR-1K/slurm/eval_logs/run_eval-%A_%a.log
#SBATCH --error=/home/cgokmen/projects/BEHAVIOR-1K/slurm/eval_logs/run_eval-%A_%a.log
#SBATCH --array=0-155

# This script launches a configurable number of concurrent python processes.
# Each process reads its job file and runs data collection for each row.

# --- Configuration ---
SCRIPT_NAME="run_eval"
NUM_JOBS=1
START_FROM=${1:-0}

# Paths - adjust these as needed
PROJECT_ROOT="/home/cgokmen/projects/BEHAVIOR-1K"
EVAL_JOBS_DIR="/home/cgokmen/projects/BEHAVIOR-1K/slurm/eval_jobs"

# --- Sanity Check ---
if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
  echo "Error: SLURM_ARRAY_TASK_ID is not set. Exiting." >&2
  exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p "${PROJECT_ROOT}/slurm/eval_logs"

# --- Process Management Function ---
run_process() {
  local process_id=$1
  local job_file="${EVAL_JOBS_DIR}/${process_id}.csv"
  local log_file="${PROJECT_ROOT}/slurm/eval_logs/process-${SCRIPT_NAME}_${SLURM_ARRAY_JOB_ID}_${process_id}.log"

  # Remove stale log file from previous runs
  if [ -f "${log_file}" ]; then
    rm -f "${log_file}"
  fi

  # Check if job file exists
  if [ ! -f "${job_file}" ]; then
    echo "[$(date)] No job file found for process ID: ${process_id}. Skipping." | tee -a "${log_file}"
    return 0
  fi

  echo "[$(date)] Processing job file for ID: ${process_id}" | tee -a "${log_file}"

  # Read each line from the CSV file (skip header if present)
  while IFS=',' read -r checkpoint target_episodes_file output_path; do
    # Skip empty lines or header
    if [ -z "${checkpoint}" ] || [ "${checkpoint}" = "checkpoint" ]; then
      continue
    fi

    # Trim whitespace
    checkpoint=$(echo "${checkpoint}" | xargs)
    target_episodes_file=$(echo "${target_episodes_file}" | xargs)
    output_path=$(echo "${output_path}" | xargs)

    # Form output path using dataset/split
    mkdir -p "${output_path}"

    echo "[$(date)] Running evaluation: checkpoint=${checkpoint}, target_episodes_file=${target_episodes_file}, output_path=${output_path}" | tee -a "${log_file}"

    cd "${PROJECT_ROOT}/vid2room_policy"
    OMNIGIBSON_HEADLESS=1 OMNIGIBSON_APPDATA_PATH=/tmp/omnigibson/${process_id} python -m vid2scene_policy.policy_evaluation.eval \
    --policy_path "${checkpoint}" \
    --n_episodes 1 \
    --output_dir "${output_path}" \
    --max_steps 3000 \
    --episodes_file "${target_episodes_file}" < /dev/null >> "${log_file}" 2>&1

    exit_code=$?
    if [ ${exit_code} -ne 0 ]; then
      echo "[$(date)] Warning: Evaluation failed for checkpoint ${checkpoint} with exit code ${exit_code}" | tee -a "${log_file}"
    else
      echo "[$(date)] Successfully evaluated checkpoint ${checkpoint}" | tee -a "${log_file}"
    fi

  done < "${job_file}"

  echo "[$(date)] Finished processing job file for ID: ${process_id}" | tee -a "${log_file}"
}

# --- Main Execution ---
echo "Starting rollout collection for SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID} with ${NUM_JOBS} jobs"

for (( i=0; i<NUM_JOBS; i++ )); do
  task_id=$((SLURM_ARRAY_TASK_ID * NUM_JOBS + i + START_FROM))
  run_process "${task_id}" &
done

echo "Waiting for all ${NUM_JOBS} background processes to complete..."
wait
echo "All processes finished. Exiting."

