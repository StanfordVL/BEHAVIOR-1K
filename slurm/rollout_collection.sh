#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=h200_core_shared
#SBATCH --account=clear
#SBATCH --job-name=rollout_collection
#SBATCH --output=/home/cgokmen/projects/BEHAVIOR-1K/slurm/logs/rollout_collection-%A_%a.log
#SBATCH --error=/home/cgokmen/projects/BEHAVIOR-1K/slurm/logs/rollout_collection-%A_%a.log
#SBATCH --array=0-255

# This script launches a configurable number of concurrent python processes.
# Each process reads its job file and runs data collection for each row.

# --- Configuration ---
SCRIPT_NAME="rollout_collection"
NUM_JOBS=${1:-4}

# Paths - adjust these as needed
PROJECT_ROOT="/home/cgokmen/projects/BEHAVIOR-1K"
ROLLOUT_JOBS_DIR="/home/cgokmen/projects/BEHAVIOR-1K/slurm/rollout_jobs"
OUTPUT_BASE="/checkpoint/clear/cgokmen/lerobot_datasets"

# --- Sanity Check ---
if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
  echo "Error: SLURM_ARRAY_TASK_ID is not set. Exiting." >&2
  exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p "${PROJECT_ROOT}/slurm/logs"

# --- Process Management Function ---
run_process() {
  local process_id=$1
  local job_file="${ROLLOUT_JOBS_DIR}/${process_id}.csv"
  local log_file="${PROJECT_ROOT}/slurm/logs/process-${SCRIPT_NAME}_${SLURM_ARRAY_JOB_ID}_${process_id}.log"

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
  while IFS=',' read -r dataset split scene count; do
    # Skip empty lines or header
    if [ -z "${dataset}" ] || [ "${dataset}" = "dataset" ]; then
      continue
    fi

    # Trim whitespace
    dataset=$(echo "${dataset}" | xargs)
    split=$(echo "${split}" | xargs)
    scene=$(echo "${scene}" | xargs)
    count=$(echo "${count}" | xargs)
    uuid=$(tr -dc 'a-z0-9' < /dev/urandom | head -c 6)

    # Form output path using dataset/split
    output_path="${OUTPUT_BASE}/${dataset}-${split}"
    mkdir -p "${output_path}"

    echo "[$(date)] Running collection: dataset=${dataset}, split=${split}, scene=${scene}, count=${count}" | tee -a "${log_file}"

    cd "${PROJECT_ROOT}/vid2room_policy"
    OMNIGIBSON_HEADLESS=1 OMNIGIBSON_APPDATA_PATH=/tmp/omnigibson/${process_id} python -u collect_data.py \
      --dataset "${dataset}" \
      --scene "${scene}" \
      --episodes "${count}" \
      --output "${output_path}" \
      --repo-id "${scene}-${uuid}" \
      >> "${log_file}" 2>&1

    exit_code=$?
    if [ ${exit_code} -ne 0 ]; then
      echo "[$(date)] Warning: Collection failed for scene ${scene} with exit code ${exit_code}" | tee -a "${log_file}"
    else
      echo "[$(date)] Successfully collected ${count} episodes for scene ${scene}" | tee -a "${log_file}"
    fi

  done < "${job_file}"

  echo "[$(date)] Finished processing job file for ID: ${process_id}" | tee -a "${log_file}"
}

# --- Main Execution ---
echo "Starting rollout collection for SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID} with ${NUM_JOBS} jobs"

for (( i=0; i<NUM_JOBS; i++ )); do
  task_id=$((SLURM_ARRAY_TASK_ID * NUM_JOBS + i + 8192))
  run_process "${task_id}" &
done

echo "Waiting for all ${NUM_JOBS} background processes to complete..."
wait
echo "All processes finished. Exiting."

