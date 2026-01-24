#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=siro_high
#SBATCH --account=siro
#SBATCH --job-name=ai2thor_to_behavior1k
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --array=0-120

# This script launches a configurable number of concurrent python processes.
# Each process is managed by a separate function call running in the background.
# The script will re-launch a process if it terminates, until a file named
# "[ID].success" is found, where [ID] is the integer argument for that process.

# --- Configuration ---
# The number of concurrent jobs to run per array task.
# This value is taken from the first command-line argument ($1).
# If no argument is provided, it defaults to 8. ⚙️
NUM_JOBS=${1:-2}
TOTAL_JOBS_IN_ARRAY=$((NUM_JOBS * SLURM_ARRAY_TASK_COUNT))

# --- Sanity Check ---
# Ensure the SLURM_ARRAY_TASK_ID environment variable is set.
if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
  echo "Error: SLURM_ARRAY_TASK_ID is not set. Exiting." >&2
  exit 1
fi

# --- Process Management Function ---
# This function handles the lifecycle of a single process.
manage_process() {
  # The unique integer ID for the process.
  local process_id=$1
  # The name of the file that signals successful completion.
  local success_file="/checkpoint/clear/cgokmen/behavior-data2/ai2thor/jobs/${process_id}.success"
  local log_file="logs/ai2thor_${process_id}.log"

  # If the success file already exists, remove the success file to start fresh.
  if [ -f "${success_file}" ]; then
    rm -f "${success_file}"
  fi

  if [ -f "${log_file}" ]; then
    rm -f "${log_file}"
  fi

  # Loop indefinitely until the success file exists.
  while [ ! -f "${success_file}" ]; do
    echo "[$(date)] Launching process for ID: ${process_id} ${TOTAL_JOBS_IN_ARRAY}"
    
    # Execute the python script with the calculated ID as an argument.
    python -u -m omnigibson.examples.scenes.import_habitat_scenes ai2thor '/checkpoint/clear/cgokmen/procthor/ai2thor/ai2thor-hab/configs/scenes/**/*.scene_instance.json' "${process_id}" "${TOTAL_JOBS_IN_ARRAY}" >> ${log_file} 2>&1
  done
  
  echo "[$(date)] Success file found for ID: ${process_id}. Process complete."
}

# --- Main Execution ---
echo "Starting process manager for SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID} with ${NUM_JOBS} jobs"

# Loop from 0 to (NUM_JOBS - 1) to launch the processes.
for (( i=0; i<NUM_JOBS; i++ )); do
  # Calculate the unique ID for this specific task based on the SLURM variable and the loop index.
  # The formula is now: (SLURM_ARRAY_TASK_ID * NUM_JOBS) + i
  task_id=$((SLURM_ARRAY_TASK_ID * NUM_JOBS + i))
  
  # Launch the manage_process function in the background (&) for concurrency.
  # This allows all jobs to be managed simultaneously.
  manage_process "${task_id}" &
done

# The 'wait' command blocks the script from exiting until all background jobs have completed.
echo "Waiting for all ${NUM_JOBS} background processes to complete..."
wait
echo "All processes finished successfully. Exiting."