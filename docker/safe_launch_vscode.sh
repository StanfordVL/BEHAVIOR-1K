#!/usr/bin/env bash
set -e -o pipefail

# Get the user's currently running vscode job count
USERNAME=$(whoami)
CURRENTLY_RUNNING_JOBS=$(squeue -u $USERNAME -o "%j:%i" | grep omnigibson-vscode || true)

if [ -z "$CURRENTLY_RUNNING_JOBS" ]; then
    # Queue a new job for the user
    echo "Starting new job"
    sbatch /cvgl/group/Gibson/og-docker/launch_vscode.sh

    # Wait for the file to show up
    while ! (squeue -u $USERNAME -o "%j:%i" | grep -q omnigibson-vscode); do
        echo "Waiting for the job to launch."
        sleep 3
    done
fi

# Get the job id
LAUNCHED_JOB_ID=$(squeue -u $USERNAME -o "%j:%i" | grep -m 1 omnigibson-vscode | sed "s/.*://g" | tr -d '\n')
echo "Job ID: $LAUNCHED_JOB_ID"

# Check that the output file exists
OUTPUT_FILE="slurm-${LAUNCHED_JOB_ID}.out"
while [ ! -f "$OUTPUT_FILE" ]; do
    echo "Waiting for the job to start outputting."
    sleep 3
done

# Wait for the output file to contain the string OMNIGIBSON-VSCODE exactly 3 times
while [ "$(grep -c "OMNIGIBSON-VSCODE" "$OUTPUT_FILE" || true)" -lt 3 ]; do
    echo "Waiting for the job to allocate ports."
    sleep 3
done
# echo "Ports allocated successfully."

# Wait for the output file to contain the string "HTTP server listening"
# while ! grep -q "HTTP server listening" "$OUTPUT_FILE"; do
#     echo "Waiting for the job to start the HTTP server."
#     sleep 3
# done
# echo "HTTP server started successfully."
# echo ""

# Echo the OMNIGIBSON-VSCODE lines
grep "OMNIGIBSON-VSCODE" "$OUTPUT_FILE"