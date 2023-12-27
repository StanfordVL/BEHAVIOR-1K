#!/usr/bin/env bash

# Get the user's currently running vscode job count
USERNAME=$(whoami)
CURRENTLY_RUNNING_JOBS=$(squeue -u $USERNAME -o "%j:%i" | grep omnigibson-vscode)
CURRENTLY_RUNNING_JOBS_COUNT=$(echo -n $CURRENTLY_RUNNING_JOBS | wc -l)

if [ "$CURRENTLY_RUNNING_JOBS_COUNT" -gt 0 ]; then
    CURRENT_JOB_IDS=$(echo -n "$CURRENTLY_RUNNING_JOBS" | sed "s/.*://g" | tr '\n' ',') 
    echo "You already have omnigibson-vscode running. First cancel those jobs by running: scancel ${CURRENT_JOB_IDS}"
    exit 1
fi

# Queue a new job for the user
sbatch /cvgl/group/Gibson/og-docker/launch_vscode.sh

# Wait for the file to show up
JOBS_AFTER_LAUNCH=$(squeue -u $USERNAME -o "%j:%i" | grep omnigibson-vscode)
JOBS_AFTER_LAUNCH_COUNT=$(echo -n $JOBS_AFTER_LAUNCH | wc -l)

if [ "$JOBS_AFTER_LAUNCH_COUNT" -eq 1 ]; then
    echo "Job could not be queued. Please contact the OmniGibson team."
    exit 1
fi

# Get the job id
LAUNCHED_JOB_ID=$(echo -n "$JOBS_AFTER_LAUNCH" | sed "s/.*://g" | tr '\n' '')

# Check that the output file exists
OUTPUT_FILE="~/slurm-${LAUNCHED_JOB_ID}.out"
while [ ! -f "$OUTPUT_FILE" ]; do
    echo "Waiting for the job to launch."
    sleep 3
done

# Use tail to output the first 3 lines of the file that contain the string OMNIGIBSON-VSCODE
tail -f "$OUTPUT_FILE" | grep -m 3 "OMNIGIBSON-VSCODE"