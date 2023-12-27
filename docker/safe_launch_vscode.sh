#!/usr/bin/env bash
set -e -o pipefail

# Get the user's currently running vscode job count
USERNAME=$(whoami)
CURRENTLY_RUNNING_JOBS=$(squeue -u $USERNAME -o "%j:%i" | grep omnigibson-vscode || true)

if [ ! -z "$CURRENTLY_RUNNING_JOBS" ]; then
    CURRENT_JOB_IDS=$(echo -n "$CURRENTLY_RUNNING_JOBS" | sed "s/.*://g" | tr '\n' ',') 
    echo "You already have omnigibson-vscode running. First cancel those jobs by running: scancel ${CURRENT_JOB_IDS}"
    exit 1
fi

# Queue a new job for the user
srun \
    --job-name=omnigibson-vscode \
    --account=cvgl \
    --partition=svl-interactive --qos=normal \
    --nodes=1 \
    --cpus-per-task=8 \
    --mem=30G \
    --gres=gpu:2080ti:1 \
    /cvgl/group/Gibson/og-docker/launch_vscode.sh 2>&1 | grep -m 3 --line-buffered "OMNIGIBSON-VSCODE"

echo "You may need to wait up to 1 minute for vscode to launch. The password is your username."
