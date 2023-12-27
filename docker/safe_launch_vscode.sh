#!/usr/bin/env bash

# Get the user's currently running vscode job count
CURRENTLY_RUNNING_JOBS = $(squeue -u cgokmen -o "%j:%i" | grep vscode)  # TODO: omnigibson-vscode
CURRENTLY_RUNNING=$(squeue -u cgokmen -o "%j" | grep vscode | wc -l)

if [ "$CURRENTLY_RUNNING" -gt 0 ]; then
    echo "You already have a job running."
    exit 1
fi