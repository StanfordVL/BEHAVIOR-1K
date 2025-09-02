#!/bin/bash
USER_DIR="/vision/u/$(whoami)/BEHAVIOR-1K"
SCRIPT="$USER_DIR/OmniGibson/omnigibson/learning/scripts/update_jobs.sbatch.sh"

mkdir -p /vision/u/$(whoami)/BEHAVIOR-1K/outputs/sc/replay_data
mkdir -p /vision/u/$(whoami)/BEHAVIOR-1K/outputs/sc/update_jobs

jobid=$(cd "$USER_DIR" && /usr/local/bin/sbatch "$SCRIPT" "$@" | awk '{print $4}')

# Wait briefly for SLURM to assign state
sleep 120

state=$(/usr/local/bin/sacct -j "$jobid" --format=State --noheader | head -n1 | awk '{print $1}')

if [[ "$state" == "PENDING" ]]; then
    echo "Job $jobid is not running — cancelling"
    /usr/local/bin/scancel "$jobid"
else
    echo "Job $jobid started ($state)"
fi
