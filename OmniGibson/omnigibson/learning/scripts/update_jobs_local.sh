#!/bin/bash

# source conda
source ~/miniconda3/etc/profile.d/conda.sh
LOG_DIR="~/Documents/logs"
mkdir -p $LOG_DIR

TIMESTAMP=$(date +%Y-%m-%d_%H-%M)
LOG_FILE="$LOG_DIR/update_jobs_$TIMESTAMP.log"

cd ~/Research/BEHAVIOR-1K
python OmniGibson/omnigibson/learning/scripts/update_jobs.py --local > "$LOG_FILE" 2>&1