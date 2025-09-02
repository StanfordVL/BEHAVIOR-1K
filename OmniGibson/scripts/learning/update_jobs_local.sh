#!/bin/bash

# === CONFIG ===
MAX_AGE=$((20 * 60))        # 20 minutes in seconds for watchdog
CHECK_INTERVAL=$((5 * 60)) # check every 5 minutes
LOG_DIR="$HOME/Documents/logs/update_jobs"
SCRIPT_DIR="$HOME/Research/BEHAVIOR-1K"
PYTHON_SCRIPT="OmniGibson/omnigibson/learning/scripts/update_jobs.py"
CONDA_ENV="behavior"

# === SETUP ===
# Source conda
source "$HOME/miniconda3/bin/activate" "$CONDA_ENV"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Combine script + arguments for pgrep
PATTERN="$PYTHON_SCRIPT $*"

# Infinite loop to keep checking/launching the job
while true; do
    # Check if a job is already running
    JOB_PID=$(pgrep -f "$PATTERN")

    if [ -z "$JOB_PID" ]; then
        # No job running, launch a new one
    TIMESTAMP=$(date +%Y-%m-%d_%H-%M)
    PATTERN_LAST_CHAR="${PATTERN: -1}"
    LOG_FILE="$LOG_DIR/update_jobs_${TIMESTAMP}_${PATTERN_LAST_CHAR}.log"
        cd "$SCRIPT_DIR" || exit 1

        python "$PYTHON_SCRIPT" "$@" --local > "$LOG_FILE" 2>&1 &
        JOB_PID=$!
        echo "$(date): Launched new job with PID $JOB_PID, logging to $LOG_FILE"
    else
        # Job is running, find its log file (most recent)
        LOG_FILE=$(ls -t "$LOG_DIR"/update_jobs_*.log 2>/dev/null | head -n1)

        if [ -f "$LOG_FILE" ]; then
            last_mod=$(stat -c %Y "$LOG_FILE")
            now=$(date +%s)
            age=$((now - last_mod))

            if [ "$age" -gt "$MAX_AGE" ]; then
                echo "$(date): Log file stale ($age s), killing PID $JOB_PID" >> "$LOG_FILE"
                kill "$JOB_PID"
                # Launch a new job immediately after killing
                TIMESTAMP=$(date +%Y-%m-%d_%H-%M)
                PATTERN_LAST_CHAR="${PATTERN: -1}"
                LOG_FILE="$LOG_DIR/update_jobs_${TIMESTAMP}_${PATTERN_LAST_CHAR}.log"
                cd "$SCRIPT_DIR" || exit 1
                python "$PYTHON_SCRIPT" "$@" --local > "$LOG_FILE" 2>&1 &
                JOB_PID=$!
                echo "$(date): Relaunched new job with PID $JOB_PID, logging to $LOG_FILE"
            else
                echo "$(date): Job $JOB_PID running, log updated $age seconds ago"
            fi
        fi
    fi

    # Sleep until next check
    sleep "$CHECK_INTERVAL"
done