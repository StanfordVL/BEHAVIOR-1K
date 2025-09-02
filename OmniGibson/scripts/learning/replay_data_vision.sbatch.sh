#!/bin/bash
#SBATCH --job-name="replay_data"
#SBATCH --account=cvgl
#SBATCH --partition=svl,napoli-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/sc/replay_data/%j.out
#SBATCH --error=outputs/sc/replay_data/%j.err

# list out some useful information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NAME="$SLURM_JOB_NAME
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

source /vision/u/$(whoami)/miniconda3/bin/activate behavior

echo "Current time: $(date)"
echo "Running with args: $@"


ERR_FILE="/vision/u/$(whoami)/BEHAVIOR-1K/outputs/sc/replay_data/${SLURM_JOB_ID}.err"

# Start watchdog in background
(
  prev_lines=-1
  while true; do
    if [[ -f "$ERR_FILE" ]]; then
      current_lines=$(wc -l < "$ERR_FILE")
      if [[ "$current_lines" -le "$prev_lines" ]]; then
        echo "No new output detected in $ERR_FILE. Killing job $SLURM_JOB_ID."
        /usr/local/bin/scancel $SLURM_JOB_ID
        exit 0
      else
        prev_lines=$current_lines
      fi
    else
      echo "$ERR_FILE does not exist yet."
    fi
    sleep 3600  # wait 1h
  done
) &

OMNIGIBSON_HEADLESS=1 python OmniGibson/omnigibson/learning/scripts/replay_obs.py $@ --low_dim --rgbd --seg

echo "Job finished."
exit 0
