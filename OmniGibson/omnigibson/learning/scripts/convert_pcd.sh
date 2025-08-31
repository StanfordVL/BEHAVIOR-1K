#!/bin/bash
set -euo pipefail

source /vision/u/$(whoami)/miniconda3/bin/activate behavior

# Arguments
data_dir="$1"
task_id="$2"

# Zero-pad task_id to 4 digits
task_id_padded=$(printf "%04d" "$task_id")

# Directory containing parquet files
task_dir="${data_dir}/2025-challenge-demos/data/task-${task_id_padded}"

# Check if directory exists
if [ ! -d "$task_dir" ]; then
  echo "Error: Directory $task_dir does not exist."
  exit 1
fi

# Loop over parquet files
for file in "$task_dir"/episode_*.parquet; do
  # Extract demo_id from filename (last 8 digits before .parquet)
  filename=$(basename "$file")
  demo_id=$(echo "$filename" | sed -E 's/^episode_([0-9]{8})\.parquet$/\1/')

  # Skip if demo_id extraction failed
  if [ -z "$demo_id" ]; then
    echo "Warning: Could not parse demo_id from $filename"
    continue
  fi

  echo "Running replay for demo_id=$demo_id ..."
  python OmniGibson/omnigibson/learning/scripts/replay_obs.py \
    --data_folder="$data_dir" \
    --task_name=turning_on_radio \
    --demo_id="$demo_id" \
    --pcd_vid
done
