#!/bin/bash
set -euo pipefail

# Arguments
data_dir="$1"
task_id="$2"
start_idx="$3"   # 1-based index of first demo to run
end_idx="$4"     # 1-based index of last demo to run

# Zero-pad task_id to 4 digits
task_id_padded=$(printf "%04d" "$task_id")

# Directory containing parquet files
task_dir="${data_dir}/2025-challenge-demos/data/task-${task_id_padded}"

# Check if directory exists
if [ ! -d "$task_dir" ]; then
  echo "Error: Directory $task_dir does not exist."
  exit 1
fi

# Gather and sort files
files=($(ls "$task_dir"/episode_*.parquet | sort))

# Adjust indices (bash arrays are 0-based)
start=$((start_idx - 1))
end=$((end_idx - 1))

# Sanity checks
if [ "$start" -lt 0 ] || [ "$end" -ge "${#files[@]}" ] || [ "$start" -gt "$end" ]; then
  echo "Error: Invalid start_idx=$start_idx or end_idx=$end_idx (total=${#files[@]})"
  exit 1
fi

# Loop over the selected range
for i in $(seq "$start" "$end"); do
  file="${files[$i]}"
  filename=$(basename "$file")
  demo_id=$(echo "$filename" | sed -E 's/^episode_([0-9]{8})\.parquet$/\1/')

  echo "[$((i+1))/${#files[@]}] Running replay for demo_id=$demo_id ..."
  python OmniGibson/omnigibson/learning/scripts/replay_obs.py \
    --data_folder="$data_dir" \
    --task_name=turning_on_radio \
    --demo_id="$demo_id" \
    --pcd_vid
done
