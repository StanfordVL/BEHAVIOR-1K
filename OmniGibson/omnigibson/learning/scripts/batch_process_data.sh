#!/bin/bash

BEHAVIOR_DIR="/vision/u/wsai/BEHAVIOR-1K"

# batch process data for a giving task
# get the task name from the first argument
if [ -z "$1" ]; then
    echo "Usage: $0 <task_name>"
    exit 1
fi
task_name=$1
echo "Processing task: $task_name"
# make sure the task is what the user wants with a confirmation
read -p "Are you sure you want to process the task '$task_name'? (y/n) " confirm
if [[ $confirm != "y" ]]; then
    echo "Aborting."
    exit 1
fi
base_dir=/vision/u/wsai/behavior/$task_name
# get all the files in the base directory
# make sure the base directory exists
if [ ! -d "$base_dir/raw" ]; then
    echo "Base directory $base_dir/raw does not exist."
    exit 1
fi
cd $base_dir/raw
# get all filenames that ends with .hdf5.
filenames=$(ls *.hdf5 2>/dev/null)
if [ -z "$filenames" ]; then
    echo "No .hdf5 files found in $base_dir/raw."
    exit 1
else
    echo "Found number of hdf5 files: $(echo "$filenames" | wc -l)"
fi

cd $BEHAVIOR_DIR

# only process the first 2 files for testing
# filenames=$(echo "$filenames" | head -n 2)

# loop through all the files in the directory
for file in $filenames; do
    sbatch OmniGibson/omnigibson/learning/scripts/process_data.sbatch.sh "$base_dir" "$file"
done

echo "All files submitted for processing."