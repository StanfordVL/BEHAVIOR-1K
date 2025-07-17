#!/bin/bash

BEHAVIOR_DIR="/vision/u/wsai/BEHAVIOR-1K"
mkdir -p /vision/u/wsai/BEHAVIOR-1K/outputs/sc

# batch process data for a giving task
# get the task name from the first argument
if [ -z "$1" ]; then
    echo "Usage: $0 <task_name>"
    exit 1
fi
task_name=$1
echo "Processing task: $task_name"

# if the second argument is provided, process the files in the range [start_idx, end_idx]
if [ ! -z "$2" ]; then
    start_idx=$2
    end_idx=$3
fi

base_dir=/vision/u/wsai/data/behavior
# get all the files in the base directory
# make sure the base directory exists
if [ ! -d "$base_dir/raw/$task_name" ]; then
    echo "Base directory $base_dir/raw/$task_name does not exist."
    exit 1
fi
cd $base_dir/raw/$task_name
# get all filenames that ends with .hdf5.
filenames=$(ls *.hdf5 2>/dev/null)
if [ -z "$filenames" ]; then
    echo "No .hdf5 files found in $base_dir/raw."
    exit 1
else
    echo "Found number of hdf5 files: $(echo "$filenames" | wc -l)"
fi

cd $BEHAVIOR_DIR

# only process the files in the range [start_idx, end_idx]
# If end_idx is not provided, process all the files
if [ -z "$end_idx" ]; then
    end_idx=$(echo "$filenames" | wc -l)
fi
filenames=$(echo "$filenames" | tail -n +$start_idx | head -n $((end_idx - start_idx + 1)))
echo "Processing $(echo "$filenames" | wc -l) files in the range [$start_idx, $end_idx]: $filenames"


# make sure the task is what the user wants with a confirmation
read -p "Are you sure you want to replay the task '$task_name'? (y/n) " confirm
if [[ $confirm != "y" ]]; then
    echo "Aborting."
    exit 1
fi


# loop through all the files in the directory
for file in $filenames; do
    sbatch OmniGibson/omnigibson/learning/scripts/replay_data.sbatch.sh "$base_dir/raw/$task_name/$file"
done

echo "All files submitted for replay."