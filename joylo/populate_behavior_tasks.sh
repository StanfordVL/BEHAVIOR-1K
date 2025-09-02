#!/bin/bash

# Get the dataset path
DATASET_PATH=$(python3 -c "from omnigibson.utils.asset_utils import get_dataset_path; print(get_dataset_path('behavior-1k-assets'))")
echo "Using dataset path: $DATASET_PATH"

SRC_DIR="./sampled_task"

# Set FORCE to true by default
FORCE=true

# Allow disabling force with -n flag
while getopts "n" opt; do
  case $opt in
    n) FORCE=false ;;
    *) echo "Usage: $0 [-n]" >&2
       echo "  -n  Do not overwrite existing files (by default, will overwrite)" >&2
       exit 1 ;;
  esac
done

# Find all task directories within the source directory
for task_dir in "$SRC_DIR"/*/; do
    # Extract task name from directory path
    task_name=$(basename "$task_dir")
    
    # Skip if not a directory
    if [ ! -d "$task_dir" ]; then
        continue
    fi
    
    echo "Processing task directory: $task_name"
    
    # Process each JSON template file in the task directory
    for file in "$task_dir"/*; do
        # Skip if no files match the pattern
        [ -e "$file" ] || continue

        filename=$(basename "$file")

        # Extract scene model from filename (everything before "_task")
        scene_model=$(echo "$filename" | sed -E 's/(.*)_task.*/\1/')
        echo "scene model: ${scene_model}"

        if [ -d "$file" ]; then
            flag="-d"
        elif [ -f "$file" ]; then
            flag="-f"
        else
            echo "$file is a special file, skipping"
            continue
        fi
        
        dest_dir="${DATASET_PATH}/scenes/${scene_model}/json"
        if [ ! -d "$dest_dir" ]; then
            echo "ERROR: Destination directory $dest_dir does not exist for file $filename. Terminating."
            exit 1
        fi
        
        # Check if file already exists in destination
        if [ "$flag" "$dest_dir/$filename" ] && [ "$FORCE" = false ]; then
            echo "INFO: $filename already exists in $dest_dir/ - skipping"
        else
            # Now the default is to overwrite
            if [ "$flag" "$dest_dir/$filename" ]; then
                echo "Overwriting $filename in $dest_dir/"
            else
                echo "Copying $filename to $dest_dir/"
            fi
            cp -r "$file" "$dest_dir/"
        fi
    done
done

echo "All JSON files have been copied to their corresponding directories."