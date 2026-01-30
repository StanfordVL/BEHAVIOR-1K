#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=cpu_lowest
#SBATCH --account=clear
#SBATCH --job-name=zip-dirs
#SBATCH --output=logs/zip-dirs-%A_%a.log
#SBATCH --error=logs/zip-dirs-%A_%a.log
#SBATCH --array=0-1

set -e

# Path to file containing directories to zip (one per line)
# Pass as first argument, or use default
DIRS_FILE="${1:-directories_to_zip.txt}"

if [[ ! -f "$DIRS_FILE" ]]; then
    echo "Error: Directory list file not found: $DIRS_FILE"
    exit 1
fi

# Read directories into array (skip empty lines and comments)
mapfile -t DIRECTORIES < <(grep -v '^\s*#' "$DIRS_FILE" | grep -v '^\s*$')

# Get the directory for this array task
if [[ -z "${SLURM_ARRAY_TASK_ID}" ]]; then
    echo "Error: This script must be run as a SLURM array job"
    exit 1
fi

if [[ $SLURM_ARRAY_TASK_ID -ge ${#DIRECTORIES[@]} ]]; then
    echo "Error: Array task ID $SLURM_ARRAY_TASK_ID exceeds number of directories (${#DIRECTORIES[@]})"
    exit 1
fi

TARGET_DIR="${DIRECTORIES[$SLURM_ARRAY_TASK_ID]}"

if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Error: Target directory does not exist: $TARGET_DIR"
    exit 1
fi

# Create output path (same location as directory, with .tar.gz extension)
OUTPUT_FILE="${TARGET_DIR}.tar"

echo "=== Compressing Directory ==="
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Source: $TARGET_DIR"
echo "Output: $OUTPUT_FILE"
echo "Excluding: .success files"
echo ""

# Use -C to change into target dir so archive contains just the contents
# --exclude must come BEFORE the source directory
# Exclude any file ending in .success (e.g., task.success, job.success, etc.)
tar -C "$TARGET_DIR" --exclude='*.success' -cvf "$OUTPUT_FILE" .

echo ""
echo "=== Compression Complete ==="
echo "Output file: $OUTPUT_FILE"
ls -lh "$OUTPUT_FILE"

