#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=cpu_lowest
#SBATCH --account=clear
#SBATCH --job-name=archive-datasets
#SBATCH --output=logs/archive-datasets-%A_%a.log
#SBATCH --error=logs/archive-datasets-%A_%a.log
#SBATCH --array=0-399

#==============================================================================
# Configuration
#==============================================================================
# Array layout:
#   0-99:   spoc scenes (100 shards)
#   100-199: spoc objects (100 shards)
#   200-299: vid2room scenes (100 shards)
#   300-399: vid2room objects (100 shards)

SCRIPT_DIR="/home/cgokmen/projects/BEHAVIOR-1K/slurm"
OUTPUT_BASE="/checkpoint/clear/cgokmen/behavior-data2-compressed"
DATASET_BASE="/checkpoint/clear/cgokmen/behavior-data2"

SHARDS_PER_TYPE=100

#==============================================================================
# Script Logic
#==============================================================================
set -e

TASK_ID=${SLURM_ARRAY_TASK_ID}

# Determine which dataset type and shard based on task ID
if [[ $TASK_ID -lt 100 ]]; then
    # spoc scenes: tasks 0-99
    LIST_FILE="${SCRIPT_DIR}/scene_files_spoc.txt"
    OUTPUT_DIR="${OUTPUT_BASE}/spoc"
    DATASET_ROOT="${DATASET_BASE}/spoc"
    SHARD_ID=$TASK_ID
    TYPE_NAME="spoc_scenes"
elif [[ $TASK_ID -lt 200 ]]; then
    # spoc objects: tasks 100-199
    LIST_FILE="${SCRIPT_DIR}/object_files_spoc.txt"
    OUTPUT_DIR="${OUTPUT_BASE}/spoc"
    DATASET_ROOT="${DATASET_BASE}/spoc"
    SHARD_ID=$((TASK_ID - 100))
    TYPE_NAME="spoc_objects"
elif [[ $TASK_ID -lt 300 ]]; then
    # vid2room scenes: tasks 200-299
    LIST_FILE="${SCRIPT_DIR}/scene_files_vid2room.txt"
    OUTPUT_DIR="${OUTPUT_BASE}/vid2room"
    DATASET_ROOT="${DATASET_BASE}/vid2room"
    SHARD_ID=$((TASK_ID - 200))
    TYPE_NAME="vid2room_scenes"
else
    # vid2room objects: tasks 300-399
    LIST_FILE="${SCRIPT_DIR}/object_files_vid2room.txt"
    OUTPUT_DIR="${OUTPUT_BASE}/vid2room"
    DATASET_ROOT="${DATASET_BASE}/vid2room"
    SHARD_ID=$((TASK_ID - 300))
    TYPE_NAME="vid2room_objects"
fi

echo "=============================================="
echo "Task ID: ${TASK_ID}"
echo "Type: ${TYPE_NAME}"
echo "Shard: ${SHARD_ID}/${SHARDS_PER_TYPE}"
echo "List file: ${LIST_FILE}"
echo "Dataset root: ${DATASET_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "=============================================="

if [[ ! -f "$LIST_FILE" ]]; then
    echo "Error: List file not found: $LIST_FILE"
    exit 1
fi

python "${SCRIPT_DIR}/archive_datasets.py" "${LIST_FILE}" "${OUTPUT_DIR}" "${DATASET_ROOT}" "${SHARD_ID}" "${SHARDS_PER_TYPE}" "${TYPE_NAME}"

# Compress the tar file with pigz (parallel gzip)
TAR_FILE="${OUTPUT_DIR}/${TYPE_NAME}_shard_$(printf '%03d' ${SHARD_ID}).tar"
if [[ -f "$TAR_FILE" ]]; then
    echo "Compressing ${TAR_FILE} with pigz..."
    pigz -p 8 "$TAR_FILE"
    echo "Compressed to ${TAR_FILE}.gz"
else
    echo "Warning: Expected tar file not found: ${TAR_FILE}"
fi

echo "Task ${TASK_ID} (${TYPE_NAME} shard ${SHARD_ID}) completed."

