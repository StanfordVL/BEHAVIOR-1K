#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=cpu_lowest
#SBATCH --account=clear
#SBATCH --job-name=regenerate_spoc_seg_maps
#SBATCH --output=/home/cgokmen/projects/BEHAVIOR-1K/slurm/logs/regenerate_spoc_seg_maps-%A_%a.log
#SBATCH --error=/home/cgokmen/projects/BEHAVIOR-1K/slurm/logs/regenerate_spoc_seg_maps-%A_%a.log
#SBATCH --array=0-1023

#==============================================================================
# Script Logic
#==============================================================================
# This script uses a fixed number of array jobs (128) to process an
# arbitrary number of videos. Each job calculates which videos it is
# responsible for using the modulo operator.

# Get the total number of jobs in the array from the Slurm environment variable.
# This will be 128 based on the #SBATCH directive above.
TOTAL_JOBS=${SLURM_ARRAY_TASK_COUNT}

# Get the current task's ID (1-based) and calculate its 0-based index.
TASK_ID=${SLURM_ARRAY_TASK_ID}

echo "Task ${TASK_ID}/${TOTAL_JOBS} starting."

python -u regenerate_spoc_seg_maps.py ${TASK_ID} ${TOTAL_JOBS}

echo "Task ${TASK_ID} has finished regenerating segmentation maps and completed its work."
