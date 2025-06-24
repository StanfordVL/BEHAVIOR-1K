#!/bin/bash
#SBATCH --job-name="process_data"
#SBATCH --account=vision
#SBATCH --partition=svl
#SBATCH --nodes=1
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --time=0-2:00:00
#SBATCH --output=outputs/sc/process_data_%j.out
#SBATCH --error=outputs/sc/process_data_%j.err

# list out some useful information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NAME="$SLURM_JOB_NAME
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

mkdir -p /vision/u/wsai/OmniGibson/outputs/sc
source /vision/u/wsai/miniconda3/bin/activate omnigibson

echo "File to process: $1/raw/$2"

echo "Running replay_obs.py on $1/raw/$2"
OMNIGIBSON_HEADLESS=1 python omnigibson/learning/scripts/replay_obs.py --files "$1/raw/$2"

echo "Replay observation script finished! Now converting rgbd to pcd."
OMNIGIBSON_HEADLESS=1 python omnigibson/learning/scripts/rgbd_to_pcd.py -o --files "$1/rgbd/$2"

echo "Job finished."
exit 0
