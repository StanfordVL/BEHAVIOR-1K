#!/bin/bash
#SBATCH --job-name="replay_data"
#SBATCH --account=vision
#SBATCH --partition=svl
#SBATCH --nodes=1
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/sc/replay_data_%j.out
#SBATCH --error=outputs/sc/replay_data_%j.err

# list out some useful information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NAME="$SLURM_JOB_NAME
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

source /vision/u/wsai/miniconda3/bin/activate behavior

echo "File to process: $1"

echo "Running replay_obs.py on $1"
OMNIGIBSON_HEADLESS=1 python OmniGibson/omnigibson/learning/scripts/replay_obs.py --file $1 --low_dim --rgbd --seg --bbox --pcd_gt --pcd_vid

echo "Job finished."
exit 0
