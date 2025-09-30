#!/bin/bash
#SBATCH --job-name="update_jobs"
#SBATCH --account=vision
#SBATCH --partition=svl
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=12
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/sc/update_jobs/%j.out
#SBATCH --error=outputs/sc/update_jobs/%j.err

# list out some useful information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NAME="$SLURM_JOB_NAME
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

source /vision/u/$(whoami)/miniconda3/bin/activate behavior

echo "Current time: $(date)"
echo "Running with args: $@"

OMNIGIBSON_HEADLESS=1 python OmniGibson/omnigibson/learning/scripts/update_jobs.py $@

echo "Job finished."
exit 0
