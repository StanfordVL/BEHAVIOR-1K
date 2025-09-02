#!/bin/bash
#SBATCH --job-name="replay_data"
#SBATCH --account=cvgl
#SBATCH --partition=svl,napoli-gpu
#SBATCH --exclude=svl4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=480G
#SBATCH --cpus-per-task=72
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/sc/upload_to_hf_%j.out
#SBATCH --error=outputs/sc/upload_to_hf_%j.err

# list out some useful information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NAME="$SLURM_JOB_NAME
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

source /vision/u/$(whoami)/miniconda3/bin/activate behavior

python OmniGibson/omnigibson/learning/scripts/upload_to_hf.py $@

echo "Job finished."
exit 0
