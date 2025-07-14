#!/bin/bash
#SBATCH --job-name="train_policy"
#SBATCH --account=viscam
#SBATCH --partition=viscam
#SBATCH --exclude=viscam1
#SBATCH --nodes=1
#SBATCH --gres=gpu:a5000:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=240G
#SBATCH --cpus-per-task=12
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/sc/train_policy_%j.out
#SBATCH --error=outputs/sc/train_policy_%j.err
# notifications for job done & fail
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=wsai@stanford.edu

# list out some useful information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NAME="$SLURM_JOB_NAME
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

source /vision/u/wsai/miniconda3/bin/activate behavior

OMNIGIBSON_HEADLESS=1 python OmniGibson/omnigibson/learning/train.py policy=wbvima task=picking_up_trash data_dir=/vision/u/wsai/behavior/picking_up_trash

echo "Job finished."
exit 0
