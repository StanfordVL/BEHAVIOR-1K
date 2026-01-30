#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=cpu_lowest
#SBATCH --account=clear
#SBATCH --job-name=merge_rollout_datasets
#SBATCH --output=/home/cgokmen/projects/BEHAVIOR-1K/slurm/logs/merge_rollout_datasets-%A.log
#SBATCH --error=/home/cgokmen/projects/BEHAVIOR-1K/slurm/logs/merge_rollout_datasets-%A.log

python -u merge_rollout_datasets.py ${1}
