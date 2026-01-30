#!/bin/bash
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=h200_core_shared
#SBATCH --account=clear
#SBATCH --job-name=policy-customdpddp
#SBATCH --output=logs-bigrun2/policy-customdpddp-%A_%a.log
#SBATCH --error=logs-bigrun2/policy-customdpddp-%A_%a.log
#SBATCH --array=0-3

# Dataset combinations
COMBINATIONS=("bpv" "bp" "bv" "b")
DATASET=${COMBINATIONS[$SLURM_ARRAY_TASK_ID]}

# Get number of GPUs for accelerate
NUM_GPUS=${SLURM_GPUS_PER_TASK:-1}

# Multi-GPU DDP mode using accelerate
# Effective batch size = batch_size * NUM_GPUS = 64 * 8 = 512
accelerate launch \
  /home/cgokmen/projects/BEHAVIOR-1K/vid2room_policy/vid2scene_policy/policy_training/train_lerobot_diffusion.py \
  --train_dataset_path /checkpoint/clear/cgokmen/merged_lerobot_datasets_2/${DATASET} \
  --val_dataset_path /checkpoint/clear/cgokmen/merged_lerobot_datasets/spoc-val \
  --output_dir /checkpoint/clear/cgokmen/policies-bigrun2/customdpddp-${DATASET}-${SLURM_JOB_ID} \
  --run_name customdpddp-${DATASET}-${SLURM_JOB_ID} \
  --batch_size 64 \
  --num_workers 64 \
  --steps=100000

