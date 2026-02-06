#!/bin/bash
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=h200_core_shared
#SBATCH --account=clear
#SBATCH --job-name=policy-customdp
#SBATCH --output=logs-bigrun2/policy-customdp-%A_%a.log
#SBATCH --error=logs-bigrun2/policy-customdp-%A_%a.log
#SBATCH --array=0-3

# Dataset combinations
COMBINATIONS=("bpv" "bp" "bv" "b")
DATASET=${COMBINATIONS[$SLURM_ARRAY_TASK_ID]}

# Single-GPU mode (default) - works the same as before
python /home/cgokmen/projects/BEHAVIOR-1K/vid2room_policy/vid2scene_policy/policy_training/train_lerobot_diffusion.py \
  --train_dataset_path /checkpoint/clear/cgokmen/merged_lerobot_datasets_2/${DATASET} \
  --val_dataset_path /checkpoint/clear/cgokmen/merged_lerobot_datasets/spoc-val \
  --output_dir /checkpoint/clear/cgokmen/policies-bigrun2/customdp-${DATASET}-${SLURM_JOB_ID} \
  --run_name customdp-${DATASET}-${SLURM_JOB_ID} \
  --batch_size 64 \
  --num_workers 64 \
  --steps=1000000
