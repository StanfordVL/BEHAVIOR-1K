#!/bin/bash
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=h200_core_shared
#SBATCH --account=clear
#SBATCH --job-name=policy-customdp-spoc
#SBATCH --output=logs/policy-customdp-spoc-%A.log
#SBATCH --error=logs/policy-customdp-spoc-%A.log

echo "Copying dataset to /tmp/spoc-train"
cp -R /checkpoint/clear/cgokmen/merged_lerobot_datasets/spoc-train /tmp/spoc-train
echo "Dataset copied to /tmp/spoc-train"

echo "Copying dataset to /tmp/spoc-val"
cp -R /checkpoint/clear/cgokmen/merged_lerobot_datasets/spoc-val /tmp/spoc-val
echo "Dataset copied to /tmp/spoc-val"

python /home/cgokmen/projects/BEHAVIOR-1K/vid2room_policy/vid2scene_policy/policy_training/train_lerobot_diffusion.py \
  --train_dataset_path /tmp/spoc-train \
  --val_dataset_path /tmp/spoc-val \
  --output_dir /checkpoint/clear/cgokmen/policies/customdp-spoc-${SLURM_JOB_ID} \
  --run_name customdp-spoc-${SLURM_JOB_ID} \
  --batch_size 128 \
  --num_workers 64