#!/bin/bash
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=h200_core_shared
#SBATCH --account=clear
#SBATCH --job-name=policy-act-spoc
#SBATCH --output=logs/policy-act-spoc-%A.log
#SBATCH --error=logs/policy-act-spoc-%A.log

accelerate launch $(which lerobot-train) \
  --dataset.repo_id=spoc-train \
  --dataset.root=/checkpoint/clear/cgokmen/merged_lerobot_datasets/spoc-train \
  --output_dir=/checkpoint/clear/cgokmen/policies/act-spoc \
  --job_name=act-spoc \
  --policy.type=act \
  --policy.repo_id=act-spoc \
  --policy.use_amp=true \
  --policy.push_to_hub=false \
  --save_freq=1000 \
  --steps=100000 \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=vid2room-policies \
  --batch_size=512