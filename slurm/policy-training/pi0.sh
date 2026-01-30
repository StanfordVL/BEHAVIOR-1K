#!/bin/bash
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=h200_core_shared
#SBATCH --account=clear
#SBATCH --job-name=policy-pi05-spoc
#SBATCH --output=logs/policy-pi05-spoc-%A.log
#SBATCH --error=logs/policy-pi05-spoc-%A.log

echo "Copying dataset to /tmp/spoc-train"
cp -R /checkpoint/clear/cgokmen/merged_lerobot_datasets/spoc-train /tmp/spoc-train
echo "Dataset copied to /tmp/spoc-train"

lerobot-train \
  --dataset.repo_id=spoc-train \
  --dataset.root=/tmp/spoc-train \
  --output_dir=/checkpoint/clear/cgokmen/policies/pi05-spoc-${SLURM_JOB_ID} \
  --job_name=pi05-spoc-${SLURM_JOB_ID} \
  --policy.type=pi05 \
  --policy.repo_id=pi05-spoc-${SLURM_JOB_ID} \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.push_to_hub=false \
  --save_freq=1000 \
  --steps=100000 \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=vid2room-policies \
  --batch_size=64 \
  --num_workers=64