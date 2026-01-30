#!/bin/bash
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=h200_core_shared
#SBATCH --account=clear
#SBATCH --job-name=policy-dp
#SBATCH --output=logs-bigrun2/policy-dp-%A_%a.log
#SBATCH --error=logs-bigrun2/policy-dp-%A_%a.log
#SBATCH --array=0-3

# Dataset combinations
COMBINATIONS=("bpv" "bp" "bv" "b")
DATASET=${COMBINATIONS[$SLURM_ARRAY_TASK_ID]}

lerobot-train \
  --dataset.repo_id=${DATASET} \
  --dataset.root=/checkpoint/clear/cgokmen/merged_lerobot_datasets_2/${DATASET} \
  --dataset.video_backend=pyav \
  --output_dir=/checkpoint/clear/cgokmen/policies-bigrun2/dp-${DATASET}-${SLURM_JOB_ID} \
  --job_name=dp-${DATASET}-${SLURM_JOB_ID} \
  --policy.type=diffusion \
  --policy.repo_id=dp-${DATASET}-${SLURM_JOB_ID} \
  --policy.use_amp=true \
  --policy.push_to_hub=false \
  --save_freq=1000 \
  --steps=1000000 \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=vid2room-policies-bigrun2 \
  --batch_size=64 \
  --num_workers=64
