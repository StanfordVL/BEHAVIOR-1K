#!/bin/bash
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=h200_core_shared
#SBATCH --account=clear
#SBATCH --job-name=policy-pi05ddp
#SBATCH --output=logs-bigrun2/policy-pi05ddp-%A_%a.log
#SBATCH --error=logs-bigrun2/policy-pi05ddp-%A_%a.log
#SBATCH --array=0-3

# Dataset combinations
COMBINATIONS=("bpv" "bp" "bv" "b")
DATASET=${COMBINATIONS[$SLURM_ARRAY_TASK_ID]}

accelerate launch $(which lerobot-train) \
  --dataset.repo_id=${DATASET} \
  --dataset.root=/checkpoint/clear/cgokmen/merged_lerobot_datasets_2/${DATASET} \
  --dataset.video_backend=pyav \
  --output_dir=/checkpoint/clear/cgokmen/policies-bigrun2/pi05ddp-${DATASET}-${SLURM_JOB_ID} \
  --job_name=pi05ddp-${DATASET}-${SLURM_JOB_ID} \
  --policy.type=pi05 \
  --policy.repo_id=pi05ddp-${DATASET}-${SLURM_JOB_ID} \
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
  --wandb.project=vid2room-policies-bigrun2 \
  --batch_size=32 \
  --num_workers=8
