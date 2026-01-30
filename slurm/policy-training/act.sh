#!/bin/bash
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=h200_core_shared
#SBATCH --account=clear
#SBATCH --job-name=policy-act
#SBATCH --output=logs-bigrun/policy-act-%A_%a.log
#SBATCH --error=logs-bigrun/policy-act-%A_%a.log
#SBATCH --array=0-3

# Dataset combinations
COMBINATIONS=("bpv" "bp" "bv" "b")
DATASET=${COMBINATIONS[$SLURM_ARRAY_TASK_ID]}

echo "Copying dataset to /tmp/${DATASET}"
cp -R /checkpoint/clear/cgokmen/merged_lerobot_datasets_2/${DATASET} /tmp/${DATASET}
echo "Dataset copied to /tmp/${DATASET}"

lerobot-train \
  --dataset.repo_id=${DATASET} \
  --dataset.root=/tmp/${DATASET} \
  --output_dir=/checkpoint/clear/cgokmen/policies-bigrun/act-${DATASET}-${SLURM_JOB_ID} \
  --job_name=act-${DATASET}-${SLURM_JOB_ID} \
  --policy.type=act \
  --policy.repo_id=act-${DATASET}-${SLURM_JOB_ID} \
  --policy.use_amp=true \
  --policy.push_to_hub=false \
  --save_freq=1000 \
  --steps=100000 \
  --num_workers=32 \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=vid2room-policies-bigrun \
  --batch_size=128
