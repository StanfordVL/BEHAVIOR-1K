#!/bin/bash
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=h200_core_shared
#SBATCH --account=clear
#SBATCH --job-name=policy-customdp
#SBATCH --output=logs-bigrun/policy-customdp-%A_%a.log
#SBATCH --error=logs-bigrun/policy-customdp-%A_%a.log
#SBATCH --array=0-3

# Dataset combinations
COMBINATIONS=("bpv" "bp" "bv" "b")
DATASET=${COMBINATIONS[$SLURM_ARRAY_TASK_ID]}

echo "Copying dataset to /tmp/${DATASET}"
cp -R /checkpoint/clear/cgokmen/merged_lerobot_datasets_2/${DATASET} /tmp/${DATASET}
echo "Dataset copied to /tmp/${DATASET}"

echo "Copying dataset to /tmp/spoc-val"
cp -R /checkpoint/clear/cgokmen/merged_lerobot_datasets/spoc-val /tmp/spoc-val
echo "Dataset copied to /tmp/spoc-val"

python /home/cgokmen/projects/BEHAVIOR-1K/vid2room_policy/vid2scene_policy/policy_training/train_lerobot_diffusion.py \
  --train_dataset_path /tmp/${DATASET} \
  --val_dataset_path /tmp/spoc-val \
  --output_dir /checkpoint/clear/cgokmen/policies-bigrun/customdp-${DATASET}-${SLURM_JOB_ID} \
  --run_name customdp-${DATASET}-${SLURM_JOB_ID} \
  --batch_size 128 \
  --num_workers 64
