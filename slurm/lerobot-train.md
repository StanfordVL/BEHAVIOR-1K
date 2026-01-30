srun \
--cpus-per-task=32 \
--gpus-per-task=1 \
--time=7-00:00:00 \
--nodes=1 \
--ntasks-per-node=1 \
--qos=h200_core_shared \
--account=clear \
--job-name=lerobot-train \
--output=logs/lerobot-train-%A.log \
--error=logs/lerobot-train-%A.log \
accelerate launch $(which lerobot-train) \
  --dataset.repo_id=spoc-train \
  --dataset.root=/checkpoint/clear/cgokmen/merged_lerobot_datasets/spoc-train \
  --output_dir=./policies/pi05-spoc \
  --job_name=pi05-spoc \
  --policy.type=pi05 \
  --policy.repo_id=pi05-spoc \
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
  --batch_size=48

lerobot-train \
  --dataset.repo_id=spoc-train \
  --dataset.root=/checkpoint/clear/cgokmen/merged_lerobot_datasets/spoc-train \
  --output_dir=./policies/pi05-spoc \
  --job_name=pi05-spoc \
  --policy.type=pi05 \
  --policy.repo_id=pi05-spoc \
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
  --batch_size=48


  lerobot-train \
  --dataset.repo_id=spoc-train \
  --dataset.root=/checkpoint/clear/cgokmen/merged_lerobot_datasets/spoc-train \
  --output_dir=./policies/dp-spoc \
  --job_name=dp-spoc \
  --policy.type=diffusion \
  --policy.repo_id=diffusion-spoc \
  --policy.use_amp=true \
  --policy.push_to_hub=false \
  --save_freq=1000 \
  --steps=100000 \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=vid2room-policies \
  --batch_size=256