```
conda activate lerobot
```

lerobot-train \
    --policy.type=diffusion_policy \
    --policy.dim_model=64 \
    --policy.n_action_steps=20 \
    --policy.chunk_size=20 \
    --policy.device="cuda" \
    --policy.push_to_hub=false \
    --dataset.repo_id=stretch_vid2scene_simple \
    --dataset.root="/home/yalcintr/workspace/vid2scene_policy/my_datasets/stretch_vid2scene_simple" \
    --dataset.image_transforms.enable=true \
    --batch_size=1 \
    --steps=20000 \
    --eval_freq=0 \
    --save_freq=20000 \
    --save_checkpoint=true \
    --log_freq=10 \
    --wandb.enable=false \
    --output_dir=outputs/act/test 