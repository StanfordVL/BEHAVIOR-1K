# vid2room_policy

Run all commands from the repository root (`BEHAVIOR-1K`).

## Data Collection Command

Collect pick-only demonstrations into a LeRobot-format dataset.

```bash
python vid2room_policy/collect_data.py \
  --scene_dataset behavior-1k-assets \
  --scene_name Rs_int \
  --num_episodes 100 \
  --output ./lerobot_datasets \
  --repo-id fetch_pick_task_test
```

Common options:

- `--object-filter {whitelist,classifier}`
- `--classifier-embeddings <path>`
- `--classifier-models <path>`
- `--classifier-threshold <float>`
- `--max-arm-reach-m <float>`
- `--support-search-radius-m <float>`
- `--support-erosion-extra-margin-m <float>`
- `--ignore-nav-obstacle-categories straight_chair,armchair`
- `--remove-other-movable-objects`

## Training Command

Train diffusion policy on a collected dataset.

```bash
python vid2room_policy/vid2scene_policy/policy_training/diffusion_policy.py \
  --dataset_path ./lerobot_datasets/fetch_pick_task_test \
  --batch_size 256 \
  --num_epochs 60 \
  --learning_rate 1e-3 \
  --pred_horizon 16 \
  --obs_horizon 2 \
  --output_dir runs/diffusion_policy_100
```

Useful flags:

- `--num_workers 8`
- `--cache_size 2048`
- `--amp`
- `--save_every_epochs 20`
- `--overfit_single_batch`

## Replay Command

Replay one saved episode using recorded actions.

```bash
python vid2room_policy/replay_episode.py \
  --dataset-root ./lerobot_datasets/fetch_pick_task_test \
  --episode-idx 0 \
  --env-config ./OmniGibson/omnigibson/configs/fetch_vid2scene.yaml \
  --realtime \
  --debug
```

Optional:

- `--no-sticky-grasp`
- `--hold-seconds 5`

## Eval Command

Evaluate a trained checkpoint in OmniGibson using dataset episode metadata.

```bash
cd vid2room_policy
python -m vid2scene_policy.policy_evaluation.eval \
  --dataset-root ../lerobot_datasets/fetch_pick_task_100 \
  --checkpoint /home/behavior/Downloads/step_03000.pt \
  --env-config ../OmniGibson/omnigibson/configs/fetch_vid2scene.yaml \
  --num-diffusion-iters 100 \
  --device auto \
  --output-dir ../outputs/policy_eval_bh1/run_001
```

You can evaluate multiple episodes via:

```bash
python vid2room_policy/vid2scene_policy/policy_evaluation/eval.py \
  --dataset-root ./lerobot_datasets/fetch_pick_task_100 \
  --checkpoint ./runs/diffusion_policy_100/checkpoints/epoch_0020.pt \
  --episodes-file ./episode_ids.txt
```
