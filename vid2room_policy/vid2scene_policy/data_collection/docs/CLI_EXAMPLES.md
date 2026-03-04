# CLI Examples

Example commands for running pick-only data collection:

```bash
python vid2room_policy/collect_data.py \
  --scene_dataset proctor \
  --scene_name scene_name \
  --num_episodes 50 \
  --output ./lerobot_datasets \
  --repo-id r1pro_pick_proctor
```

```bash
python vid2room_policy/collect_data.py \
  --scene_dataset behavior-1k-assets \
  --scene_name Rs_int \
  --num_episodes 100 \
  --output ./lerobot_datasets \
  --repo-id r1pro_pick_behavior1k
```

```bash
python vid2room_policy/collect_data.py \
  --scene_dataset spoc \
  --scene_name train_505 \
  --num_episodes 25 \
  --output ./lerobot_datasets \
  --repo-id r1pro_pick_spoc
```

```bash
python vid2room_policy/collect_data.py \
  --scene_dataset behavior-1k-assets \
  --scene_name house_double_floor_lower \
  --num_episodes 10 \
  --object-filter whitelist \
  --classifier-threshold 0.5
```

```bash
python vid2room_policy/collect_data.py \
  --scene_dataset behavior-1k-assets \
  --scene_name house_double_floor_lower \
  --num_episodes 10 \
  --max-arm-reach-m 1.15 \
  --support-search-radius-m 3.0
```
