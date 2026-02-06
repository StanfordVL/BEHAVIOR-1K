# Policy Evaluation

Evaluate trained LeRobot policies in OmniGibson environments.

## Basic Usage

```bash
python -m vid2room_policy.vid2scene_policy.policy_evaluation.eval \
    --policy_path /path/to/checkpoint/pretrained_model \
    --n_episodes 1 \
    --output_dir ./eval_results \
    --max_steps 100
```

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--policy_path` | Path to policy checkpoint (required) | - |
| `--n_episodes` | Number of episodes to evaluate | `10` |
| `--max_steps` | Maximum steps per episode | `500` |
| `--output_dir` | Output directory for results | `./eval_results` |
| `--seed` | Random seed | `42` |
| `--metadata` | Single episode metadata as JSON string | `None` |
| `--episodes_file` | JSON file with list of episode metadata | `None` |
| `--no_save_videos` | Disable video recording | `False` |

## Specifying Episode Metadata

You can specify episode configuration via `--metadata` (single episode) or `--episodes_file` (multiple episodes).

### Single Episode via `--metadata`

```bash
OMNIGIBSON_HEADLESS=1 python -m vid2room_policy.vid2scene_policy.policy_evaluation.eval \
    --policy_path /checkpoint/clear/cgokmen/policies/pi05-spoc/checkpoints/006000/pretrained_model \
    --n_episodes 1 \
    --output_dir ./eval_results \
    --max_steps 3000 \
    --episodes_file "/home/cgokmen/projects/BEHAVIOR-1K/slurm/eval-starts/Beechwood_1_int-0.json"
```

### Multiple Episodes via `--episodes_file`

Create a JSON file with a list of episode metadata:

```json
[
    {
        "source_support_name": "table_1",
        "target_support_name": "shelf_1",
        "robot_start_x_y_z_theta": [1.0, 2.0, 0.0, 0.0],
        "spawned_target_object": false,
        "target_object_name": "",
        "spawned_target_object_position": [0, 0, 0],
        "spawned_target_object_orientation": [0, 0, 0, 1],
        "spawned_target_object_category": "",
        "spawned_target_object_model": ""
    },
    {
        "source_support_name": "table_2",
        "target_support_name": "cabinet_1",
        "robot_start_x_y_z_theta": [3.0, 4.0, 0.0, 1.57],
        "spawned_target_object": true,
        "target_object_name": "cup_1",
        "spawned_target_object_position": [3.5, 4.2, 0.8],
        "spawned_target_object_orientation": [0, 0, 0, 1],
        "spawned_target_object_category": "cup",
        "spawned_target_object_model": "cup_model_id"
    }
]
```

Then run:

```bash
python -m vid2room_policy.vid2scene_policy.policy_evaluation.eval \
    --policy_path /path/to/checkpoint/pretrained_model \
    --episodes_file episodes.json \
    --output_dir ./eval_results
```

### Metadata Fields

| Field | Description |
|-------|-------------|
| `source_support_name` | Name of the support object where the target starts |
| `target_support_name` | Name of the support object where target should be placed |
| `robot_start_x_y_z_theta` | Robot starting pose `[x, y, z, theta]` |
| `spawned_target_object` | Whether to spawn a target object |
| `target_object_name` | Name for the spawned object |
| `spawned_target_object_position` | Position `[x, y, z]` for spawned object |
| `spawned_target_object_orientation` | Quaternion `[x, y, z, w]` for spawned object |
| `spawned_target_object_category` | Category of the object (e.g., "helmet", "cup") |
| `spawned_target_object_model` | Model ID for the object |
| `spawned_target_object_dataset_name` | Dataset containing the object (default: "behavior-1k-assets") |

### Default Behavior (No Metadata)

If no metadata is provided, the robot starts at position `[0, 0, 0]` with no target object spawned.

## Output

Results are saved to `--output_dir`:
- `eval_results.json` - Summary statistics and per-episode results
- `videos/` - Recorded episode videos (unless `--no_save_videos`)
