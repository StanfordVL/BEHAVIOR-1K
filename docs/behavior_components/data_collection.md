# :material-database-arrow-down: **Data Collection**

Collect demonstrations for BEHAVIOR tasks using the JoyLo teleoperation system.

## Preparation

This preparation step needs only be done once.

- Make sure you are on the latest `main` branch of `BEHAVIOR-1K`
- Update the latest robot assets
    ```bash
    python OmniGibson/omnigibson/utils/asset_utils.py --update_omnigibson_robot_assets
    ```
- Go inside `2025-challenge-task-instances` and `git pull`
- Clone [2026-challenge-task-instances](https://github.com/wensi-ai/2026-challenge-task-instances.git) into `BEHAVIOR-1K/datasets`

**Outcome:** You should see the following folders under `BEHAVIOR-1K/datasets`:

- `2025-challenge-task-instances`
- `2026-challenge-task-instances`
- `behavior-1k-assets`
- `omnigibson-robot-assets`

## Data Collection Workflow

### Step 1: Pull the Latest Code and Sampled Tasks

Pull the latest changes from both the `main` branch of BEHAVIOR-1K and `datasets/2026-challenge-task-instances`.


```bash
git pull
```

### Step 2: Pick a Task

Available tasks can be found in `2026-challenge-task-instances/metadata/available_tasks.yaml`.

Launch the following scripts to start data collection:

```bash
python joylo/scripts/launch_og.py --task-name TASK_NAME --recording-path HDF_PATH
```

```bash
python joylo/scripts/run_joylo.py
```

### Step 3: Replay Trajectory

Run the following script to replay the trajectory. This will create one `video.mp4` for visual QA and a `qa_results.json`, which includes the results from the QA script.

```bash
python joylo/scripts/replay_data.py HDF_PATH --task TASK_NAME --qa
```

If the HDF5 contains multiple saved demos, the replay script prints an episode-selection table with each `demo_N` episode ID and its trajectory length. The episode ID is the number in `demo_N`, so episode ID `2` replays `demo_2`. Press Enter to replay the longest trajectory, or enter an episode ID to replay a specific demo. For non-interactive runs, pass the episode explicitly:

```bash
python joylo/scripts/replay_data.py HDF_PATH --task TASK_NAME --qa --episode_id EPISODE_ID
```
