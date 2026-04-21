# :material-chart-scatter-plot: **Task Sampling**

Generate fresh instances of existing tasks with randomized elements for variety and robustness testing.

## Getting Started

Clone `2026-challenge-task-instances` into `gm.DATA_PATH`:

```bash
git clone https://github.com/wensi-ai/2026-challenge-task-instances
```

## Sampling Workflow


### Step 1: Review BDDL and Generate the JSON Template

Pick a task,review the bddl definition under `bddl3/bddl/activity_definitions/TASK_NAME/problem_0.bddl`. Make sure the defintion is reasonable. In particular watch out for wildcard expansions. 

Then, generate a JSON template for your task:

```bash
python OmniGibson/scripts/sampling/autogenerate_task_custom_list_template.py -t TASK_NAME
```

The command will output a template like this:

```json
{
    "picking_up_trash": {
        "room_types": [
            "living_room",
            "kitchen"
        ],
        "__TODO__SCENE__": {
            "whitelist": {
                "ashcan.n.01": {
                    "ashcan": {
                        "__TODO__MODEL__": null
                    }
                },
                "can__of__soda.n.01": {
                    "can__of__soda": {
                        "__TODO__MODEL__": null
                    }
                }
            },
            "blacklist": {}
        }
    }
}
```

Fill in `__TODO__SCENE__` and `__TODO__MODEL__`, then copy the modified JSON snippts into `datasets/2026-challenge-task-instances/metadata/task_custom_lists.json`.

**Available scenes:** [BEHAVIOR Knowledgebase - Scenes](https://behavior.stanford.edu/knowledgebase/scenes/index.html)

**Object models:** Choose any objects that fall into the category, e.g. for `ashcan` see [ashcan.n.01](https://behavior.stanford.edu/knowledgebase/synsets/ashcan.n.01.html)

An example filled JSON looks like this:

```json
"picking_up_trash": {
    "room_types": [
        "living_room",
        "kitchen"
    ],
    "house_double_floor_lower": {
        "whitelist": {
            "can__of__soda.n.01": {
                "can_of_soda": {
                    "itolcg": null,
                    "lugwcz": null,
                    "opivig": null
                }
            },
            "ashcan.n.01": {
                "trash_can": {
                    "wkxtxh": null
                }
            }
        },
        "blacklist": {}
    }
}
```

> **Note:** Copy into `2026-challenge-task-instances`, not `2025-challenge-task-instances`. The 2026 version takes precedence.


### Step 2: Sample Task-Related Objects (TRO)

```bash
python OmniGibson/scripts/sampling/sample_b1k_tasks.py -t TASK_NAME -s SCENE_NAME
```

It is highly recommended to run this command with `-m pdb`, so it will stop at the error during sampling and you can debug interactively to see what's wrong. 

Make sure the script runs all the way until sampling succeeded. After that, one file is generated under `datasets/2026-challenge-task-instances/scenes/SCENE_NAME/json`, with a name like `house_double_floor_lower_task_picking_up_trash_0_0_template-partial_rooms.json`.

### Step 3: Postprocess Sampled JSON

Add static scene objects to the generated file:

```bash
python OmniGibson/scripts/sampling/postprocess_sampled_task.py -t TASK_NAME -s SCENE_NAME
```

After this command, another file is generated under `datasets/2026-challenge-task-instances/scenes/SCENE_NAME/json`: `house_double_floor_lower_task_picking_up_trash_0_0_template.json`.

### Step 4: Generate Instances

Randomly generate 1 instances for your task:

```bash
python OmniGibson/scripts/sampling/multiply_b1k_tasks.py --partial_save --start_idx 1 --end_idx 1 -t TASK_NAME -s SCENE_NAME
```

After this step, a folder named `house_double_floor_lower_task_picking_up_trash_instances` appears under `datasets/2026-challenge-task-instances/scenes/SCENE_NAME/json`, containing files named like `house_double_floor_lower_task_picking_up_trash_0_1_template-tro_state.json`.

### Step 5: Presample Robot Poses

```bash
python OmniGibson/scripts/sampling/sample_robot_pose.py -t TASK_NAME -s SCENE_NAME
```

### Step 6: Register New Task

```bash
python OmniGibson/scripts/sampling/extract_task_information.py
```

### Step 7: Update Task Misc

Put a new entry in `2026-challenge-task-instances/metadata/B100_task_misc.csv`


### Step 8: Verify Task Viability

Prepare the joylo device, and run the following commands:

```bash
python joylo/scripts/launch_og.py --task-name TASK_NAME --recording-path HDF_PATH
```

```bash
python joylo/scripts/run_joylo.py
```

You should be able to complete the task without major bottlenecks. Watch out for any issues during teleoperation. Here are some examples:

    - Cannot complete the task (e.g. not able to navigate to a room because of narrow corridor)
    - Major artifacts / bad appearances in the scenes or objects 
    - The task requires a lot of effort to complete (e.g. need to pick something up from a very high cabinet).
    - The tasks induces unavoidable collisions between robot and the environment to complete (e.g. robot can't pick up food from the oven without colliding with the door)
    - Other unreasonable behavior during teleoperation (e.g. )

If the following happens, either redo the previous sampling steps while fixing bugs, or discard the task and restart with another task. 

After teleoperation succeeds, you should see a `hdf5` file at `HDF_PATH`. Run the following replay script, which will generate the video and QA result JSON file:

```bash
python joylo/scripts/replay_data.py HDF_PATH --task TASK_NAME --qa
```

Share the generated MP4 file with the team for review.


### Step 9: Generate the rest of the 300 instances

Run the multiply script again, this time with index 2 to 300, and then sample robot poses, then update task yaml:

```bash
python OmniGibson/scripts/sampling/multiply_b1k_tasks.py --partial_save --start_idx 2 --end_idx 300 -t TASK_NAME -s SCENE_NAME
```

```bash
python OmniGibson/scripts/sampling/sample_robot_pose.py -t TASK_NAME -s SCENE_NAME
```

```bash
python OmniGibson/scripts/sampling/extract_task_information.py
```

### Step 10: Prepare all files and submit PR

After the task design is finalized, create a seperate branch in [2026-challenge-task-instances](https://github.com/wensi-ai/2026-challenge-task-instances), commit the files created:

    - two seed instance json files: `0_0_template.json`, `0_0_template-partial_rooms.json`
    - 300 task intance files under 
    - updated `task_custom_list.json` and `available_tasks.yaml`

Watch out for merge conflicts from main, which will most likely happen on `task_custom_list.json` and `available_tasks.yaml`. 

Submit a PR and tag the team for review.