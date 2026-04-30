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

The script will interactively prompt you to:

- **Scene**: choose from `house_double_floor_lower`, `house_double_floor_upper`, `house_single_floor`, or enter a custom scene name.
- **Models**: for each required synset and category, choose one or more model IDs from those available on disk. A link to the synset page on the BEHAVIOR Knowledgebase (e.g. https://behavior.stanford.edu/knowledgebase/synsets/ashcan.n.01.html) is printed alongside each prompt to help you browse available models.

The script writes the completed entry directly to `datasets/2026-challenge-task-instances/metadata/task_custom_lists.json`. The result looks like:

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


### Step 2: Sample Task-Related Objects (TRO)

```bash
python OmniGibson/scripts/sampling/sample_b1k_tasks.py -t TASK_NAME
```

It is highly recommended to run this command with `-m pdb`, so it will stop at the error during sampling and you can debug interactively to see what's wrong. 

The scene is read automatically from `task_custom_lists.json`. After this command, you should see 2 files generated under `datasets/2026-challenge-task-instances/scenes/SCENE_NAME/json`: `house_double_floor_lower_task_picking_up_trash_0_0_template-partial_rooms.json` (intermediate) and `house_double_floor_lower_task_picking_up_trash_0_0_template.json` (postprocessed, with full scene objects merged in).

### Step 3: Generate Instances

Randomly generate 1 instances for your task:

```bash
python OmniGibson/scripts/sampling/multiply_b1k_tasks.py --partial_save --start_idx 1 --end_idx 1 -t TASK_NAME
```

After this step, a folder named `house_double_floor_lower_task_picking_up_trash_instances` appears under `datasets/2026-challenge-task-instances/scenes/SCENE_NAME/json`, containing files named like `house_double_floor_lower_task_picking_up_trash_0_1_template-tro_state.json`.

### Step 4: Presample Robot Poses

```bash
python OmniGibson/scripts/sampling/sample_robot_pose.py -t TASK_NAME
```

### Step 5: Register New Task

```bash
python OmniGibson/scripts/sampling/extract_task_information.py
```

### Step 6: Update Task Misc

Assign an ID to the task (also update the google sheet!), then put a new entry in `2026-challenge-task-instances/metadata/B100_task_misc.csv`. Take a look at the floor plan, and put in the task relevant rooms in the entry. Think of it as "what are the minimal set of rooms that is required for the robot to complete the task as if it's in a fully-loaded scene?"

Note that the rooms should include not only the rooms that contains the robot and TROs, but also those that connects them in between (e.g. corridors), as well as any room that could be in sight of the robot. Take house single floor as an example, for a task that only requires kitchen_0, we will need to load in the following 7 rooms:
    - corridor_0
    - dining_room_0
    - entryway_0
    - garden_0
    - kitchen_0
    - living_room_0
    - living_room_1 


### Step 7: Verify Task Viability

Prepare the joylo device, and run the following commands:

```bash
python joylo/scripts/launch_og.py --task-name TASK_NAME --recording-path HDF_PATH
```

```bash
python joylo/scripts/run_joylo.py
```

You should be able to complete the task without major bottlenecks. Watch out for any issues during teleoperation. Here are some examples:

    - Cannot complete the task (e.g. not able to navigate to a room because of narrow corridor)
    - Major artifacts / bad appearances in the scenes or objects. 
    - The task requires a lot of effort to complete (e.g. need to pick something up from a very high cabinet).
    - The tasks induces unavoidable collisions between robot and the environment to complete (e.g. robot can't pick up food from the oven without colliding with the door)
    - Other unreasonable behavior during teleoperation (e.g. object is too heavy to pickup, door is very hard to open, etc.)

If any of the above happens, either redo the previous sampling steps while fixing bugs, or if it's unfixable, discard the task and restart with another task. 

After teleoperation succeeds, you should see a `hdf5` file at `HDF_PATH`. Run the following replay script, which will generate the video and QA result JSON file:

```bash
OMNIGIBSON_HEADLESS=1 python joylo/scripts/replay_data.py HDF_PATH --task TASK_NAME --qa
```

If the HDF5 contains multiple saved demos, the replay script prints an episode-selection table with each `demo_N` episode ID and its trajectory length. The episode ID is the number in `demo_N`, so episode ID `2` replays `demo_2`. Press Enter to replay the longest trajectory, or enter an episode ID to replay a specific demo. For non-interactive runs, pass the episode explicitly:

```bash
OMNIGIBSON_HEADLESS=1 python joylo/scripts/replay_data.py HDF_PATH --task TASK_NAME --qa --episode_id EPISODE_ID
```

Check the QA json file output. **All QA should pass unless for task-specific reasons** (e.g. one failed grasp is allowed for `turning_on_radio` because the gripper needs to close to poke the button). Also check mp4 to make sure the replay visual looks reasonable. For example, you shouldn't see a whole in the ground, which might indicates you forgot to put one room in the task misc csv. 

If all outputs seems reasonable, share the generated MP4 file and QA result with the team for review.


### Step 8: Generate the rest of the 300 instances

Run the multiply script again, this time with index 2 to 300, and then sample robot poses, then update task yaml:

```bash
python OmniGibson/scripts/sampling/multiply_b1k_tasks.py --partial_save --start_idx 2 --end_idx 300 -t TASK_NAME -s SCENE_NAME
```

```bash
python OmniGibson/scripts/sampling/sample_robot_pose.py -t TASK_NAME
```

```bash
python OmniGibson/scripts/sampling/extract_task_information.py
```

### Step 9: Prepare all files and submit PR

After the task design is finalized, create a seperate branch in [2026-challenge-task-instances](https://github.com/wensi-ai/2026-challenge-task-instances), commit the files created:

    - two seed instance json files: `0_0_template.json`, `0_0_template-partial_rooms.json`
    - 300 task intance files under 
    - updated `task_custom_list.json` and `available_tasks.yaml`

Watch out for merge conflicts from main, which will most likely happen on `task_custom_list.json` and `available_tasks.yaml`. 

Submit a PR and tag the team for review.
