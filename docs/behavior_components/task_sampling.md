
### Sampling New Task Instances

Generate fresh instances of existing tasks with randomized elements for variety and robustness testing.

**Example: Sampling a pick up trash task, task name = picking_up_trash**
0. Clone `2026-challenge-task-instances` into `gm.DATA_PATH`: https://github.com/wensi-ai/2026-challenge-task-instances 

1. Enter the sampling directory,
```sh
cd YOUR_PATH_TO_BEHAVIOR/BEHAVIOR-1K/OmniGibson/scripts/sampling
```

2. Review the BDDL definition for your task to understand what objects and conditions are involved:

```
bddl3/bddl/activity_definitions/TASK_NAME/problem0.bddl
```

3. Generate and write the task custom list entry for your task:
```sh
python autogenerate_task_custom_list_template.py -t TASK_NAME 
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


4. Sample task related objects

```sh
python sample_b1k_tasks.py -t TASK_NAME
```
The scene is read automatically from `task_custom_lists.json`. After this command, you should see 2 files generated under `datasets/2026-challenge-task-instances/scenes/SCENE_NAME/json`: `house_double_floor_lower_task_picking_up_trash_0_0_template-partial_rooms.json` (intermediate) and `house_double_floor_lower_task_picking_up_trash_0_0_template.json` (postprocessed, with full scene objects merged in).

5. Randomly generate 300 instances for your task
```sh
python multiply_b1k_tasks.py --partial_save --start_idx 1 --end_idx 300 -t TASK_NAME
```
After this step, you should see a folder named house_double_floor_lower_task_picking_up_trash_instances appeared under `datasets/2026-challenge-task-instances/scenes/SCENE_NAME/json`, in which there are files with name like house_double_floor_lower_task_picking_up_trash_0_(index)_template-tro_state.json.

6. Presample poses for all supported robots
```sh
python sample_robot_pose.py -t TASK_NAME
```

7. To verify whether your sampling is successful, specify your task name in OmniGibson/omnigibson/configs/r1pro_behavior.yaml and run 
```sh
python /BEHAVIOR-1K/joylo/scripts/launch_og.py --task-name TASK_NAME --recording-path HDF_PATH
```

```sh
python /BEHAVIOR-1K/joylo/scripts/run_joylo.py
```

Try to complete the task by running the replay script to generate the video and qa result json file;

```sh
python /BEHAVIOR-1K/joylo/scripts/replay_data.py HDF_PATH --task TASK_NAME --qa
```

8. Finishing up

Share the generated mp4 file to @wensi-ai for review. 

After the task design is finalized, submit a PR to 2026-challenge-task-instances (be careful for merge conflicts!) and tag @wensi-ai for review. 