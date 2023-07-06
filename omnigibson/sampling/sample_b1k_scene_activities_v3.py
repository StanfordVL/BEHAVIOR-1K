import logging
import os
import yaml
import copy
import argparse
import bddl
import pkgutil
import omnigibson as og
from omnigibson.macros import gm
import json
import csv
import traceback
from omnigibson.objects import DatasetObject
from omnigibson.tasks import BehaviorTask
from omnigibson.systems import remove_callback_on_system_init, remove_callback_on_system_clear
from omnigibson.systems.system_base import clear_all_systems
from omnigibson.utils.python_utils import clear as clear_pu
from omnigibson.utils.bddl_utils import OBJECT_TAXONOMY
from bddl.activity import Conditions
import numpy as np
import gspread

"""
1. gcloud auth login
2. gcloud auth application-default login
3. gcloud config set project lucid-inquiry-205018
4. gcloud iam service-accounts create cremebrule
5. gcloud iam service-accounts keys create key.json --iam-account=cremebrule@lucid-inquiry-205018.iam.gserviceaccount.com
6. mv key.json /home/cremebrule/.config/gcloud/key.json
"""

SAMPLING_SHEET_KEY = "1Vt5s3JrFZ6_iCkfzZr0eb9SBt2Pkzx3xxzb4wtjEaDI"
CREDENTIALS = "key.json"
WORKSHEET = "Sheet1"
USER = "chengshu"

client = gspread.service_account(filename=CREDENTIALS)
worksheet = client.open_by_key(SAMPLING_SHEET_KEY).worksheet(WORKSHEET)

ACTIVITY_TO_ROW = {activity: i + 2 for i, activity in enumerate(worksheet.col_values(1)[1:])}

SCENE_MAPPING_FPATH = "BEHAVIOR-1K Tasks.csv"

UNSUPPORTED_PREDICATES = {"broken", "assembled", "attached"}


parser = argparse.ArgumentParser()
parser.add_argument("--scene_model", type=str, required=True,
                    help="Scene model to sample tasks in")
parser.add_argument("--activities", type=str, default=None, help="Activity/ie(s) to be sampled, if specified. This should be a comma-delimited list of desired activities. Otherwise, will try to sample all tasks in this scene")
parser.add_argument("--start_at", type=str, default=None, help="If specified, activity to start at, ignoring all previous")
parser.add_argument("--overwrite_existing", action="store_true",
                    help="If set, will overwrite any existing tasks that are found. Otherwise, will skip.")

args = parser.parse_args()


gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = False
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False


def get_predicates(conds):
    preds = []
    if isinstance(conds, str):
        return preds
    assert isinstance(conds, list)
    contains_list = np.any([isinstance(ele, list) for ele in conds])
    if contains_list:
        for ele in conds:
            preds += get_predicates(ele)
    else:
        preds.append(conds[0])
    return preds

def get_subjects(conds):
    subjs = []
    if isinstance(conds, str):
        return subjs
    assert isinstance(conds, list)
    contains_list = np.any([isinstance(ele, list) for ele in conds])
    if contains_list:
        for ele in conds:
            subjs += get_predicates(ele)
    else:
        subjs.append(conds[1])
    return subjs


def get_valid_tasks():
    return set(activity for activity in os.listdir(os.path.join(bddl.__path__[0], "activity_definitions")))


def parse_task_mapping(fpath):
    mapping = dict()
    rows = []
    with open(fpath) as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        for row in reader:
            rows.append(row)

    for row in rows[1:]:
        activity_name = row[0].split("-")[0]

        # Skip any that is missing a synset
        if len(row[1]) > 0:
            continue

        # Write matched ready scenes
        ready_scenes = set(list(entry.strip() for entry in row[2].split(","))[:-1])

        # There's always a leading whitespace
        if len(ready_scenes) == 0:
            continue

        mapping[activity_name] = ready_scenes

    return mapping


def get_scene_compatible_activities(scene_model, mapping):
    return [activity for activity, scenes in mapping.items() if scene_model in scenes]


def main(random_selection=False, headless=False, short_exec=False):
    # Define the configuration to load -- we'll use a Fetch
    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": args.scene_model,
        },
        "robots": [
            {
                "type": "Fetch",
                "obs_modalities": ["scan", "rgb", "depth"],
                "grasping_mode": "physical",
                "default_arm_pose": "diagonal30",
                "reset_joint_pos": "tuck",
            },
        ],
    }

    valid_tasks = get_valid_tasks()
    mapping = parse_task_mapping(fpath=SCENE_MAPPING_FPATH)
    activities = get_scene_compatible_activities(scene_model=args.scene_model, mapping=mapping)

    # Create the environment
    # Attempt to sample the activity
    env = og.Environment(configs=copy.deepcopy(cfg))

    # Store the initial state -- this is the safeguard to reset to!
    scene_initial_state = copy.deepcopy(env.scene._initial_state)
    og.sim.stop()

    n_scene_objects = len(env.scene.objects)

    # Set environment configuration after environment is loaded, because we will load the task
    env.task_config["type"] = "BehaviorTask"
    env.task_config["online_object_sampling"] = True

    should_start = args.start_at is None
    for activity in sorted(activities):
        if not should_start:
            if args.start_at == activity:
                should_start = True
            else:
                continue

        # Don't sample any invalid activities
        if activity not in valid_tasks:
            continue

        # Get info from spreadsheet
        row = ACTIVITY_TO_ROW[activity]
        success, scene_id, user, reason, misc = worksheet.get(f"B{row}:F{row}")[0]
        # If we've already sampled successfully (success is populated with a 1) and we don't want to overwrite the
        # existing sampling result, skip
        if success != "" and int(success) != 0 and not args.overwrite_existing:
            continue

        should_sample, success, reason = True, False, ""

        # Skip any with unsupported predicates, but still record the reason why we can't sample
        conditions = Conditions(activity, 0, simulator_name="omnigibson")
        init_predicates = set(get_predicates(conditions.parsed_initial_conditions))
        unsupported_predicates = set.intersection(init_predicates, UNSUPPORTED_PREDICATES)
        if len(unsupported_predicates) > 0:
            should_sample = False
            reason = f"Unsupported predicate(s): {unsupported_predicates}"

        # check for cloth covered
        for cond in conditions.parsed_initial_conditions:
            pred, subj = cond[0], cond[1]
            if pred == "covered" and "cloth" in OBJECT_TAXONOMY.get_abilities("_".join(subj.split("_")[:-1])):
                should_sample = False
                reason = f"Requires cloth covered support"
                break
        # init_subjects = set("_".join(subj.split("_")[:-1]) for subj in get_subjects(conditions.parsed_initial_conditions))

        env.task_config["activity_name"] = activity
        scene_instance = BehaviorTask.get_cached_activity_scene_filename(
            scene_model=args.scene_model,
            activity_name=activity,
            activity_definition_id=0,
            activity_instance_id=0,
        )

        # Make sure sim is stopped
        assert og.sim.is_stopped()

        # Attempt to sample
        try:
            if should_sample:
                env._load_task()
                assert og.sim.is_stopped()

                success = env.task.feedback is None
                if success:
                    # Sampling success
                    og.sim.play()
                    # This will actually reset the objects to their sample poses
                    env.task.reset(env)

                    # TODO: figure out whether we also should update in_room for newly imported objects
                    env.task.save_task(override=args.overwrite_existing)

                    og.sim.stop()
                    og.log.info(f"\n\nSampling success: {activity}\n\n")
                    reason = ""
                else:
                    og.log.error(f"\n\nSampling failed: {activity}.\n\nFeedback: {env.task.feedback}\n\n")
                    reason = env.task.feedback
            
            else:
                og.log.error(f"\n\nSampling failed: {activity}.\n\nFeedback: {reason}\n\n")

            # Write to google sheets
            cell_list = worksheet.range(f"B{row}:E{row}")
            for cell, val in zip(cell_list, (int(success), args.scene_model, USER, "" if reason is None else reason)):
                cell.value = val
            worksheet.update_cells(cell_list)

            # Clear task callbacks if sampled
            if should_sample:
                callback_name = f"{activity}_refresh"
                og.sim.remove_callback_on_import_obj(name=callback_name)
                og.sim.remove_callback_on_remove_obj(name=callback_name)
                remove_callback_on_system_init(name=callback_name)
                remove_callback_on_system_clear(name=callback_name)

                # Remove all the additionally added objects
                for obj in env.scene.objects[n_scene_objects:]:
                    og.sim.remove_object(obj)

            # Clear all systems
            clear_all_systems()
            clear_pu()

            og.sim.step()
            og.sim.play()
            # This will clear out the previous attachment group in macro particle systems
            og.sim.scene.load_state(scene_initial_state)
            og.sim.step()
            og.sim.scene.update_initial_state()
            og.sim.stop()

        except Exception as e:
            og.log.error(traceback.format_exc())
            og.log.error(f"\n\nCaught exception sampling activity {activity} in scene {args.scene_model}:\n\n{e}\n\n")

            try:
                # Stop sim, clear simulator, and re-create environment
                e_str = f"{e}"
                og.sim.stop()
                og.sim.clear()
            except AttributeError as e:
                # This is the "GetPath" error that happens sporatically. It's benign, so we ignore it
                pass

            env = og.Environment(configs=copy.deepcopy(cfg))

            # Store the initial state -- this is the safeguard to reset to!
            scene_initial_state = copy.deepcopy(env.scene._initial_state)
            og.sim.stop()

            n_scene_objects = len(env.scene.objects)

            # Set environment configuration after environment is loaded, because we will load the task
            env.task_config["type"] = "BehaviorTask"
            env.task_config["online_object_sampling"] = True

    # Shutdown at the end
    og.shutdown()

if __name__ == "__main__":
    main()
