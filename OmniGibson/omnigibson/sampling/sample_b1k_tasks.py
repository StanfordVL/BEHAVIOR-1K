import logging
import os
import copy
import argparse
import omnigibson as og
from omnigibson.macros import gm, macros
import json
import traceback
from omnigibson.objects import DatasetObject
from omnigibson.object_states import Contains
from omnigibson.tasks import BehaviorTask
from omnigibson.utils.asset_utils import get_dataset_path
from omnigibson.utils.python_utils import clear as clear_pu
from omnigibson.utils.constants import PrimType
from bddl.activity import Conditions
from utils import (
    ACTIVITY_TO_ROW,
    create_stable_scene_json,
    validate_scene_can_be_sampled,
    get_scene_compatible_activities,
    get_unsuccessful_activities,
    get_rooms,
    get_predicates,
    get_valid_tasks,
    hide_all_lights,
    parse_task_mapping_new,
    UNSUPPORTED_PREDICATES,
    USER,
    validate_task,
    worksheet,
)
import numpy as np
import random


# TASK_CUSTOM_LISTS = {
#     ("picking_up_trash", "Rs_int"): {
#         "whitelist": {
#             "pad.n.01": {
#                 "sticky_note": ["tghqep"],
#             },
#         },
#         "blacklist": None,
#     },
#     ("packing_recreational_vehicle_for_trip", "Merom_0_garden"): {
#         "whitelist": {
#             "wicker_basket.n.01": {
#                 "wicker_basket": ["tsjvyu"],
#             },
#         },
#         "blacklist": None,
#     },
#     ("datagen_tidy_table", "house_single_floor"): {
#         "whitelist": {
#             "countertop.n.01": {
#                 "bar": ["udatjt"],
#             },
#             "teacup.n.02": {
#                 "teacup": ["kccqwj"],
#             },
#         },
#         "blacklist": None,
#     },
#     ("datagen_cook_brussels_sprouts", "house_single_floor"): {
#         "whitelist": {
#             "countertop.n.01": {
#                 "bar": ["udatjt"],
#             },
#             "burner.n.02": {
#                 "burner": ["mjvqii"],
#             },
#             "brussels_sprouts.n.01": {
#                 "brussels_sprouts": ["hkwyzk"],
#             },
#             "stockpot.n.01": {
#                 "stockpot": ["grrcna"],
#             },
#             "tupperware.n.01": {
#                 "tupperware": ["mkstwr"],
#             },
#         },
#         "blacklist": None,
#     },
#     ("datagen_wash_dishes", "house_single_floor"): {
#         "whitelist": {
#             "countertop.n.01": {
#                 "bar": ["gjeoer"],
#             },
#             "frying_pan.n.01": {
#                 "frying_pan": ["jpzusm"],
#             },
#             "scrub_brush.n.01": {
#                 "scrub_brush": ["hsejyi"],
#             },
#         },
#         "blacklist": None,
#     },
#     ("datagen_dishes_away", "house_single_floor"): {
#         "whitelist": {
#             "countertop.n.01": {
#                 "bar": ["gjeoer"],
#             },
#             "plate.n.04": {
#                 "plate": ["akfjxx"],
#             },
#             "shelf.n.01": {
#                 "shelf": ["pfusrd"],
#             },
#         },
#         "blacklist": None,
#     },
#     ("datagen_pick", "Rs_int"): {
#         "whitelist": {
#             "breakfast_table.n.01": {
#                 "breakfast_table": ["bhszwe"],
#             },
#             "coffee_cup.n.01": {
#                 "coffee_cup": ["dkxddg"],
#             },
#         },
#         "blacklist": None,
#     },
# }

with open("task_custom_lists.json", "r") as f:
    TASK_CUSTOM_LISTS = json.load(f)

# TODO:
# 1. Set boundingCube approximation earlier (maybe right after importing the scene objects). Otherwise after loading the robot, we will elapse one physics step
# 2. Enable transition rule and refresh all rules before online validation

parser = argparse.ArgumentParser()
parser.add_argument("--scene_model", type=str, default=None, help="Scene model to sample tasks in")
parser.add_argument(
    "--activities",
    type=str,
    default=None,
    help="Activity/ie(s) to be sampled, if specified. This should be a comma-delimited list of desired activities. Otherwise, will try to sample all tasks in this scene",
)
parser.add_argument(
    "--room_types",
    type=str,
    default=None,
    help="room types to be loaded, if specified. This should be a comma-delimited list of desired room types. Otherwise, will try to load all room types in this scene",
)
parser.add_argument(
    "--start_at", type=str, default=None, help="If specified, activity to start at, ignoring all previous"
)
parser.add_argument(
    "--thread_id", type=str, default=None, help="If specified, ID to assign to the thread when tracking in_progress"
)
parser.add_argument("--randomize", action="store_true", help="If set, will randomize order of activities.")
parser.add_argument(
    "--overwrite_existing",
    action="store_true",
    help="If set, will overwrite any existing tasks that are found. Otherwise, will skip.",
)
parser.add_argument(
    "--offline", action="store_true", help="If set, will sample offline, and will not sync / check with google sheets"
)
parser.add_argument(
    "--ignore_in_progress", action="store_true", help="If set and --offline is False, will in progress flag"
)

# gm.HEADLESS = False
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False  # Must be False! We permute this later

macros.systems.micro_particle_system.MICRO_PARTICLE_SYSTEM_MAX_VELOCITY = 0.5
macros.systems.macro_particle_system.MACRO_PARTICLE_SYSTEM_MAX_DENSITY = 200.0
# macros.prims.entity_prim.DEFAULT_SLEEP_THRESHOLD = 0.0
macros.utils.object_state_utils.DEFAULT_HIGH_LEVEL_SAMPLING_ATTEMPTS = 5
macros.utils.object_state_utils.DEFAULT_LOW_LEVEL_SAMPLING_ATTEMPTS = 5

logging.getLogger().setLevel(logging.INFO)


def main(random_selection=False, headless=False, short_exec=False):
    args = parser.parse_args()

    # Parse arguments based on whether values are specified in os.environ
    # Priority is:
    # 1. command-line args
    # 2. environment level variables
    if args.scene_model is None:
        # This MUST be specified
        assert os.environ.get(
            "SAMPLING_SCENE_MODEL"
        ), "scene model MUST be specified, either as a command-line arg or as an environment variable!"
        args.scene_model = os.environ["SAMPLING_SCENE_MODEL"]
    if args.activities is None and os.environ.get("SAMPLING_ACTIVITIES"):
        args.activities = os.environ["SAMPLING_ACTIVITIES"]
    if args.start_at is None and os.environ.get("SAMPLING_START_AT"):
        args.start_at = os.environ["SAMPLING_START_AT"]
    if args.thread_id is None:
        # This checks for both "" and non-existent key
        args.thread_id = os.environ["SAMPLING_THREAD_ID"] if os.environ.get("SAMPLING_THREAD_ID") else "1"
    if not args.randomize:
        args.randomize = os.environ.get("SAMPLING_RANDOMIZE") in {"1", "true", "True"}
    if not args.overwrite_existing:
        args.overwrite_existing = os.environ.get("SAMPLING_OVERWRITE_EXISTING") in {"1", "true", "True"}
    if not args.ignore_in_progress:
        args.ignore_in_progress = os.environ.get("SAMPLING_IGNORE_IN_PROGRESS") in {"1", "true", "True"}

    # Make sure scene can be sampled by current user
    scene_row = None if args.offline else validate_scene_can_be_sampled(scene=args.scene_model)

    if not args.offline and not args.randomize:
        completed = worksheet.get(f"W{scene_row}")
        if completed and completed[0] and str(completed[0][0]) == "1":
            # If completed is set, then immediately return
            print(f"\nScene {args.scene_model} already completed sampling, terminating immediately!\n")
            return

        # Potentially update start_at based on current task observed
        # Current task is either an empty list [] or a filled list [['<ACTIVITY>']]
        current_task = worksheet.get(f"Y{scene_row}")
        if not args.randomize and args.start_at is None and current_task and current_task[0]:
            args.start_at = current_task[0][0]
            # Also clear the in_progress bar in case this is from a failed run
            worksheet.update_acell(f"B{ACTIVITY_TO_ROW[args.start_at]}", "")

        # Set the thread id for the given scene
        worksheet.update_acell(f"X{scene_row}", args.thread_id)

    # If we want to create a stable scene config, do that now
    default_scene_fpath = os.path.join(
        get_dataset_path("behavior-1k-assets"), "scenes", args.scene_model, "json", f"{args.scene_model}_stable.json"
    )
    if not os.path.exists(default_scene_fpath):
        create_stable_scene_json(scene_model=args.scene_model)

    # Get the default scene instance
    assert os.path.exists(default_scene_fpath), "Did not find default stable scene json!"
    with open(default_scene_fpath, "r") as f:
        default_scene_dict = json.load(f)

    # Define the configuration to load -- we'll use a Fetch
    cfg = {
        # Use default frequency
        "env": {
            "action_frequency": 30,
            "physics_frequency": 120,
        },
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_file": default_scene_fpath,
            "scene_model": args.scene_model,
            "seg_map_resolution": 0.1,
            # "load_object_categories": ["floors"],
        },
        "robots": [
            {
                "type": "R1Pro",
                "obs_modalities": [],
                "default_reset_mode": "untuck",
                "position": np.ones(3) * -50.0,
            },
        ],
    }

    # Currently our sampling script always samples partial rooms so we specify there to delineate between full
    # scene templates
    task_suffix = "partial_rooms"
    if args.room_types is not None:
        cfg["scene"]["load_room_types"] = args.room_types.split(",")
    else:
        activities = args.activities.split(",")
        assert len(activities) == 1
        cfg["scene"]["load_room_types"] = TASK_CUSTOM_LISTS[activities[0]]["room_types"]

    valid_tasks = get_valid_tasks()
    # mapping = parse_task_mapping(fpath=TASK_INFO_FPATH)
    mapping = parse_task_mapping_new()
    activities = (
        get_scene_compatible_activities(scene_model=args.scene_model, mapping=mapping)
        if args.activities is None
        else args.activities.split(",")
    )

    # if we're not offline, only keep the failure cases
    if not args.offline:
        activities = list(set(activities).intersection(get_unsuccessful_activities()))

    # Create the environment
    # Attempt to sample the activity
    # env = create_env_with_stable_objects(cfg)
    with gm.unlocked():
        gm.ENABLE_TRANSITION_RULES = True
        env = og.Environment(configs=copy.deepcopy(cfg))
        gm.ENABLE_TRANSITION_RULES = False
    if gm.HEADLESS:
        hide_all_lights()
    else:
        og.sim.enable_viewer_camera_teleoperation()

    # After we load the robot, we do self.scene.reset() (one physics step) and then self.scene.update_initial_file().
    # We need to set all velocities to zero after this. Otherwise, the visual only objects will drift.
    for obj in env.scene.objects:
        obj.keep_still()
    env.scene.update_initial_file()

    # Store the initial state -- this is the safeguard to reset to!
    scene_initial_file = copy.deepcopy(env.scene._initial_file)
    og.sim.stop()

    n_scene_objects = len(env.scene.objects)

    # Set environment configuration after environment is loaded, because we will load the task
    env.task_config["type"] = "BehaviorTask"
    env.task_config["online_object_sampling"] = True

    should_start = args.start_at is None
    if args.randomize:
        random.shuffle(activities)
    else:
        activities = sorted(activities)
    for activity in activities:
        print(f"Checking activity: {activity}...")
        if not should_start:
            if args.start_at == activity:
                should_start = True
            else:
                continue

        # Don't sample any invalid activities
        if activity not in valid_tasks:
            continue

        if not args.offline:
            if activity not in ACTIVITY_TO_ROW:
                continue

            # Get info from spreadsheet
            row = ACTIVITY_TO_ROW[activity]

            in_progress, success, validated, scene_id, user, reason, exception, misc = worksheet.get(f"B{row}:I{row}")[
                0
            ]

            # If we manually do not want to sample the task (DO NOT SAMPLE == "DNS", skip)
            if success.lower() == "dns":
                continue

            # Only sample stuff which is fixed
            # if "fixed" not in misc.lower():
            #     continue

            # If we've already sampled successfully (success is populated with a 1) and we don't want to overwrite the
            # existing sampling result, skip
            if success != "" and int(success) == 1 and not args.overwrite_existing:
                continue

            # If another thread is already in the process of sampling, skip
            if not args.ignore_in_progress and in_progress not in {None, ""}:
                continue

            # Reserve this task by marking in_progress = 1
            worksheet.update_acell(f"B{row}", args.thread_id)
            worksheet.update_acell(f"Y{scene_row}", activity)

        should_sample, success, reason = True, False, ""

        # Skip any with unsupported predicates, but still record the reason why we can't sample
        conditions = Conditions(activity, 0, simulator_name="omnigibson")
        all_predicates = set(
            get_predicates(conditions.parsed_initial_conditions) + get_predicates(conditions.parsed_goal_conditions)
        )
        unsupported_predicates = set.intersection(all_predicates, UNSUPPORTED_PREDICATES)
        if len(unsupported_predicates) > 0:
            should_sample = False
            reason = f"Unsupported predicate(s): {unsupported_predicates}"

        env.task_config["activity_name"] = activity
        # activity_scene_combo = (activity, args.scene_model)
        # if activity_scene_combo in TASK_CUSTOM_LISTS:
        #     whitelist = TASK_CUSTOM_LISTS[activity_scene_combo]["whitelist"]
        #     blacklist = TASK_CUSTOM_LISTS[activity_scene_combo]["blacklist"]
        if activity in TASK_CUSTOM_LISTS and args.scene_model in TASK_CUSTOM_LISTS[activity]:
            whitelist = TASK_CUSTOM_LISTS[activity][args.scene_model]["whitelist"]
            blacklist = TASK_CUSTOM_LISTS[activity][args.scene_model]["blacklist"]
        else:
            whitelist, blacklist = None, None
        env.task_config["sampling_whitelist"] = whitelist
        env.task_config["sampling_blacklist"] = blacklist
        print("white_list", whitelist)
        print("black_list", blacklist)
        assert whitelist is not None, "whitelist should not be None for manual sampling"
        BehaviorTask.get_cached_activity_scene_filename(
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
                relevant_rooms = set(get_rooms(conditions.parsed_initial_conditions))
                print(f"relevant rooms: {relevant_rooms}")
                for obj in env.scene.objects:
                    if isinstance(obj, DatasetObject):
                        obj_rooms = {"_".join(room.split("_")[:-1]) for room in obj.in_rooms}
                        active = len(relevant_rooms.intersection(obj_rooms)) > 0 or obj.category in {"floors", "walls"}
                        obj.visual_only = not active
                        obj.visible = active

                og.log.info(f"Sampling task: {activity}")
                env._load_task()
                assert og.sim.is_stopped()

                success, feedback = env.task.feedback is None, env.task.feedback

                if success:
                    # Set masses of all task-relevant objects to be very high
                    # This is to avoid particles from causing instabilities
                    # Don't use this on cloth since these may be unstable at high masses
                    for obj in env.scene.objects[n_scene_objects:]:
                        if obj.prim_type != PrimType.CLOTH and Contains in obj.states:
                            obj.root_link.mass = max(1.0, obj.root_link.mass)

                    # Sampling success
                    og.sim.play()
                    # This will actually reset the objects to their sample poses
                    env.task.reset(env)

                    for i in range(300):
                        og.sim.step()

                    # Remove any particles that fell out of the world
                    for system in env.scene.active_systems.values():
                        if system.n_particles > 0:
                            particle_positions, _ = system.get_particles_position_orientation()
                            remove_idxs = np.where(particle_positions[:, -1] < -1.0)[0]
                            if len(remove_idxs) > 0:
                                system.remove_particles(remove_idxs)

                    # Make sure objects are settled
                    for _ in range(10):
                        og.sim.step()

                    task_final_state = env.scene.dump_state()
                    task_scene_dict = {"state": task_final_state}
                    # from IPython import embed; print("validate_task"); embed()
                    for obj in env.task.object_scope.values():
                        if isinstance(obj, DatasetObject):
                            obj.wake()
                    validated, error_msg = validate_task(env.task, task_scene_dict, default_scene_dict)
                    if not validated:
                        success = False
                        feedback = error_msg
                        print("validation failed")
                        print(f"REASON: {feedback}")
                        breakpoint()

                if success:
                    env.scene.load_state(task_final_state)
                    env.scene.update_initial_file()
                    print("sampling succeed")
                    breakpoint()
                    env.task.save_task(env=env, override=True, task_relevant_only=False, suffix=task_suffix)
                    og.log.info(f"\n\nSampling success: {activity}\n\n")
                    reason = ""
                else:
                    reason = feedback
                    og.log.error(f"\n\nSampling failed: {activity}.\n\nFeedback: {reason}\n\n")
                    breakpoint()
                og.sim.stop()
            else:
                og.log.error(f"\n\nSampling failed: {activity}.\n\nFeedback: {reason}\n\n")

            assert og.sim.is_stopped()

            # Write to google sheets
            if not args.offline:
                # Check if another thread succeeded already
                already_succeeded = worksheet.get(f"C{row}")
                if not (already_succeeded and already_succeeded[0] and str(already_succeeded[0][0]) == "1"):
                    cell_list = worksheet.range(f"B{row}:H{row}")
                    for cell, val in zip(cell_list, ("", int(success), "", args.scene_model, USER, reason, "")):
                        cell.value = val
                    worksheet.update_cells(cell_list)

            # Clear task callbacks if sampled
            if should_sample:
                callback_name = f"{activity}_refresh"
                og.sim.remove_callback_on_add_obj(name=callback_name)
                og.sim.remove_callback_on_remove_obj(name=callback_name)
                og.sim.remove_callback_on_system_init(name=callback_name)
                og.sim.remove_callback_on_system_clear(name=callback_name)

                # Remove all the additionally added objects
                objs_to_remove = tuple(env.scene.objects[n_scene_objects:])
                og.sim.batch_remove_objects(objs_to_remove)

                # Clear all systems
                for system in env.scene.active_systems.values():
                    env.scene.clear_system(system_name=system.name)
                clear_pu()
                og.sim.step()

                # Update the scene initial state to the original state
                env.scene.update_initial_file(scene_initial_file)

        except Exception as e:
            traceback_str = f"{traceback.format_exc()}"
            og.log.error(traceback_str)
            og.log.error(f"\n\nCaught exception sampling activity {activity} in scene {args.scene_model}:\n\n{e}\n\n")

            print("exception")
            breakpoint()

            if not args.offline:
                # Check if another thread succeeded already
                already_succeeded = worksheet.get(f"C{row}")
                if not (already_succeeded and already_succeeded[0] and str(already_succeeded[0][0]) == "1"):
                    # Clear the in_progress reservation and note the exception
                    cell_list = worksheet.range(f"B{row}:H{row}")
                    for cell, val in zip(cell_list, ("", 0, "", args.scene_model, USER, reason, traceback_str)):
                        cell.value = val
                    worksheet.update_cells(cell_list)

            try:
                # Stop sim, clear simulator, and re-create environment
                og.sim.stop()
                og.clear()
            except AttributeError as e:
                # This is the "GetPath" error that happens sporatically. It's benign, so we ignore it
                pass

            # env = create_env_with_stable_objects(cfg)
            # Make sure transition rules are loaded properly
            with gm.unlocked():
                gm.ENABLE_TRANSITION_RULES = True
                env = og.Environment(configs=copy.deepcopy(cfg))
                gm.ENABLE_TRANSITION_RULES = False

            if gm.HEADLESS:
                hide_all_lights()

            # After we load the robot, we do self.scene.reset() (one physics step) and then self.scene.update_initial_file().
            # We need to set all velocities to zero after this. Otherwise, the visual only objects will drift.
            for obj in env.scene.objects:
                obj.keep_still()
            env.scene.update_initial_file()

            # Store the initial state -- this is the safeguard to reset to!
            scene_initial_file = copy.deepcopy(env.scene._initial_file)
            og.sim.stop()

            n_scene_objects = len(env.scene.objects)

            # Set environment configuration after environment is loaded, because we will load the task
            env.task_config["type"] = "BehaviorTask"
            env.task_config["online_object_sampling"] = True

    print("Successful shutdown!")

    if not args.offline:
        # Record when we successfully complete all the activities
        worksheet.update_acell(f"W{scene_row}", 1)
        worksheet.update_acell(f"Y{scene_row}", "")


if __name__ == "__main__":
    main()

    # Shutdown at the end
    og.shutdown()
