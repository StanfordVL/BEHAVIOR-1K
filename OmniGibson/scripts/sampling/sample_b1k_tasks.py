import logging
import os
import copy
import argparse
import omnigibson as og
from omnigibson.macros import gm, macros
import json
from omnigibson.objects import DatasetObject
from omnigibson.object_states import Contains
from omnigibson.tasks import BehaviorTask
from omnigibson.utils.python_utils import clear as clear_pu
from omnigibson.utils.constants import PrimType
from omnigibson.utils.bddl_utils import get_knowledge_base
from omnigibson.utils.ui_utils import create_module_logger
from utils import (
    get_predicates,
    get_valid_tasks,
    hide_all_lights,
    UNSUPPORTED_PREDICATES,
    validate_task,
    get_scene_model,
    resolve_scene_model,
)
from constants import DATASET_2026_PATH, TASK_CUSTOM_LIST_PATH
from postprocess_sampled_task import postprocess_task
import numpy as np

log = create_module_logger(module_name="sample_b1k_tasks")
log.setLevel(logging.INFO)


# task_custom_lists.json always takes precedence.
assert os.path.exists(TASK_CUSTOM_LIST_PATH), f"task_custom_lists.json not found: {TASK_CUSTOM_LIST_PATH}"
with open(TASK_CUSTOM_LIST_PATH, "r") as f:
    TASK_CUSTOM_LISTS = json.load(f)


parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--activity",
    type=str,
    required=True,
    help="Activity to be sampled",
)
parser.add_argument(
    "-r",
    "--room_types",
    type=str,
    default=None,
    help="room types to be loaded, if specified. This should be a comma-delimited list of desired room types. Otherwise, will try to load all room types in this scene",
)
parser.add_argument(
    "-w",
    "--overwrite",
    action="store_true",
    help="If set, will overwrite any existing tasks that are found. Otherwise, will skip.",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    default=None,
    help="Output directory for sampled tasks (default: gm.DATA_PATH/2026-challenge-task-instances)",
)

# gm.HEADLESS = False
gm.USE_GPU_DYNAMICS = False
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

    scene_model_key = get_scene_model(TASK_CUSTOM_LISTS[args.activity])
    scene_model = resolve_scene_model(scene_model_key, os.path.join(DATASET_2026_PATH, "scenes"))

    if args.output_dir is None:
        args.output_dir = os.path.join(DATASET_2026_PATH, "scenes", scene_model, "json")

    # If we want to create a stable scene config, do that now
    default_scene_fpath = os.path.join(DATASET_2026_PATH, "scenes", scene_model, "json", f"{scene_model}_stable.json")
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
            "scene_model": scene_model,
            "seg_map_resolution": 0.1,
        },
        "robots": [
            {
                "type": "R1Pro",
                "obs_modalities": [],
                "default_reset_mode": "tuck",
                "position": np.ones(3) * -50.0,
            },
        ],
    }

    activity = args.activity

    # Check if activity is valid
    valid_tasks = get_valid_tasks()
    if activity not in valid_tasks:
        log.error(f"Activity {activity} not in valid tasks!")
        return

    log.info(f"Sampling activity: {activity}...")

    # Currently our sampling script always samples partial rooms so we specify there to delineate between full
    # scene templates
    task_suffix = "partial_rooms"
    if args.room_types is not None:
        cfg["scene"]["load_room_types"] = args.room_types.split(",")
    else:
        cfg["scene"]["load_room_types"] = TASK_CUSTOM_LISTS[activity]["room_types"]

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

    should_sample, success, reason = True, False, ""

    # Skip any with unsupported predicates, but still record the reason why we can't sample
    task_obj = get_knowledge_base().get_task(f"{activity}-0")
    conditions, object_scope, inroom_assignments = task_obj.parse_base_scope()
    all_predicates = set(
        get_predicates(conditions.parsed_initial_conditions) + get_predicates(conditions.parsed_goal_conditions)
    )
    unsupported_predicates = set.intersection(all_predicates, UNSUPPORTED_PREDICATES)
    if len(unsupported_predicates) > 0:
        should_sample = False
        reason = f"Unsupported predicate(s): {unsupported_predicates}"

    env.task_config["activity_name"] = activity
    if activity in TASK_CUSTOM_LISTS and scene_model_key in TASK_CUSTOM_LISTS[activity]:
        whitelist = TASK_CUSTOM_LISTS[activity][scene_model_key]["whitelist"]
        blacklist = TASK_CUSTOM_LISTS[activity][scene_model_key]["blacklist"]
    else:
        whitelist, blacklist = None, None
    env.task_config["sampling_whitelist"] = whitelist
    env.task_config["sampling_blacklist"] = blacklist
    log.info(f"white_list: {whitelist}")
    log.info(f"black_list: {blacklist}")
    assert whitelist is not None, "whitelist should not be None for manual sampling"
    BehaviorTask.get_cached_activity_scene_filename(
        scene_model=scene_model,
        activity_name=activity,
        activity_definition_id=0,
        activity_instance_id=0,
    )

    # Make sure sim is stopped
    assert og.sim.is_stopped()

    # Attempt to sample
    if should_sample:
        active_room_instances = env.scene.load_room_instances
        log.info(f"relevant room instances: {active_room_instances}")
        relevant_room_instances = set(active_room_instances) if active_room_instances is not None else None
        for obj in env.scene.objects:
            if isinstance(obj, DatasetObject):
                active = (
                    relevant_room_instances is None
                    or len(relevant_room_instances.intersection(obj.in_rooms)) > 0
                    or obj.category in {"floors", "walls"}
                )
                obj.visual_only = not active
                obj.visible = active

        og.log.info(f"Sampling task: {activity}")
        original_task_cfg = env.task_config
        original_task_cfg["use_presampled_robot_pose"] = False
        env._load_task(original_task_cfg)
        assert og.sim.is_stopped()
        success, feedback = env.task.feedback is None, env.task.feedback

        if not success:
            raise ValueError(f"Initial task feedback not None: {feedback}")

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
        assert validate_task(
            env.task,
            task_scene_dict,
            default_scene_dict,
            active_room_instances=active_room_instances,
        )
        # BREAKPOINT: Validation failed - inspect the task state to understand why
        # At this breakpoint, you can:
        # - Run: for _ in range(1000): og.sim.render()
        # - Move the camera around to inspect objects and their states
        # - Check env.task for task details
        # - Examine feedback variable for the validation error message

        env.scene.load_state(task_final_state)
        env.scene.update_initial_file()
        log.info("\n\nsampling succeed! Please continue to save task and scene reload...\n\n")
        # BREAKPOINT: Sampling succeeded - inspect the final task state before saving
        # At this breakpoint, you can:
        # - Run: for _ in range(1000): og.sim.render()
        # - Move the camera around to visually verify the sampled task looks correct
        # After inspection, continue to save the task to disk
        breakpoint()
        save_dir = os.path.join(args.output_dir)
        os.makedirs(save_dir, exist_ok=True)
        env.task.save_task(
            env=env,
            save_dir=save_dir,
            override=args.overwrite,
            task_relevant_only=False,
            suffix=task_suffix,
        )
        postprocess_task(save_dir, scene_model, activity, overwrite=args.overwrite)
        og.sim.stop()

    assert og.sim.is_stopped()

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
        og.sim.play()
        og.sim.step()

        # Update the scene initial state to the original state
        env.scene.update_initial_file(scene_initial_file)

        # Stop sim, clear simulator
        og.sim.stop()
        og.clear()

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

    log.info(f"Finished sampling activity: {activity} with success: {success} and reason: {reason}")


if __name__ == "__main__":
    main()

    # Shutdown at the end
    og.shutdown()
