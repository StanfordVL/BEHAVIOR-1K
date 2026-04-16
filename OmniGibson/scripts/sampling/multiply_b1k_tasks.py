import os
import argparse
import omnigibson as og
from omnigibson.macros import gm, macros
import json
from omnigibson.objects import DatasetObject
from omnigibson.utils.asset_utils import get_dataset_path
import numpy as np
from utils import validate_task

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--activity",
    type=str,
    default=None,
    required=True,
    help="Activity to be sampled.",
)
parser.add_argument("-s", "--scene_model", type=str, default=None, required=True, help="Scene model to sample tasks in")
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Instance ID to use as seed",
)
parser.add_argument(
    "--start_idx",
    type=int,
    default=1,
    help="Instance ID to start (inclusive)",
)
parser.add_argument(
    "--end_idx",
    type=int,
    default=10,
    help="Instance ID to end (inclusive)",
)
parser.add_argument(
    "--partial_save",
    action="store_true",
    help="Whether to only save the task-relevant object scope states instead of the entire scene json",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    default=None,
    help="Output directory for sampled tasks (default: gm.DATA_PATH/2026-challenge-task-instances/<activity>)",
)

task_custom_list_path = os.path.join(gm.DATA_PATH, "2026-challenge-task-instances", "task_custom_lists.json")
assert os.path.exists(task_custom_list_path), f"task_custom_lists.json not found: {task_custom_list_path}"
with open(task_custom_list_path, "r") as f:
    TASK_CUSTOM_LISTS = json.load(f)

gm.HEADLESS = False
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = True

macros.systems.micro_particle_system.MICRO_PARTICLE_SYSTEM_MAX_VELOCITY = 0.5
macros.systems.macro_particle_system.MACRO_PARTICLE_SYSTEM_MAX_DENSITY = 200.0
# macros.prims.entity_prim.DEFAULT_SLEEP_THRESHOLD = 0.0
macros.utils.object_state_utils.DEFAULT_HIGH_LEVEL_SAMPLING_ATTEMPTS = 5
macros.utils.object_state_utils.DEFAULT_LOW_LEVEL_SAMPLING_ATTEMPTS = 5


def main():
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(gm.DATA_PATH, "2026-challenge-task-instances", args.scene_model, "json")

    task_scene_file = os.path.join(args.output_dir, f"{args.scene_model}_task_{args.activity}_0_0_template.json")
    assert os.path.exists(task_scene_file), "Did not find task scene template json at expected path: {}".format(
        task_scene_file
    )
    # Define the configuration to load -- we'll use a Fetch
    cfg = {
        # Use default frequency
        "env": {
            "action_frequency": 30,
            "physics_frequency": 120,
        },
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": args.scene_model,
            "scene_file": task_scene_file,
            "seg_map_resolution": 0.1,
            "load_room_types": TASK_CUSTOM_LISTS[args.activity]["room_types"],
        },
        "robots": [
            {
                "type": "R1Pro",
                "obs_modalities": [],
                "default_reset_mode": "untuck",
                "position": np.ones(3) * -50.0,
            },
        ],
        "task": {
            "type": "BehaviorTask",
            "online_object_sampling": False,
            "activity_name": args.activity,
            "activity_instance_id": args.seed,
            "use_presampled_robot_pose": False,
        },
    }
    env = og.Environment(cfg)

    # Define where to save instances
    save_dir = os.path.join(args.output_dir, f"{env.task.scene_name}_task_{args.activity}_instances")

    # If we want to create a stable scene config, do that now
    default_scene_fpath = os.path.join(
        get_dataset_path("behavior-1k-assets"), "scenes", args.scene_model, "json", f"{args.scene_model}_stable.json"
    )
    # Get the default scene instance
    assert os.path.exists(default_scene_fpath), "Did not find default stable scene json!"
    with open(default_scene_fpath, "r") as f:
        default_scene_dict = json.load(f)

    # Needed for _sample_initial_conditions_final()
    env.task.sampler._compiled_task = env.task.compiled_task
    env.task.sampler._parse_inroom_object_room_assignment()
    env.task.sampler._build_sampling_order()

    # Clear all the system particles
    for system in env.scene.active_systems.values():
        system.remove_all_particles()

    og.sim.step()

    # Store the state without any particles
    initial_state = og.sim.dump_state()

    num_trials = 50
    for activity_instance_id in range(args.start_idx, args.end_idx + 1):
        success = False
        for i in range(num_trials):
            og.sim.load_state(initial_state)
            og.sim.step()

            # Will sample new particles to satisfy states like Filled
            error_msg = env.task.sampler._sample_initial_conditions_final()

            if error_msg is not None:
                print(f"instance {activity_instance_id} trial {i} sampling failed: {error_msg}")
                continue

            for _ in range(10):
                og.sim.step()

            for obj in env.task.object_scope.values():
                if isinstance(obj, DatasetObject):
                    obj.keep_still()

            for _ in range(10):
                og.sim.step()

            task_final_state = env.scene.dump_state()
            task_scene_dict = {"state": task_final_state}
            if not validate_task(env.task, task_scene_dict, default_scene_dict):
                print(f"instance {activity_instance_id} trial {i} validation failed")
                continue

            env.scene.load_state(task_final_state)
            env.scene.update_initial_file()
            print(f"instance {activity_instance_id} trial {i} succeeded.")

            env.task.activity_instance_id = activity_instance_id
            env.task.save_task(env=env, save_dir=save_dir, override=True, task_relevant_only=args.partial_save)
            print(f"instance {activity_instance_id} trial {i} saved")
            success = True
            break
        if not success:
            raise ValueError(f"instance {activity_instance_id} failed all {num_trials} trials")

    print(f"\n\nMultiply task for {args.activity} successful! Saved to {save_dir}.\n\n")
    og.shutdown()


if __name__ == "__main__":
    main()
