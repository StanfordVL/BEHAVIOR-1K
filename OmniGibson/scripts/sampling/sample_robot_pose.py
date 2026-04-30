import argparse
import copy
import json
import os
import sys
import omnigibson as og
import torch as th
from pathlib import Path
from typing import Dict, List
from omnigibson.macros import gm
from omnigibson.objects.primitive_object import PrimitiveObject
from constants import DATASET_2026_PATH, TASK_CUSTOM_LIST_PATH
from gello.utils.og_teleop_utils import generate_robot_config
from utils import get_scene_model
from omnigibson.object_states import OnTop
import omnigibson.utils.transform_utils as T
from omnigibson.utils.bddl_utils import is_system_bddl_inst
from omnigibson.utils.python_utils import recursively_convert_to_torch

sys.path.append(str(Path(__file__).parent.parent))

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--activity",
    type=str,
    default=None,
    required=True,
    help="Activity to be sampled",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    default=None,
    help="Output directory for sampled tasks (default: gm.DATA_PATH/2026-challenge-task-instances)",
)
# Constants
MAX_ATTEMPTS = 10
MAX_UPRIGHT_TILT = 0.1

gm.ENABLE_TRANSITION_RULES = False


def find_given_tasks(data_dir, activities: List[str] = []) -> List[Dict]:
    """
    Find all instance files, partial json and full template json of the given task under folder data_dir.

    Returns:
        List of dictionaries containing task info: name, path, template files, instance dir
    """
    instance_dirs = list(data_dir.glob("*_instances"))
    template_files = list(data_dir.glob("*_template.json"))
    partial_files = list(data_dir.glob("*_template-partial_rooms.json"))

    tasks = []
    for activity in activities:
        instance_dirs = [d for d in instance_dirs if activity in d.name]
        template_files = [f for f in template_files if activity in f.name]
        partial_files = [f for f in partial_files if activity in f.name]
        # assert all threes are not empty list
        assert len(instance_dirs) > 0, f"No instance directories found in {data_dir} for activity {activity}"
        assert len(template_files) > 0, f"No template files found in {data_dir} for activity {activity}"
        assert len(partial_files) > 0, f"No partial template files found in {data_dir} for activity {activity}"
        tasks.append(
            {
                "name": activity,
                "path": data_dir,
                "template_file": template_files[0],
                "partial_file": partial_files[0] if partial_files else None,
                "instance_dir": instance_dirs[0],
                "tro_files": sorted(list(instance_dirs[0].glob("*-tro_state.json"))),
            }
        )
    print("tasks ", tasks)
    return tasks


def is_robot_pose_upright(quat):
    """
    Test if the given quaternion orientation is approximately upright (i.e. z-axis aligned) within a certain tilt threshold.
    """
    z_angle = T.z_angle_from_quat(quat)
    upright_euler = th.stack([quat.new_tensor(0.0), quat.new_tensor(0.0), z_angle])
    upright_quat = T.euler2quat(upright_euler)
    tilt_quat = T.quat_distance(quat, upright_quat)
    tilt_angle = 2.0 * th.atan2(th.norm(tilt_quat[:3]), th.abs(tilt_quat[3]))
    return tilt_angle <= MAX_UPRIGHT_TILT


def sample_robot_poses(env) -> Dict[str, List[Dict]]:
    """
    Sample a valid pose for the generic cylinder robot in the current environment state.

    Args:
        env: The OmniGibson environment already loaded and potentially reset to a specific state

    Returns:
        Dictionary mapping "robot" to list of pose dicts
    """
    # Find the reference object from initial conditions
    reference_object_name = None
    for cond in env.task.activity_initial_conditions:
        object_list = []
        for bddl_name in cond.get_relevant_objects():
            obj = env.task.object_scope.get(bddl_name, None)
            if obj is not None:
                object_list.append(obj.name)

        # Check if this condition involves a robot
        robot_objects = [obj_name for obj_name in object_list if "robot" in obj_name]
        if robot_objects:
            # Get the non-robot object from this condition
            non_robot_objects = [obj_name for obj_name in object_list if "robot" not in obj_name]
            reference_object_name = non_robot_objects[0]  # Take the first non-robot object
            print(f"    Found reference object for robot positioning: {reference_object_name}")
            break

    assert reference_object_name is not None, "No reference object found in initial conditions for robot pose sampling"
    reference_object = env.scene.object_registry("name", reference_object_name)

    # Create the generic cylinder object

    cylinder = PrimitiveObject(
        name="generic_cylinder",
        primitive_type="Cylinder",
        height=2.0,
        radius=0.5,
        fixed_base=False,
    )

    # Add cylinder to scene. Stopping the scene and playing it again causes a reset, so dump the state first.
    state = og.sim.dump_state()
    og.sim.stop()
    env.scene.add_object(cylinder)
    og.sim.play()
    og.sim.load_state(state)

    # Sample pose using OnTop state
    sampled_cylinder_pose = None
    for _ in range(MAX_ATTEMPTS):
        trial_success = cylinder.states[OnTop].set_value(reference_object, True)
        if trial_success:
            # Settle cylinder
            for _ in range(10):
                og.sim.step()
            sampled_cylinder_pose = cylinder.get_position_orientation()
            if not is_robot_pose_upright(sampled_cylinder_pose[1]):
                sampled_cylinder_pose = None
                continue
            break

    assert sampled_cylinder_pose is not None, "Failed to sample valid upright cylinder pose"
    # set z to the cylinder base
    sampled_cylinder_pose[0][2] -= cylinder.height / 2
    robot_poses = {
        "robot": [{"position": sampled_cylinder_pose[0].tolist(), "orientation": sampled_cylinder_pose[1].tolist()}]
    }

    # Remove cylinder from scene
    env.scene.remove_object(cylinder)

    return robot_poses


def process_task(task_info: Dict):
    """
    Process a single task: sample cylinder pose and update all files.

    Args:
        task_info: Dictionary containing task information
    """
    print(f"Processing task: {task_info['name']}")

    # Load template file
    with open(task_info["template_file"], "r") as f:
        template_data = json.load(f)

    # Create environment configuration using template file as scene
    cfg = {
        "env": {
            "action_frequency": 30,
            "rendering_frequency": 30,
            "physics_frequency": 120,
        },
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": template_data["init_info"]["args"]["scene_model"],
            "scene_file": str(task_info["template_file"]),
            "include_robots": False,
        },
        "task": {
            "type": "BehaviorTask",
            "activity_name": task_info["name"],
            "activity_definition_id": 0,
            "activity_instance_id": 0,
            "online_object_sampling": False,
            "debug_object_sampling": False,
            "highlight_task_relevant_objects": False,
            "termination_config": {
                "max_steps": 50000,
            },
            "reward_config": {
                "r_potential": 1.0,
            },
            "include_obs": False,
            "use_presampled_robot_pose": False,
        },
        "robots": generate_robot_config(
            robot_type="r1pro",
            robot_name="robot",
        ),
    }
    # Create environment once for this task
    env = og.Environment(configs=cfg)

    # Sample cylinder pose for the template instance
    template_robot_poses = sample_robot_poses(env)

    # 2. Update template file
    template_data["metadata"]["task"]["robot_poses"] = template_robot_poses

    # Save updated template
    with open(task_info["template_file"], "w") as f:
        json.dump(template_data, f, indent=4)

    # 3. Update partial template file if it exists
    if task_info["partial_file"] is not None:
        with open(task_info["partial_file"], "r") as f:
            partial_data = json.load(f)

        partial_data["metadata"]["task"]["robot_poses"] = template_robot_poses

        with open(task_info["partial_file"], "w") as f:
            json.dump(partial_data, f, indent=4)

    # 4. Update all TRO files - each needs its own cylinder pose sampling
    for i, tro_file in enumerate(task_info["tro_files"]):
        print(f"    Processing TRO {i + 1}/{len(task_info['tro_files'])}: {tro_file.name}")

        env.scene.reset()
        with open(tro_file, "r") as f:
            tro_data = json.load(f)
            tro_torch_state = recursively_convert_to_torch(copy.deepcopy(tro_data))

        # Reset environment to this TRO instance state
        for bddl_name, obj_state in tro_torch_state.items():
            if "agent" in bddl_name or "robot_poses" in bddl_name:
                continue
            env.task.object_scope[bddl_name].load_state(obj_state, serialized=False)
        for _ in range(25):
            og.sim.step_physics()
            for bddl_name, entity in env.task.object_scope.items():
                if not is_system_bddl_inst(bddl_name) and entity is not None:
                    entity.keep_still()

        # Sample cylinder pose for this specific TRO instance
        tro_robot_poses = sample_robot_poses(env)

        tro_data["robot_poses"] = tro_robot_poses

        with open(tro_file, "w") as f:
            json.dump(tro_data, f, indent=4)

    print(f"  Added pose for: {list(template_robot_poses.keys())}")
    print(f"  Updated template, partial template, and {len(task_info['tro_files'])} TRO files")


def main():
    """
    Main function to process all tasks with instance directories.
    """
    args = parser.parse_args()

    with open(TASK_CUSTOM_LIST_PATH) as f:
        task_custom_lists = json.load(f)
    scene_model = get_scene_model(task_custom_lists[args.activity])

    if args.output_dir is None:
        args.output_dir = os.path.join(DATASET_2026_PATH, "scenes", scene_model, "json")

    # Find tasks in output_dir
    tasks = find_given_tasks(Path(args.output_dir), [args.activity])

    print(f"Found {len(tasks)} tasks with instance directories")

    # Process each task
    for i, task_info in enumerate(tasks):
        print(f"\n[{i + 1}/{len(tasks)}] ", end="")
        process_task(task_info)
        if i < len(tasks) - 1:
            og.clear()

    og.shutdown()

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
