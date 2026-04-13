import os
import json
import torch as th
import numpy as np
from pathlib import Path
from typing import Dict, List
import copy
import argparse
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.python_utils import recursively_convert_to_torch
import omnigibson.utils.transform_utils as T
from omnigibson.object_states import OnTop
from omnigibson.utils.asset_utils import get_dataset_path
import sys

sys.path.append(str(Path(__file__).parent.parent))

parser = argparse.ArgumentParser()
parser.add_argument("--scene_model", type=str, default=None, help="Scene model to sample tasks in")
parser.add_argument(
    "--activities",
    type=str,
    default=None,
    help="Activity to be sampled, if specified. This should be a comma-delimited list of desired activities. Otherwise, will try to sample all tasks in this scene",
)
parser.add_argument(
    "--data_path",
    type=str,
    default="2025-challenge-task-instances",
    help="Where the instance folder, partial json and full json file is stored",
)
# Constants
SAMPLED_TASK_DIR = os.path.join(get_dataset_path("2025-challenge-task-instances"), "scenes")
SUPPORTED_ROBOTS = ["R1", "Fetch", "Tiago", "Stretch"]  # All mobile manipulators
MAX_ATTEMPTS = 10

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


def find_tasks_with_instances() -> List[Dict]:
    """
    Find all tasks that have ..._instances directories under the directory SAMPLED_TASK_DIR.

    The task instance structure is:
        scenes/
            <scene_name>/
                json/
                    <scene_name>_task_<activity_name>_instances/
                    <scene_name>_task_<activity_name>_0_0_template.json
                    <scene_name>_task_<activity_name>_0_0_template-partial_rooms.json

    Returns:
        List of dictionaries containing task info: name, path, template files, instance dir
    """
    tasks = []

    # Iterate through scene directories (e.g., house_double_floor_lower)
    for scene_dir in SAMPLED_TASK_DIR.iterdir():
        if not scene_dir.is_dir():
            continue

        # Look for the json subdirectory
        json_dir = scene_dir / "json"
        if not json_dir.exists() or not json_dir.is_dir():
            continue

        # Look for instance directories in the json subdirectory
        instance_dirs = list(json_dir.glob("*_instances"))
        if not instance_dirs:
            continue

        # Process each instance directory
        for instance_dir in instance_dirs:
            # Extract the task prefix (everything before _instances)
            task_prefix = instance_dir.name.replace("_instances", "")

            # Find corresponding template files
            template_pattern = f"{task_prefix}_*_template.json"
            partial_pattern = f"{task_prefix}_*_template-partial_rooms.json"

            template_files = list(json_dir.glob(template_pattern))
            partial_files = list(json_dir.glob(partial_pattern))

            if template_files:
                # Extract activity name from the task prefix
                # Format: <scene_name>_task_<activity_name>
                # We want just the activity name part
                parts = task_prefix.split("_task_")
                activity_name = parts[1] if len(parts) > 1 else task_prefix

                tasks.append(
                    {
                        "name": activity_name,
                        "path": json_dir,
                        "template_file": template_files[0],
                        "partial_file": partial_files[0] if partial_files else None,
                        "instance_dir": instance_dir,
                        "tro_files": sorted(list(instance_dir.glob("*-tro_state.json"))),
                    }
                )

    return tasks


def sample_robot_poses(env, existing_r1pro_pose: Dict) -> Dict[str, List[Dict]]:
    """
    Sample valid poses for all supported robots (except R1Pro) in the current environment state.

    Args:
        env: The OmniGibson environment already loaded and potentially reset to a specific state
        existing_r1pro_pose: The existing R1Pro pose dict with 'position' and 'orientation'

    Returns:
        Dictionary mapping robot names to list of pose dicts
    """
    robot_poses = {"R1Pro": [existing_r1pro_pose]}

    # Find the reference object from initial conditions
    reference_object_name = None
    for cond in env.task.activity_initial_conditions:
        object_list = [obj.name for obj in cond.get_relevant_objects() if obj.exists]

        # Check if this condition involves a robot
        robot_objects = [obj_name for obj_name in object_list if "robot" in obj_name]
        if robot_objects:
            # Get the non-robot object from this condition
            # This list looks something like ['floors_ulujpr_0', 'robot_fetch']
            non_robot_objects = [obj_name for obj_name in object_list if "robot" not in obj_name]
            reference_object_name = non_robot_objects[0]  # Take the first non-robot object
            print(f"    Found reference object for robot positioning: {reference_object_name}")
            break

    assert reference_object_name is not None, "No reference object found in initial conditions for robot pose sampling"
    reference_object = env.scene.object_registry("name", reference_object_name)

    for robot in env.robots:
        initial_pose = robot.get_position_orientation()
        robot.reset()
        sampled_robot_base_pose = None
        for _ in range(MAX_ATTEMPTS):
            trial_success = robot.states[OnTop].set_value(reference_object, True)
            if trial_success:
                # Settle robot
                for _ in range(10):
                    og.sim.step()
                sampled_robot_base_pose = robot.get_position_orientation()
                break
        assert sampled_robot_base_pose is not None
        robot_poses[robot.model_name] = [
            {"position": sampled_robot_base_pose[0].tolist(), "orientation": sampled_robot_base_pose[1].tolist()}
        ]
        robot.set_position_orientation(*initial_pose)
        og.sim.step()

    return robot_poses


def generate_robot_configs():
    """
    Generate robot configurations for all supported robot types.
    Each robot is placed at a remote location to avoid interfering with the scene.

    Returns:
        List of robot configuration dictionaries
    """
    robot_configs = []

    for i, robot_type in enumerate(SUPPORTED_ROBOTS):
        robot_config = {
            "type": robot_type,
            "name": f"robot_{robot_type.lower()}",  # e.g., robot_r1pro, robot_fetch
            "action_normalize": False,
            "self_collisions": True,
            "obs_modalities": [],
            # Place robots in a remote location, spaced apart
            "position": [50.0, i * 2.0, 10.0],  # Space robots 2m apart along Y axis
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "default_reset_mode": "untuck",
        }
        robot_configs.append(robot_config)

    return robot_configs


def process_task(task_info: Dict):
    """
    Process a single task: extract R1Pro pose, sample other robot poses, and update all files.

    Args:
        task_info: Dictionary containing task information
    """
    print(f"Processing task: {task_info['name']}")

    # 1. Extract R1Pro pose from the template file
    with open(task_info["template_file"], "r") as f:
        template_data = json.load(f)

    # Get robot name from metadata
    robot_bddl_name = "agent.n.01_1"  # Default agent name
    inst_to_name = template_data["metadata"].get("inst_to_name", {})
    robot_name = inst_to_name.get(robot_bddl_name, None)

    # Extract R1Pro pose from state registry
    registry = template_data["state"].get("registry", {})
    object_registry = registry.get("object_registry", {})
    robot_state = object_registry[robot_name]
    # Extract position and orientation
    # Position: joint_pos[:3] + root_link.pos
    robot_pos = np.array(robot_state["joint_pos"][:3]) + np.array(robot_state["root_link"]["pos"])
    # Orientation: Only Rz component from joint_pos[5]
    robot_quat = T.euler2quat(th.tensor([0, 0, robot_state["joint_pos"][5]]))
    r1pro_pose_dict = {"position": robot_pos.tolist(), "orientation": robot_quat.numpy().tolist()}

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
            # "scene_file": str(task_info['template_file']),  # Use template file as scene
            "include_robots": False,  # Don't load robot from scene file
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
        "robots": generate_robot_configs(),  # Add all robot configs
    }

    # Create environment once for this task
    env = og.Environment(configs=cfg)

    # Sample robot poses for the template instance
    template_robot_poses = sample_robot_poses(env, r1pro_pose_dict)

    # 2. Update template file
    # template_data['metadata']=template_data['metadata']['robot_poses']
    template_data["metadata"]["robot_poses"] = template_robot_poses

    # Remove robot from file
    template_data["state"]["registry"]["object_registry"].pop(robot_name, None)
    template_data["objects_info"]["init_info"].pop(robot_name, None)
    template_data["metadata"]["inst_to_name"]["agent.n.01_1"] = "robot"

    # Save updated template
    # task_info['template_file']
    with open("template_file.json", "w") as f:
        json.dump(template_data, f, indent=4)

    # 3. Update partial template file if it exists
    if task_info["partial_file"] is not None:
        with open(task_info["partial_file"], "r") as f:
            partial_data = json.load(f)

        # Add robot_poses to metadata (use same poses as template since it's the same instance)
        partial_data["metadata"]["robot_poses"] = template_robot_poses

        # Remove robot entries similar to template
        partial_data["state"]["registry"]["object_registry"].pop(robot_name, None)
        partial_data["objects_info"]["init_info"].pop(robot_name, None)
        partial_data["metadata"]["inst_to_name"]["agent.n.01_1"] = "robot"
        # task_info['partial_file']
        with open("partial_file.json", "w") as f:
            json.dump(partial_data, f, indent=4)

    # 4. Update all TRO files - each needs its own robot pose sampling
    for i, tro_file in enumerate(task_info["tro_files"]):
        print(f"    Processing TRO {i+1}/{len(task_info['tro_files'])}: {tro_file.name}")

        env.scene.reset()
        with open(tro_file, "r") as f:
            tro_data = json.load(f)
            tro_torch_state = recursively_convert_to_torch(copy.deepcopy(tro_data))

        # Extract R1Pro pose from this specific TRO
        tro_r1pro_pose = None
        flag = False
        for key, value in list(tro_data.items()):
            if "agent" in key:
                flag = True
                # Extract pose from this agent entry
                robot_pos = np.array(value["joint_pos"][:3]) + np.array(value["root_link"]["pos"])
                robot_quat = T.euler2quat(th.tensor([0, 0, value["joint_pos"][5]]))
                tro_r1pro_pose = {"position": robot_pos.tolist(), "orientation": robot_quat.numpy().tolist()}
                # Remove the agent entry
                del tro_data[key]
                break
        if not flag:
            continue
        # TODO: Reset environment to this TRO instance state
        for bddl_name, obj_state in tro_torch_state.items():
            if "agent" in bddl_name:
                continue
            env.task.object_scope[bddl_name].load_state(obj_state, serialized=False)
        for _ in range(25):
            og.sim.step_physics()
            for entity in env.task.object_scope.values():
                if not entity.is_system and entity.exists:
                    entity.keep_still()

        # Sample robot poses for this specific TRO instance
        tro_robot_poses = sample_robot_poses(env, tro_r1pro_pose)

        # Add robot_poses to TRO data
        tro_data["robot_poses"] = tro_robot_poses

        with open(tro_file, "w") as f:
            json.dump(tro_data, f, indent=4)

    # Clean up environment
    og.clear()

    print(f"  Updated template, partial template, and {len(task_info['tro_files'])} TRO files")
    print(f"  Added poses for robots: {list(template_robot_poses.keys())}")


def main():
    """
    Main function to process all tasks with instance directories.
    """
    args = parser.parse_args()
    if args.activities is not None:
        activities = args.activities.split(",")
        path = os.path.join(get_dataset_path(args.data_path), "scenes", args.scene_model, "json")
        tasks = find_given_tasks(Path(path), activities)
    else:
        # Find all tasks with instance directories in default task dir
        tasks = find_tasks_with_instances()
    print(f"Found {len(tasks)} tasks with instance directories")

    # Process each task
    for i, task_info in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] ", end="")
        process_task(task_info)

    og.shutdown()

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
