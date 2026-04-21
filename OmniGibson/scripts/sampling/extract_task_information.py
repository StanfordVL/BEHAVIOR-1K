import os
import json
import math
import yaml
import re
from constants import DATASET_2026_PATH


def euler_to_quat(euler):
    """Convert XYZ euler angles (radians) to [x, y, z, w] quaternion."""
    cx, cy, cz = math.cos(euler[0] / 2), math.cos(euler[1] / 2), math.cos(euler[2] / 2)
    sx, sy, sz = math.sin(euler[0] / 2), math.sin(euler[1] / 2), math.sin(euler[2] / 2)
    return [
        sx * cy * cz + cx * sy * sz,
        cx * sy * cz - sx * cy * sz,
        cx * cy * sz + sx * sy * cz,
        cx * cy * cz - sx * sy * sz,
    ]


def main():
    scenes_dir = os.path.join(DATASET_2026_PATH, "scenes")

    # Create a new empty dictionary to store tasks
    tasks_data = {}

    # Traverse scenes/<scene_model>/json/<task_instances_dir>/
    for scene_model in os.listdir(scenes_dir):
        json_dir = os.path.join(scenes_dir, scene_model, "json")
        if not os.path.isdir(json_dir):
            continue

        for task_instances_dir in os.listdir(json_dir):
            task_path = os.path.join(json_dir, task_instances_dir)
            if not os.path.isdir(task_path):
                continue

            # Dir name format: {scene_model}_task_{task_name}_instances
            prefix = f"{scene_model}_task_"
            suffix = "_instances"
            if not (task_instances_dir.startswith(prefix) and task_instances_dir.endswith(suffix)):
                continue
            task_name = task_instances_dir[len(prefix) : -len(suffix)]

            # Get all template JSON files (ending with _template-tro_state.json)
            json_files = [f for f in os.listdir(task_path) if f.endswith("_template-tro_state.json")]

            if not json_files:
                print(f"No JSON files found in task directory: {task_instances_dir}")
                continue

            if task_name not in tasks_data:
                tasks_data[task_name] = {}

            for json_file in json_files:
                json_file_path = os.path.join(task_path, json_file)

                with open(json_file_path, "r") as f:
                    json_content = json.load(f)

                # Filename format: {scene_model}_task_{task_name}_0_{instance}_template-tro_state.json
                filename_pattern = (
                    rf"{re.escape(scene_model)}_task_{re.escape(task_name)}_0_(\d+)_template-tro_state\.json"
                )
                match = re.match(filename_pattern, json_file)
                if not match:
                    print(f"Could not extract instance number from filename: {json_file}")
                    continue
                instance_number = int(match.group(1))

                # Extract robot pose from robot_poses key
                robot_pose = json_content["robot_poses"]["robot"][0]
                robot_start_position = robot_pose["position"]
                robot_start_orientation = robot_pose["orientation"]

                tasks_data[task_name][instance_number] = {
                    "scene_model": scene_model,
                    "robot_start_position": robot_start_position,
                    "robot_start_orientation": robot_start_orientation,
                }

                print(f"Processed file: {json_file} from directory: {task_instances_dir}")
                print(f"  Task: {task_name}")
                print(f"  Instance: {instance_number}")
                print(f"  Scene model: {scene_model}")
                print(f"  Robot start position: {robot_start_position}")
                print(f"  Robot start orientation: {robot_start_orientation}")
                print("-" * 50)

            # Instance 0 lives in the parent json/ folder as _0_0_template.json (old format, no robot_poses key)
            template_file = os.path.join(json_dir, f"{scene_model}_task_{task_name}_0_0_template.json")
            if os.path.exists(template_file):
                with open(template_file, "r") as f:
                    tmpl = json.load(f)
                robot_name = tmpl["metadata"]["task"]["inst_to_name"]["agent.n.01_1"]
                obj_state = tmpl["state"]["registry"]["object_registry"][robot_name]
                root_pos = obj_state["root_link"]["pos"]
                base_joints = obj_state["joint_pos"]
                robot_start_position = [root_pos[i] + base_joints[i] for i in range(3)]
                robot_start_orientation = euler_to_quat(base_joints[3:6])
                tasks_data[task_name][0] = {
                    "scene_model": scene_model,
                    "robot_start_position": robot_start_position,
                    "robot_start_orientation": robot_start_orientation,
                }
                print(f"Processed instance 0 from: {os.path.basename(template_file)}")
                print(f"  Robot start position: {robot_start_position}")
                print(f"  Robot start orientation: {robot_start_orientation}")
                print("-" * 50)
            else:
                print(f"Warning: no instance 0 template found for {task_name} in {scene_model}")

    # Write the data to the YAML file (completely overwriting it)
    yaml_file = os.path.join(DATASET_2026_PATH, "metadata", "available_tasks.yaml")
    with open(yaml_file, "w") as f:
        yaml.dump(tasks_data, f, default_flow_style=False)

    # Count total instances
    total_instances = sum(len(instances) for instances in tasks_data.values())
    print(
        f"Created new {yaml_file} with information from {len(tasks_data)} tasks and {total_instances} total instances"
    )


if __name__ == "__main__":
    main()
