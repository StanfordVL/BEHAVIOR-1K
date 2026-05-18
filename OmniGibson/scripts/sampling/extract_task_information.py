import os
import json
import yaml
import torch as th
import omnigibson.utils.transform_utils as T
from constants import DATASET_2026_PATH


def main():
    scenes_dir = os.path.join(DATASET_2026_PATH, "scenes")

    # Create a new empty dictionary to store tasks
    tasks_data = {}

    # Traverse scenes/<scene_model>/json
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
                robot_start_orientation = T.euler2quat(th.tensor(base_joints[3:6])).tolist()
                tasks_data[task_name] = {
                    0: {
                        "scene_model": scene_model,
                        "robot_start_position": robot_start_position,
                        "robot_start_orientation": robot_start_orientation,
                    }
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
