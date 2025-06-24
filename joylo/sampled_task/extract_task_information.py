import os
import json
import yaml
import re
import torch as th
import omnigibson.utils.transform_utils as T

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get all subdirectories in the script's directory
    task_dirs = [d for d in os.listdir(script_dir) if os.path.isdir(os.path.join(script_dir, d))]
    
    # Create a new empty dictionary to store tasks
    tasks_data = {}
    
    # Process each task directory
    for task_dir in task_dirs:
        task_name = task_dir  # The directory name is the task name
        task_path = os.path.join(script_dir, task_dir)
        
        # Get all JSON files in the task directory, excluding those with a dash '-' in their name (partial/aggregate)
        json_files = [f for f in os.listdir(task_path) if f.endswith('.json') and '-' not in f]
        
        if not json_files:
            print(f"No JSON files found in task directory: {task_dir}")
            continue
        
        # Initialize the task entry in tasks_data if it doesn't exist
        if task_name not in tasks_data:
            tasks_data[task_name] = {}
        
        # Process each JSON file in the task directory
        for json_file in json_files:
            json_file_path = os.path.join(task_path, json_file)
            
            # Read JSON content
            with open(json_file_path, 'r') as f:
                json_content = json.load(f)
            
            # Extract scene_model
            scene_model = json_content['init_info']['args']['scene_model']
            
            # Extract instance number from filename using regex
            # Filename format: {scene_model}_task_{task_name}_0_{instance}_template.json
            filename_pattern = f"{scene_model}_task_{task_name}_0_(\d+)_template.json"
            match = re.match(filename_pattern, json_file)
            if match:
                instance_number = int(match.group(1))
            else:
                print(f"Could not extract instance number from filename: {json_file}")
                continue
            
            # Extract robot information
            robot_name = json_content['metadata']['task']['inst_to_name']['agent.n.01_1']
            robot_root_link_position = json_content['state']['registry']['object_registry'][robot_name]['root_link']['pos']
            robot_base_link_position = json_content['state']['registry']['object_registry'][robot_name]['joint_pos'][:3]
            
            # Calculate robot_start_position
            robot_start_position = [
                robot_root_link_position[0] + robot_base_link_position[0],
                robot_root_link_position[1] + robot_base_link_position[1],
                robot_root_link_position[2] + robot_base_link_position[2]
            ]
            
            # Calculate robot_start_orientation
            robot_joint_orientation = json_content['state']['registry']['object_registry'][robot_name]['joint_pos'][3:6]
            robot_start_orientation = T.euler2quat(th.tensor(robot_joint_orientation)).tolist()
            
            # Add instance to tasks_data
            tasks_data[task_name][instance_number] = {
                'scene_model': scene_model,
                'robot_start_position': robot_start_position,
                'robot_start_orientation': robot_start_orientation
            }
            
            print(f"Processed file: {json_file} from directory: {task_dir}")
            print(f"  Task: {task_name}")
            print(f"  Instance: {instance_number}")
            print(f"  Scene model: {scene_model}")
            print(f"  Robot start position: {robot_start_position}")
            print(f"  Robot start orientation: {robot_start_orientation}")
            print("-" * 50)
    
    # Write the data to the YAML file (completely overwriting it)
    yaml_file = os.path.join(script_dir, 'available_tasks.yaml')
    with open(yaml_file, 'w') as f:
        yaml.dump(tasks_data, f, default_flow_style=False)
    
    # Count total instances
    total_instances = sum(len(instances) for instances in tasks_data.values())
    print(f"Created new {yaml_file} with information from {len(tasks_data)} tasks and {total_instances} total instances")

if __name__ == "__main__":
    main()