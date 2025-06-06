import numpy as np
import torch as th
from collections import OrderedDict

# Robot parameters
ROBOT_TYPE = "R1Pro"  # This should always be our robot generally since GELLO is designed for this specific robot
ROBOT_NAME = "robot_r1"
RESOLUTION = [240, 240]  # Resolution for RGB and depth images


# Action indices
ACTION_QPOS_INDICES = {
    "R1Pro": OrderedDict(
        {
            "mobile_base": np.s_[0:3],
            "torso": np.s_[3:7],
            "left_arm": np.s_[7:14],
            "left_gripper": np.s_[14:15],
            "right_arm": np.s_[15:22],
            "right_gripper": np.s_[22:23],
        }
    )
}


# Proprioception configuration
PROPRIOCEPTION_INDICES = {
    "R1Pro": OrderedDict(
        {
            "joint_qpos": np.s_[0:28],
            "joint_qpos_sin": np.s_[28:56],
            "joint_qpos_cos": np.s_[56:84],
            "joint_qvel": np.s_[84:112],
            "joint_qeffort": np.s_[112:140],
            "robot_pos": np.s_[140:143],
            "robot_ori_cos": np.s_[143:146],
            "robot_ori_sin": np.s_[146:149],
            "robot_2d_ori": np.s_[149:150],
            "robot_2d_ori_cos": np.s_[150:151],
            "robot_2d_ori_sin": np.s_[151:152],
            "robot_lin_vel": np.s_[152:155],
            "robot_ang_vel": np.s_[155:158],
            "arm_left_qpos": np.s_[158:165],
            "arm_left_qpos_sin": np.s_[165:172],
            "arm_left_qpos_cos": np.s_[172:179],
            "arm_left_qvel": np.s_[179:186],
            "eef_left_pos": np.s_[186:189],
            "eef_left_quat": np.s_[189:193],
            "grasp_left": np.s_[193:194],
            "gripper_left_qpos": np.s_[194:196],
            "gripper_left_qvel": np.s_[196:198],
            "arm_right_qpos": np.s_[198:205],
            "arm_right_qpos_sin": np.s_[205:212],
            "arm_right_qpos_cos": np.s_[212:219],
            "arm_right_qvel": np.s_[219:226],
            "eef_right_pos": np.s_[226:229],
            "eef_right_quat": np.s_[229:233],
            "grasp_right": np.s_[233:234],
            "gripper_right_qpos": np.s_[234:236],
            "gripper_right_qvel": np.s_[236:238],
            "trunk_qpos": np.s_[238:242],
            "trunk_qvel": np.s_[242:246],
            "base_qpos": np.s_[246:249],
            "base_qpos_sin": np.s_[249:252],
            "base_qpos_cos": np.s_[252:255],
            "base_qvel": np.s_[255:258],
        }
    )
}

# Proprioception indices
PROPRIO_QPOS_INDICES = {
    "R1Pro": OrderedDict(
        {
            "torso": np.s_[6:10],
            "left_arm": np.s_[10:24:2],
            "right_arm": np.s_[11:24:2],
            "left_gripper": np.s_[24:26],
            "right_gripper": np.s_[26:28],
        }
    )
}

# Controller configuration for R1 robot
R1_CONTROLLER_CONFIG = {
    "arm_left": {
        "name": "JointController",
        "motor_type": "position",
        "pos_kp": 150,
        "command_input_limits": None,
        "command_output_limits": None,
        "use_impedances": False,
        "use_delta_commands": False,
    },
    "arm_right": {
        "name": "JointController",
        "motor_type": "position",
        "pos_kp": 150,
        "command_input_limits": None,
        "command_output_limits": None,
        "use_impedances": False,
        "use_delta_commands": False,
    },
    "gripper_left": {
        "name": "MultiFingerGripperController",
        "mode": "smooth",
        "command_input_limits": "default",
        "command_output_limits": "default",
    },
    "gripper_right": {
        "name": "MultiFingerGripperController",
        "mode": "smooth",
        "command_input_limits": "default",
        "command_output_limits": "default",
    },
    "base": {
        "name": "HolonomicBaseJointController",
        "motor_type": "velocity",
        "vel_kp": 150,
        "command_input_limits": [-th.ones(3), th.ones(3)],
        "command_output_limits": [-th.tensor([0.75, 0.75, 1.0]), th.tensor([0.75, 0.75, 1.0])],
        "use_impedances": False,
    },
    "trunk": {
        "name": "JointController",
        "motor_type": "position",
        "pos_kp": 150,
        "command_input_limits": None,
        "command_output_limits": None,
        "use_impedances": False,
        "use_delta_commands": False,
    },
    "camera": {
        "name": "NullJointController",
    },
}


# External camera parameters
EXTERNAL_CAMERA_CONFIGS = {
    "external_sensor0": {
        "position": [-0.4, 0, 2.0],
        "orientation": [0.2706, -0.2706, -0.6533, 0.6533],
    }
}


def get_camera_config(name, relative_prim_path, position, orientation, resolution, modalities):
    """
    Generate a camera configuration dictionary

    Args:
        name (str): Camera name
        relative_prim_path (str): Relative path to camera in the scene
        position (List[float]): Camera position [x, y, z]
        orientation (List[float]): Camera orientation [x, y, z, w]
        resolution (List[int]): Camera resolution [height, width]
        modalities (List[str]): List of modalities for the camera

    Returns:
        dict: Camera configuration dictionary
    """
    return {
        "sensor_type": "VisionSensor",
        "name": name,
        "relative_prim_path": relative_prim_path,
        "modalities": modalities,
        "sensor_kwargs": {
            "viewport_name": "Viewport",
            "image_height": resolution[0],
            "image_width": resolution[1],
        },
        "position": position,
        "orientation": orientation,
        "pose_frame": "parent",
        "include_in_obs": True,
    }


def generate_basic_environment_config(task_name=None, task_cfg=None):
    """
    Generate a basic environment configuration

    Args:
        task_name (str): Name of the task (optional)
        task_cfg: Dictionary of task config (optional)

    Returns:
        dict: Environment configuration
    """
    cfg = {
        "env": {
            "action_frequency": 30,
            "rendering_frequency": 30,
            "physics_frequency": 120,
            "external_sensors": [
                get_camera_config(
                    name="external_sensor0",
                    relative_prim_path=f"/controllable__{ROBOT_TYPE.lower()}__{ROBOT_NAME}/base_link/external_sensor0",
                    position=EXTERNAL_CAMERA_CONFIGS["external_sensor0"]["position"],
                    orientation=EXTERNAL_CAMERA_CONFIGS["external_sensor0"]["orientation"],
                    resolution=RESOLUTION,
                    modalities=["rgb", "depth_linear"],
                ),
            ],
        },
    }

    if task_name is not None and task_cfg is not None:
        # Load the environment for a particular task
        cfg["scene"] = {
            "type": "InteractiveTraversableScene",
            "scene_model": task_cfg["scene_model"],
            "load_room_types": None,
            "load_room_instances": task_cfg.get("load_room_instances", None),
            "include_robots": False,
        }

        cfg["task"] = {
            "type": "BehaviorTask",
            "activity_name": task_name,
            "activity_definition_id": 0,
            "activity_instance_id": 0,
            "predefined_problem": None,
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
        }
    return cfg


def generate_robot_config(task_name=None, task_cfg=None):
    """
    Generate robot configuration

    Args:
        task_name: Name of the task (optional)
        task_cfg: Dictionary of task config (optional)

    Returns:
        dict: Robot configuration
    """
    # Create a copy of the controller config to avoid modifying the original
    controller_config = {k: v.copy() for k, v in R1_CONTROLLER_CONFIG.items()}

    robot_config = {
        "type": ROBOT_TYPE,
        "name": ROBOT_NAME,
        "action_normalize": False,
        "controller_config": controller_config,
        "self_collisions": True,
        "obs_modalities": ["proprio", "rgb", "depth_linear"],
        "proprio_obs": list(PROPRIOCEPTION_INDICES[ROBOT_TYPE].keys()),
        "position": [0.0, 0.0, 0.0],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "grasping_mode": "assisted",
        "sensor_config": {
            "VisionSensor": {
                "sensor_kwargs": {
                    "image_height": RESOLUTION[0],
                    "image_width": RESOLUTION[1],
                },
            },
        },
    }

    # Override position and orientation for tasks
    if task_name is not None and task_cfg is not None:
        robot_config["position"] = task_cfg["robot_start_position"]
        robot_config["orientation"] = task_cfg["robot_start_orientation"]

    # Add reset joint positions
    joint_pos = th.zeros(28, dtype=th.float32)

    # Fingers MUST start open
    joint_pos[-4:] = 0.05

    # Update trunk qpos as well
    # Calculated from infer_torso_qpos_from_trunk_translate(DEFAULT_TRUNK_TRANSLATE), see og-gello repo for details
    joint_pos[6:10] = th.tensor([1.0250, -1.4500, -0.4700, 0.0000])
    robot_config["reset_joint_pos"] = joint_pos

    return robot_config


def flatten_obs_dict(obs: dict, parent_key: str = "") -> dict:
    """
    Process the observation dictionary by recursively flattening the keys.
    so obs["robot_r1"]["camera"]["rgb"] will become obs["robot_r1::camera:::rgb"].
    """
    processed_obs = {}
    for key, value in obs.items():
        new_key = f"{parent_key}::{key}" if parent_key else key
        if isinstance(value, dict):
            processed_obs.update(flatten_obs_dict(value, parent_key=new_key))
        else:
            processed_obs[new_key] = value
    return processed_obs
