import csv
import os
import random
import numpy as np
import torch as th
from collections import OrderedDict
from omnigibson.macros import gm


HEAD_RESOLUTION = (720, 720)
WRIST_RESOLUTION = (480, 480)
DEFAULT_EVAL_SEED = 0
NUM_TEST_INSTANCES = 40
NUM_PUBLIC_TEST_INSTANCES = 20
NUM_HIDDEN_TEST_INSTANCES = NUM_TEST_INSTANCES - NUM_PUBLIC_TEST_INSTANCES
TEST_INSTANCE_IDS = list(range(301, 341))


def seed_everything(seed: int, deterministic_torch: bool = False) -> int:
    """
    Seed Python, NumPy, and Torch RNGs used by eval.

    Setting PYTHONHASHSEED here helps child processes and documents the intended seed, but
    hash randomization for the current Python process is fixed at interpreter startup.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
    try:
        import warp as wp

        wp.rand_init(seed)
    except ImportError:
        pass

    if hasattr(th.backends, "cudnn"):
        th.backends.cudnn.benchmark = False
        th.backends.cudnn.deterministic = True

    if deterministic_torch:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            th.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            th.use_deterministic_algorithms(True)

    return seed


def _load_challenge_task_ids() -> dict[str, int]:
    task_ids = {}
    task_misc_path = os.path.join(gm.DATA_PATH, "2026-challenge-task-instances", "metadata", "B100_task_misc.csv")
    with open(task_misc_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            task_ids[row["Task"]] = int(row["Task ID"])
    return task_ids


TASK_NAMES_TO_INDICES = _load_challenge_task_ids()
if not TASK_NAMES_TO_INDICES:
    raise RuntimeError(
        "Could not load challenge task metadata. Expected one of: "
        f"{os.path.join(gm.DATA_PATH, '2026-challenge-task-instances', 'metadata', 'B100_task_misc.csv')}. "
        "Please ensure OMNIGIBSON_DATA_PATH / gm.DATA_PATH points at the BEHAVIOR data root."
    )

# Camera parameteres
CAMERA_INTRINSICS = {
    "R1Pro": {
        "head": np.array([[306.0, 0.0, 360.0], [0.0, 306.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float32),  # 720x720
        "left_wrist": np.array(
            [[388.6639, 0.0, 240.0], [0.0, 388.6639, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32
        ),  # 480x480
        "right_wrist": np.array(
            [[388.6639, 0.0, 240.0], [0.0, 388.6639, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32
        ),  # 480x480
    },
}

CAMERA_EXTRINSICS = {
    "R1Pro": {
        "head": np.array([[306.0, 0.0, 360.0], [0.0, 306.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float32),  # 720x720
        "left_wrist": np.array(
            [[388.6639, 0.0, 240.0], [0.0, 388.6639, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32
        ),  # 480x480
        "right_wrist": np.array(
            [[388.6639, 0.0, 240.0], [0.0, 388.6639, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32
        ),  # 480x480
    },
}


# Action indices
ACTION_QPOS_INDICES = {
    "R1Pro": OrderedDict(
        {
            "base": np.s_[0:3],
            "torso": np.s_[3:7],
            "left_arm": np.s_[7:14],
            "left_gripper": np.s_[14:15],
            "right_arm": np.s_[15:22],
            "right_gripper": np.s_[22:23],
        }
    ),
}


# Proprioception configuration
PROPRIOCEPTION_INDICES = {
    "R1Pro": OrderedDict(
        {
            "base_qvel": np.s_[0:3],
            "arm_left_qpos": np.s_[3:10],
            "arm_left_qvel": np.s_[10:17],
            "eef_left_pos": np.s_[17:20],
            "eef_left_quat": np.s_[20:24],
            "gripper_left_qpos": np.s_[24:26],
            "gripper_left_qvel": np.s_[26:28],
            "arm_right_qpos": np.s_[28:35],
            "arm_right_qvel": np.s_[35:42],
            "eef_right_pos": np.s_[42:45],
            "eef_right_quat": np.s_[45:49],
            "gripper_right_qpos": np.s_[49:51],
            "gripper_right_qvel": np.s_[51:53],
            "trunk_qpos": np.s_[53:57],
            "trunk_qvel": np.s_[57:61],
        }
    ),
}


def get_robot_camera_names(robot_name: str, robot_eval_config: dict | None) -> dict[str, str]:
    """
    Return flattened observation names for cameras configured by eval role.

    Robot configs may include:
        eval:
          camera_sensor_names:
            head: robot_r1:zed_link:Camera:0

    Values are robot sensor names, i.e. keys in robot.sensors. The flattened
    observation name adds the robot observation namespace.
    """
    robot_eval_config = robot_eval_config or {}
    camera_sensor_names = robot_eval_config.get("camera_sensor_names", {})
    return {camera_id: f"{robot_name}::{sensor_name}" for camera_id, sensor_name in camera_sensor_names.items()}


def set_sensor_modalities(sensor, modalities: set[str]) -> None:
    for modality in tuple(sensor.modalities):
        if modality not in modalities:
            sensor.remove_modality(modality)
    for modality in modalities:
        if modality not in sensor.modalities:
            sensor.add_modality(modality)


def generate_basic_environment_config(task_name, task_cfg):
    """
    Generate a basic environment configuration

    Args:
        task_name (str): Name of the task
        task_cfg: Dictionary of task config

    Returns:
        dict: Environment configuration
    """
    cfg = {
        "env": {
            "action_frequency": 30,
            "rendering_frequency": 30,
            "physics_frequency": 120,
        },
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": task_cfg["scene_model"],
            "load_room_types": None,
            "load_room_instances": task_cfg.get("load_room_instances", None),
            "include_robots": False,
        },
        "task": {
            "type": "BehaviorTask",
            "activity_name": task_name,
            "activity_definition_id": 0,
            "activity_instance_id": 0,
            "online_object_sampling": False,
            "debug_object_sampling": False,
            "highlight_task_relevant_objects": False,
            "termination_config": {
                "max_steps": 5000,
            },
            "reward_config": {
                "r_potential": 1.0,
            },
            "include_obs": False,
        },
    }
    return cfg


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


def find_start_point(base_vel):
    """
    Find the first point where the base velocity is non-zero.
    This is used to skip the initial part of the dataset where the robot is not moving.
    """
    start_idx = np.where(np.linalg.norm(base_vel, axis=-1) > 1e-5)[0]
    if len(start_idx) == 0:
        return 0
    return min(start_idx[0], 500)  # Limit to the first 100 points to avoid long initial periods
