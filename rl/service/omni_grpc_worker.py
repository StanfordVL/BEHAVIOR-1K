import asyncio
import numpy as np
import omnigibson as og
from omnigibson.macros import gm

from rollout_worker import serve

gm.USE_GPU_DYNAMICS = True

async def main():

    DIST_COEFF = 0.1
    GRASP_REWARD = 0.3

    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "load_object_categories": ["floors", "coffee_table"],
        },
        "robots": [
            {
                "type": "Fetch",
                "obs_modalities": ["rgb", "proprio"],
                "proprio_obs": ["joint_qpos", "joint_qvel", "eef_0_pos", "eef_0_quat", "grasp_0"],
                "scale": 1.0,
                "self_collisions": True,
                "action_normalize": False,
                "action_type": "continuous",
                "grasping_mode": "sticky",
                "rigid_trunk": False,
                "default_arm_pose": "diagonal30",
                "default_trunk_offset": 0.365,
                "sensor_config": {
                    "VisionSensor": {
                        "modalities": ["rgb"],
                        "sensor_kwargs": {
                            "image_width": 224,
                            "image_height": 224
                        }
                    }
                },
                "controller_config": {
                    "base": {
                        "name": "DifferentialDriveController",
                    },
                    "arm_0": {
                        "name": "InverseKinematicsController",
                        "motor_type": "velocity",
                        "command_input_limits": (np.array([-0.2, -0.2, -0.2, -np.pi, -np.pi, -np.pi]),
                        np.array([0.2, 0.2, 0.2, np.pi, np.pi, np.pi])),
                        "command_output_limits": None,
                        "mode": "pose_absolute_ori", 
                        "kv": 3.0
                    },
                    "gripper_0": {
                        "name": "MultiFingerGripperController",
                        "motor_type": "ternary",
                        "command_input_limits": [-1, 1],
                        "command_output_limits": None,
                    },
                    "camera": {
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": None,
                        "command_output_limits": None,
                        "use_delta_commands": False
                    }
                }
            }
        ],
        "task": {
            "type": "GraspTask",
            "obj_name": "cologne",
            "termination_config": {
                "max_steps": 400,
            },
            "reward_config": {
                "r_dist_coeff": DIST_COEFF,
                "r_grasp": GRASP_REWARD
            }
        },
        "objects": [
            {
                "type": "DatasetObject",
                "name": "cologne",
                "category": "bottle_of_cologne",
                "model": "lyipur",
                "position": [-0.3, -0.8, 0.5],
            },
        ]
    }

    env = og.Environment(configs=cfg, action_timestep=1 / 10., physics_timestep=1 / 60.)

    # Now start servicing!
    await serve(env)

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())