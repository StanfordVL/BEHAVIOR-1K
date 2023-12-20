import numpy as np
import omnigibson as og
from omnigibson.macros import gm

from grpc_server import serve_env_over_grpc

gm.USE_FLATCACHE = True

def main(local_addr, learner_addr):

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
                        "mode": "ternary",
                        "motor_type": "position",
                        "command_input_limits": [-1, 1],
                        "command_output_limits": None,
                    },
                    "camera": {
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": "default",
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

    env = og.Environment(configs=cfg, action_timestep=1 / 10., physics_timestep=1 / 60., flatten_obs_space=True, flatten_action_space=True)

    # Now start servicing!
    serve_env_over_grpc(env, local_addr, learner_addr)

if __name__ == "__main__":
    import sys
    local_port = int(sys.argv[1])
    main("localhost:" + str(local_port), "localhost:50051")