import asyncio
import omnigibson as og
from omnigibson.envs.rl_env import RLEnv
from omnigibson.macros import gm
import h5py

from environment_servicer import serve


async def main():
    DIST_COEFF = 0.1
    GRASP_REWARD = 0.3
    h5py.get_config().track_order = True

    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "load_object_categories": ["floors", "coffee_table"],
        },
        "robots": [
            {
                "type": "Tiago",
                "obs_modalities": ["rgb", "depth_linear", "seg_instance", "seg_semantic", "proprio"],
                "proprio_obs": ["robot_pose", "joint_qpos", "joint_qvel", "eef_left_pos", "eef_left_quat", "grasp_left"],
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
                        "modalities": ["rgb", "depth_linear", "seg_instance", "seg_semantic"],
                        "sensor_kwargs": {
                            "image_width": 224,
                            "image_height": 224
                        }
                    }
                },
                "controller_config": {
                    "base": {
                        "name": "JointController",
                        "motor_type": "velocity"
                    },
                    "arm_left": {
                        "name": "InverseKinematicsController",
                        "motor_type": "velocity",
                        "command_input_limits": None,
                        "command_output_limits": None,
                        "mode": "pose_absolute_ori", 
                        "kv": 3.0
                    },
                    # "arm_left": {
                    #     "name": "JointController",
                    #     "motor_type": "position",
                    #     "command_input_limits": None,
                    #     "command_output_limits": None, 
                    #     "use_delta_commands": False
                    # },
                    "arm_right": {
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": None,
                        "command_output_limits": None, 
                        "use_delta_commands": False
                    },
                    "gripper_left": {
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": [-1, 1],
                        "command_output_limits": None,
                        "use_delta_commands": True,
                        "use_single_command": True
                    },
                    "gripper_right": {
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": [-1, 1],
                        "command_output_limits": None,
                        "use_delta_commands": True,
                        "use_single_command": True
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
                "max_steps": 100000,
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

    reset_positions =  {
        'coffee_table_fqluyq_0': ([-0.4767243 , -1.219805  ,  0.25702515], [-3.69874935e-04, -9.39229270e-04,  7.08872199e-01,  7.05336273e-01]),
        'cologne': ([-0.30000001, -0.80000001,  0.44277492],
                    [0.        , 0.        , 0.        , 1.000000]),
        'robot0': ([0.0, 0.0, 0.05], [0.0, 0.0, 0.0, 1.0])
    }

    env_config = {
        "cfg": cfg,
        "reset_positions": reset_positions,
        "action_space_controllers": ["base", "camera", "arm_left", "gripper_left"]
    }
    env = RLEnv(env_config)

    # Now start servicing!
    await serve(env)

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())