from omnigibson.envs.rl_env import RLEnv
from ray.rllib.algorithms.ppo import PPOConfig

def main():
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
                "type": "Tiago",
                "obs_modalities": ["rgb", "depth"],
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
                        "modalities": ["rgb", "depth"],
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
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": None,
                        "command_output_limits": None, 
                        "use_delta_commands": False
                    },
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
                        "motor_type": "velocity",
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
                    [0.        , 0.        , 0.        , 1.00000012])
    }

    # Create the environment
    env_config = {
        "cfg": cfg,
        "reset_positions": reset_positions,
    }
    # env = RLEnv(env_config)
    # action = np.zeros(22)
    # for i in range(100):
    #     obs, reward, done, truncated, info = env.step(action)
    #     # from IPython import embed; embed()

    config = (  # 1. Configure the algorithm,
        PPOConfig()
        .environment(env=RLEnv, env_config=env_config)
        # .environment("Taxi-v3")
        .rollouts(num_rollout_workers=1)
        .framework("torch")
        # .training(model={"fcnet_hiddens": [64, 64]})
        .evaluation(evaluation_num_workers=1)
        .resources(num_gpus=1)
    )

    algo = config.build()  # 2. build the algorithm,

    for _ in range(5):
        print(algo.train())  # 3. train it,

    algo.evaluate()  # 4. and evaluate it.

if __name__ == "__main__":
    main()