import math
import numpy as np
import matplotlib.pyplot as plt
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, StarterSemanticActionPrimitiveSet
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject


def set_start_pose(robot):
    reset_pose_tiago = np.array([
        -1.78029833e-04,  3.20231302e-05, -1.85759447e-07, -1.16488536e-07,
        4.55182843e-08,  2.36128806e-04,  1.50000000e-01,  9.40000000e-01,
        -1.10000000e+00,  0.00000000e+00, -0.90000000e+00,  1.47000000e+00,
        0.00000000e+00,  2.10000000e+00,  2.71000000e+00,  1.50000000e+00,
        1.71000000e+00,  1.30000000e+00, -1.57000000e+00, -1.40000000e+00,
        1.39000000e+00,  0.00000000e+00,  0.00000000e+00,  4.50000000e-02,
        4.50000000e-02,  4.50000000e-02,  4.50000000e-02,
    ])
    robot.set_joint_positions(reset_pose_tiago)
    og.sim.step()

def step_sim(time):
    for _ in range(int(time*100)):
        og.sim.step()

def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action)

DIST_COEFF = 0.1
GRASP_REWARD = 0.3
RL_ITERATIONS = 10

cfg = {
    "scene": {
        "type": "InteractiveTraversableScene",
        "scene_model": "Rs_int",
        "load_object_categories": ["floors", "coffee_table"],
    },
    "robots": [
        {
            "type": "Tiago",
            "obs_modalities": ["scan", "rgb", "depth"],
            "scale": 1.0,
            "self_collisions": True,
            "action_normalize": False,
            "action_type": "continuous",
            "grasping_mode": "sticky",
            "rigid_trunk": False,
            "default_arm_pose": "diagonal30",
            "default_trunk_offset": 0.365,
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
    }
}

# Create the environment
env = og.Environment(configs=cfg, action_timestep=1 / 60., physics_timestep=1 / 60.)
scene = env.scene
robot = env.robots[0]

# Place object in scene
obj = DatasetObject(
    name="cologne",
    category="bottle_of_cologne",
    model="lyipur"
)
og.sim.import_object(obj)
obj.set_position([-0.3, -0.8, 0.5])
set_start_pose(robot)
og.sim.step()
env.scene.update_initial_state()

controller = StarterSemanticActionPrimitives(None, scene, robot)
env.task.add_primitive_controller(controller)

ctrl_gen = controller.apply_ref(StarterSemanticActionPrimitiveSet.GRASP, obj)

for i in range(RL_ITERATIONS):
    env.reset()
    try:
        for action in ctrl_gen:
            state, reward, done, info = env.step(action)
            if done:
                break
    except Exception as e:
        print(e)
        print("Error in iteration", i)