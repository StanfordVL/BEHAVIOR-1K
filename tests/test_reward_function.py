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

def test_grasp_reward():
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

    # Make sure sim is stopped
    og.sim.stop()

    # Make sure GPU dynamics are enabled (GPU dynamics needed for cloth) and no flatcache
    gm.ENABLE_OBJECT_STATES = True
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_FLATCACHE = False

    # Create the environment
    env = og.Environment(configs=cfg, action_timestep=1 / 60., physics_timestep=1 / 60.)
    scene = env.scene
    robot = env.robots[0]
    env.reset()
    
    controller = StarterSemanticActionPrimitives(None, scene, robot)

    # Place object in scene
    obj = DatasetObject(
        name="cologne",
        category="bottle_of_cologne",
        model="lyipur"
    )
    og.sim.import_object(obj)
    obj.set_position([-0.3, -0.8, 0.5])

    # Set robot position so it can grasp without movement
    set_start_pose(robot)
    pose = controller._get_robot_pose_from_2d_pose([-0.433881, -0.210183, -2.96118])
    robot.set_position_orientation(*pose)
    og.sim.step()

    ctrl_gen = controller.apply_ref(StarterSemanticActionPrimitiveSet.GRASP, obj)
    

    rewards = [0]
    total_rewards = [0]

    # Check reward going from not grasping to not grasping
    _, reward, _, _ = env.step(next(ctrl_gen))
    rewards.append(reward)
    total_rewards.append(total_rewards[-1] + reward)
    eef_pos = robot.get_eef_position(robot.default_arm)
    expected_reward = math.exp(-T.l2_distance(eef_pos, obj.aabb_center)) * DIST_COEFF
    assert math.isclose(reward, expected_reward, abs_tol=0.01)

    for action in ctrl_gen:
        prev_obj_in_hand = robot._ag_obj_in_hand[robot.default_arm]
        _, reward, _, _ = env.step(action)
        rewards.append(reward)
        total_rewards.append(total_rewards[-1] + reward)
        obj_in_hand = robot._ag_obj_in_hand[robot.default_arm]
        if prev_obj_in_hand is None and obj_in_hand is not None:
            # Check reward going from not grapsing to after grasping
            expected_reward = math.exp(-T.l2_distance(robot.aabb_center, obj.aabb_center)) * DIST_COEFF + GRASP_REWARD
            assert math.isclose(reward, expected_reward, abs_tol=0.01)
        elif prev_obj_in_hand is not None and obj_in_hand is not None:
            # Check reward going from grasping to grasping
            expected_reward = math.exp(-T.l2_distance(robot.aabb_center, obj.aabb_center)) * DIST_COEFF + GRASP_REWARD
            # assert math.isclose(reward, expected_reward, abs_tol=0.01)
            # break

    ctrl_gen = controller.apply_ref(StarterSemanticActionPrimitiveSet.RELEASE)

    for action in ctrl_gen:
        prev_obj_in_hand = robot._ag_obj_in_hand[robot.default_arm]
        _, reward, _, _ = env.step(action)
        rewards.append(reward)
        total_rewards.append(total_rewards[-1] + reward)
        obj_in_hand = robot._ag_obj_in_hand[robot.default_arm]
        if prev_obj_in_hand is not None and obj_in_hand is None:
            # Check reward going from grapsing to not grasping
            eef_pos = robot.get_eef_position(robot.default_arm)
            expected_reward = math.exp(-T.l2_distance(eef_pos, obj.aabb_center)) * DIST_COEFF
            assert math.isclose(reward, expected_reward, abs_tol=0.01)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(rewards)
    plt.subplot(212)
    plt.plot(total_rewards)
    plt.show()

test_grasp_reward()