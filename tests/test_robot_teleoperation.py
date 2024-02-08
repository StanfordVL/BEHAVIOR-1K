import pytest
import omnigibson as og
import numpy as np
from omnigibson.macros import gm
from omnigibson.utils.teleop_utils import TeleopSystem
from omnigibson.utils.transform_utils import quat2euler, euler2quat

@pytest.mark.skip(reason="test is broken")
def test_teleop():
    cfg = {
        "env": {"action_timestep": 1 / 60., "physics_timestep": 1 / 120.},
        "scene": {"type": "Scene"},
        "robots": [
            {
                "type": "Fetch",
                "action_normalize": False,
                "controller_config": {
                    "arm_0": {
                        "name": "InverseKinematicsController",
                        "mode": "pose_absolute_ori",
                        "motor_type": "position"
                    },
                }
            }
        ],
    }

    # Make sure sim is stopped
    og.sim.stop()

    # Make sure GPU dynamics are enabled (GPU dynamics needed for cloth) and no flatcache
    gm.ENABLE_OBJECT_STATES = True
    gm.USE_GPU_DYNAMICS = True
    gm.ENABLE_FLATCACHE = False

    # Create the environment
    env = og.Environment(configs=cfg)
    robot = env.robots[0]
    env.reset()
    teleop_system = TeleopSystem(robot=robot, show_control_marker=False)
    start_base_pose = robot.get_position_orientation()
    start_eef_pose = robot.links[robot.eef_link_names[robot.default_arm]].get_position_orientation()

    # test moving robot arm
    teleop_system.teleop_data.robot_attached = True
    teleop_system.reset_transform_mapping()
    target_eef_pos = start_eef_pose[0] + np.array([0.05, 0.05, 0])
    target_eef_orn = euler2quat(quat2euler(start_eef_pose[1]) + np.array([0.5, 0.5, 0]))
    teleop_system.teleop_data.is_valid["right"] = True
    teleop_system.teleop_data.transforms["base"] = np.zeros(4)
    teleop_system.teleop_data.transforms["right"] = (target_eef_pos, target_eef_orn)
    for _ in range(50):
        action = teleop_system.teleop_data_to_action()
        env.step(action)
    cur_eef_pose = robot.links[robot.eef_link_names[robot.default_arm]].get_position_orientation()
    cur_base_pose = robot.get_position_orientation()
    assert np.allclose(cur_base_pose[0], start_base_pose[0], atol=1e-2), "base pos not in starting place"
    assert np.allclose(cur_base_pose[1], start_base_pose[1], atol=1e-2), "base orn not in starting place"
    assert np.allclose(cur_eef_pose[0], target_eef_pos, atol=1e-2), "eef pos not in target place"
    assert np.allclose(cur_eef_pose[1], target_eef_orn, atol=1e-2) or np.allclose(cur_eef_pose[1], -target_eef_orn, atol=1e-2), \
        "eef orn not in target place"
    
    # test moving robot base
    teleop_system.teleop_data.transforms["right"] = cur_eef_pose
    teleop_system.teleop_data.transforms["base"] = np.array([0.02, 0, 0, 0.02])
    for _ in range(50):
        action = teleop_system.teleop_data_to_action()
        env.step(action)
    cur_base_pose = robot.get_position_orientation()
    assert cur_base_pose[0][0] > start_base_pose[0][0], "robot base not moving forward"
    assert quat2euler(cur_base_pose[1])[2] > quat2euler(start_base_pose[1])[2], "robot base not rotating counter-clockwise"
    
    # Clear the sim
    og.sim.clear()
