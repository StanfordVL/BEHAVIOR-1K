import omnigibson as og
import numpy as np
from omnigibson.macros import gm
from telemoma.human_interface.teleop_core import TeleopAction
from omnigibson.utils.transform_utils import quat2euler
import pytest

@pytest.mark.skip(reason="test hangs on CI")
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
                        "command_input_limits": None,
                    },
                }
            }
        ],
    }

    # Make sure sim is stopped
    if og.sim is not None:
        og.sim.stop()

    # Make sure GPU dynamics are enabled (GPU dynamics needed for cloth) and no flatcache
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_FLATCACHE = False

    # Create the environment
    env = og.Environment(configs=cfg)
    robot = env.robots[0]
    env.reset()
    teleop_action = TeleopAction()
    start_base_pose = robot.get_position_orientation()
    start_eef_pose = robot.links[robot.eef_link_names[robot.default_arm]].get_position_orientation()

    # test moving robot arm
    teleop_action.right = np.concatenate(([0.01], np.zeros(6)))
    for _ in range(50):
        action = robot.teleop_data_to_action(teleop_action)
        env.step(action)
    cur_eef_pose = robot.links[robot.eef_link_names[robot.default_arm]].get_position_orientation()
    assert cur_eef_pose[0][0] - start_eef_pose[0][0] > 0.02, "Robot arm not moving forward"
    
    # test moving robot base
    teleop_action.right = np.zeros(7)
    teleop_action.base = np.array([0.1, 0, 0.1])
    for _ in range(50):
        action = robot.teleop_data_to_action(teleop_action)
        env.step(action)
    cur_base_pose = robot.get_position_orientation()
    assert cur_base_pose[0][0] - start_base_pose[0][0] > 0.02, "robot base not moving forward"
    assert quat2euler(cur_base_pose[1])[2] - quat2euler(start_base_pose[1])[2] > 0.02, "robot base not rotating counter-clockwise"
    
    # Clear the sim
    og.sim.clear()
