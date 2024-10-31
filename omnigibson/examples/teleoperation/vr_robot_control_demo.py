"""
Example script for interacting with OmniGibson scenes with VR and BehaviorRobot.
"""
import torch as th

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.teleop_utils import OVXRSystem

gm.ENABLE_OBJECT_STATES = False
gm.ENABLE_TRANSITION_RULES = False


def main():
    """
    Spawn a BehaviorRobot in Rs_int and users can navigate around and interact with the scene using VR.
    """
    # Create the config for generating the environment we want
    # scene_cfg = {"type": "Scene"}
    scene_cfg = {"type": "InteractiveTraversableScene", "scene_model": "Rs_int", "load_object_categories": ["floors", "walls", "ceilings"]}
    robot0_cfg = {
        "type": "R1",
        "obs_modalities": ["rgb"],
        "controller_config": {
            "arm_left": {"name": "InverseKinematicsController", "mode": "absolute_pose", "command_input_limits": None, "command_output_limits": None},
            "arm_right": {"name": "InverseKinematicsController", "mode": "absolute_pose", "command_input_limits": None, "command_output_limits": None},
            "gripper_left": {"command_input_limits": "default"},
            "gripper_right": {"command_input_limits": "default"},
        },
        "action_normalize": False
    }
    cfg = dict(scene=scene_cfg, robots=[robot0_cfg])

    # Create the environment
    env = og.Environment(configs=cfg)
    env.reset()
    # start vrsys
    vrsys = OVXRSystem(
        robot=env.robots[0], show_control_marker=True, system="SteamVR", eef_tracking_mode="controller", align_anchor_to="camera"
    )
    vrsys.start()

    # main simulation loop
    for _ in range(1000):
        # update the VR system
        vrsys.update()
        # get the action from the VR system and step the environment
        env.step(vrsys.get_robot_teleop_action())

    # Shut down the environment cleanly at the end
    vrsys.stop()
    og.clear()


if __name__ == "__main__":
    main()
