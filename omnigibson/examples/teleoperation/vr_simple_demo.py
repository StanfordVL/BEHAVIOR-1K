"""
Example script for interacting with OmniGibson scenes with VR and BehaviorRobot.
"""

import omnigibson as og
from omnigibson.utils.teleop_utils import OVXRSystem


def main():
    """
    Spawn a BehaviorRobot in Rs_int and users can navigate around and interact with the scene using VR.
    """
    # Create the config for generating the environment we want
    scene_cfg = {"type": "Scene"}  # "InteractiveTraversableScene", "scene_model": "Rs_int"}
    robot0_cfg = {
        "type": "Tiago",
        "controller_config": {
            "gripper_left": {"command_input_limits": "default"},
            "gripper_right": {"command_input_limits": "default"},
        },
    }
    cfg = dict(scene=scene_cfg, robots=[robot0_cfg])

    # Create the environment
    env = og.Environment(configs=cfg)
    env.reset()
    # start vrsys
    vrsys = OVXRSystem(
        robot=env.robots[0], show_control_marker=False, system="SteamVR", align_anchor_to_robot_base=True
    )
    vrsys.start()
    # set headset position to be 1m above ground and facing +x
    vrsys.set_initial_transform(pos=[0, 0, 1], orn=[0, 0, 0, 1])

    # main simulation loop
    for _ in range(10000):
        # step the VR system to get the latest data from VR runtime
        vrsys.update()
        # generate robot action and step the environment
        action = vrsys.teleop_data_to_action()
        env.step(action)

    # Shut down the environment cleanly at the end
    vrsys.stop()
    og.clear()


if __name__ == "__main__":
    main()
