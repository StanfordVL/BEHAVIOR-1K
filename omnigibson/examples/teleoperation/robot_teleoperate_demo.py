"""
Example script for using external devices to teleoperate a robot.
"""

import omnigibson as og
from omnigibson.utils.ui_utils import choose_from_options

ROBOTS = {
    "FrankaPanda": "Franka Emika Panda (default)",
    "Fetch": "Mobile robot with one arm",
    "Tiago": "Mobile robot with two arms",
}
TELEOP_METHOD = {
    "keyboard": "Keyboard (default)",
    "spacemouse": "SpaceMouse",
    "oculus": "Oculus Quest",
    "vision": "Human Keypoints with Camera",
}


def main():
    """
    Spawn a robot in an empty scene with a breakfast table and some toys.
    Users can try pick and place the toy into the basket using selected external devices and robot of their choice.
    """
    from telemoma.configs.base_config import teleop_config
    from telemoma.utils.camera_utils import RealSenseCamera

    from omnigibson.utils.teleop_utils import TeleopSystem

    robot_name = choose_from_options(options=ROBOTS, name="robot")
    arm_teleop_method = choose_from_options(options=TELEOP_METHOD, name="robot arm teleop method")
    if robot_name != "FrankaPanda":
        base_teleop_method = choose_from_options(options=TELEOP_METHOD, name="robot base teleop method")
    else:
        base_teleop_method = "keyboard"  # Dummy value since FrankaPanda does not have a base
    # Generate teleop config
    teleop_config.arm_left_controller = arm_teleop_method
    teleop_config.arm_right_controller = arm_teleop_method
    teleop_config.base_controller = base_teleop_method
    teleop_config.interface_kwargs["keyboard"] = {"arm_speed_scaledown": 0.04}
    teleop_config.interface_kwargs["spacemouse"] = {"arm_speed_scaledown": 0.04}
    if arm_teleop_method == "vision" or base_teleop_method == "vision":
        teleop_config.interface_kwargs["vision"] = {"camera": RealSenseCamera()}

    # Create the config for generating the environment we want
    scene_cfg = {"type": "Scene"}
    # Add the robot we want to load
    robot_cfg = {
        "type": robot_name,
        "obs_modalities": ["rgb"],
        "action_normalize": False,
        "grasping_mode": "assisted",
    }
    arms = ["left", "right"] if robot_name == "Tiago" else ["0"]
    robot_cfg["controller_config"] = {}
    for arm in arms:
        robot_cfg["controller_config"][f"arm_{arm}"] = {
            "name": "InverseKinematicsController",
            "command_input_limits": None,
        }
        robot_cfg["controller_config"][f"gripper_{arm}"] = {
            "name": "MultiFingerGripperController",
            "command_input_limits": (0.0, 1.0),
            "mode": "smooth",
        }
    object_cfg = [
        {
            "type": "DatasetObject",
            "prim_path": "/World/breakfast_table",
            "name": "breakfast_table",
            "category": "breakfast_table",
            "model": "kwmfdg",
            "bounding_box": [2, 1, 0.4],
            "position": [0.8, 0, 0.3],
            "orientation": [0, 0, 0.707, 0.707],
        },
        {
            "type": "DatasetObject",
            "prim_path": "/World/frail",
            "name": "frail",
            "category": "frail",
            "model": "zmjovr",
            "scale": [2, 2, 2],
            "position": [0.6, -0.35, 0.5],
        },
        {
            "type": "DatasetObject",
            "prim_path": "/World/toy_figure1",
            "name": "toy_figure1",
            "category": "toy_figure",
            "model": "issvzv",
            "scale": [0.75, 0.75, 0.75],
            "position": [0.6, 0, 0.5],
        },
        {
            "type": "DatasetObject",
            "prim_path": "/World/toy_figure2",
            "name": "toy_figure2",
            "category": "toy_figure",
            "model": "nncqfn",
            "scale": [0.75, 0.75, 0.75],
            "position": [0.6, 0.15, 0.5],
        },
        {
            "type": "DatasetObject",
            "prim_path": "/World/toy_figure3",
            "name": "toy_figure3",
            "category": "toy_figure",
            "model": "eulekw",
            "scale": [0.25, 0.25, 0.25],
            "position": [0.6, 0.3, 0.5],
        },
    ]
    cfg = dict(scene=scene_cfg, robots=[robot_cfg], objects=object_cfg)

    # Create the environment
    env = og.Environment(configs=cfg)
    env.reset()
    # update viewer camera pose
    og.sim.viewer_camera.set_position_orientation(position=[-0.22, 0.99, 1.09], orientation=[-0.14, 0.47, 0.84, -0.23])
    # Start teleoperation system
    robot = env.robots[0]

    # Initialize teleoperation system
    teleop_sys = TeleopSystem(config=teleop_config, robot=robot, show_control_marker=True)
    teleop_sys.start()

    # main simulation loop
    for _ in range(10000):
        action = teleop_sys.get_action(teleop_sys.get_obs())
        env.step(action)

    # Shut down the environment cleanly at the end
    teleop_sys.stop()
    og.clear()


if __name__ == "__main__":
    main()
