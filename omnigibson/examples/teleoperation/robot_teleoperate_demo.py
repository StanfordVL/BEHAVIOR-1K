"""
Example script for using VR controller to teleoperate a robot.
"""
import omnigibson as og
from omnigibson.objects import USDObject
from omnigibson.utils.ui_utils import choose_from_options

ROBOTS = {
    "FrankaPanda": "Franka Emika Panda (default)",
    "Fetch": "Mobile robot with one arm",
    "Tiago": "Mobile robot with two arms",
}

SYSTEMS = {
    "SteamVR": "SteamVR with HTC VIVE through OmniverseXR plugin (default)",
    "Oculus": "Oculus Reader with Quest 2",
    "Spacemouse": "Spacemouse",
}


def main():
    teleop_system = choose_from_options(options=SYSTEMS, name="system")
    robot_name = choose_from_options(options=ROBOTS, name="robot")
    # Create the config for generating the environment we want
    env_cfg = {"action_timestep": 1 / 60., "physics_timestep": 1 / 180.}
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
            "mode": "pose_absolute_ori",
            "motor_type": "position"
        }
        robot_cfg["controller_config"][f"gripper_{arm}"] = {
            "name": "MultiFingerGripperController",
            "command_input_limits": (0.0, 1.0),
            "mode": "smooth",
            "inverted": True
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
            "position": [0.6, -0.3, 0.5],
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
            "position": [0.6, 0.1, 0.5],
        },
        {
            "type": "DatasetObject",
            "prim_path": "/World/toy_figure3",
            "name": "toy_figure3",
            "category": "toy_figure",
            "model": "eulekw",
            "scale": [0.25, 0.25, 0.25],
            "position": [0.6, 0.2, 0.5],
        },
        {
            "type": "DatasetObject",
            "prim_path": "/World/toy_figure4",
            "name": "toy_figure4",
            "category": "toy_figure",
            "model": "yxiksm",
            "scale": [0.25, 0.25, 0.25],
            "position": [0.6, 0.3, 0.5],
        },
        {
            "type": "DatasetObject",
            "prim_path": "/World/toy_figure5",
            "name": "toy_figure5",
            "category": "toy_figure",
            "model": "wvpqbf",
            "scale": [0.75, 0.75, 0.75],
            "position": [0.6, 0.4, 0.5],
        },
    ]
    cfg = dict(env=env_cfg, scene=scene_cfg, robots=[robot_cfg], objects=object_cfg)

    # Create the environment
    env = og.Environment(configs=cfg)
    env.reset()
    # update viewer camera pose
    og.sim.viewer_camera.set_position_orientation([-0.22, 0.99, 1.09], [-0.14, 0.47, 0.84, -0.23])

    # Start teleoperation system
    robot = env.robots[0]
    # Import robot eef marker
    target_markers = {}
    for arm in robot.arm_names:
        arm_name = "right" if arm == robot.default_arm else "left"
        target_markers[arm_name] = USDObject(name=f"target_{arm_name}", usd_path=robot.eef_usd_path[arm], visual_only=True)
        og.sim.import_object(target_markers[arm_name])

    # Initialize teleoperation system
    if teleop_system == "SteamVR":
        from omnigibson.utils.teleop_utils import OVXRSystem as TeleopSystem
    elif teleop_system == "Oculus":
        from omnigibson.utils.teleop_utils import OculusReaderSystem as TeleopSystem
    elif teleop_system == "Spacemouse":
        from omnigibson.utils.teleop_utils import SpaceMouseSystem as TeleopSystem
    teleop_sys = TeleopSystem(robot=robot, show_controller=True, disable_display_output=True, align_anchor_to_robot_base=True)
    teleop_sys.start()
    # tracker variable of whether the robot is attached to the VR system
    prev_robot_attached = False
    # main simulation loop
    for _ in range(10000):
        if og.sim.is_playing():
            teleop_sys.update()
            if teleop_sys.teleop_data["robot_attached"] == True and prev_robot_attached == False:
                teleop_sys.reset_transform_mapping()
            else:
                action = teleop_sys.teleop_data_to_action()
                env.step(action) 
            # update the target object's pose to the VR right controller's pose
            for arm_name in target_markers:
                if arm_name in teleop_sys.teleop_data["transforms"]:
                    target_markers[arm_name].set_position_orientation(*teleop_sys.teleop_data["transforms"][arm_name])  
            prev_robot_attached = teleop_sys.teleop_data["robot_attached"]
    # Shut down the environment cleanly at the end
    teleop_sys.stop()
    env.close()

if __name__ == "__main__":
    main()