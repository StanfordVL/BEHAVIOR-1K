"""
Example script for using VR controller to teleoperate a robot.
"""
import omnigibson as og
from omnigibson.utils.teleop_utils import OculusReaderSystem as VRSystem
from omnigibson.utils.ui_utils import choose_from_options

ROBOTS = {
    "FrankaPanda": "Franka Emika Panda (default)",
    "Fetch": "Mobile robot with one arm",
    "Tiago": "Mobile robot with two arms",
}


def main():
    robot_name = choose_from_options(options=ROBOTS, name="robot")
    # Create the config for generating the environment we want
    env_cfg = {"action_timestep": 1 / 10., "physics_timestep": 1 / 60.}
    scene_cfg = {"type": "Scene"}
    # Add the robot we want to load
    robot0_cfg = {
        "type": robot_name,
        "obs_modalities": ["rgb"],
        "action_normalize": False,
        "grasping_mode": "assisted",
        "controller_config": {
            "arm_0": {
                "name": "InverseKinematicsController",
                "mode": "pose_absolute_ori",
            },
            "gripper_0": {
                "name": "MultiFingerGripperController", 
                "command_input_limits": (0.0, 1.0),
                "mode": "smooth", 
                "inverted": True
            }
        }
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
        {
            "type": "PrimitiveObject",
            "primitive_type": "Cube",
            "name": "target",  
            "scale": [0.1, 0.1, 0.1],
            "visual_only": True,
            "rgba": [0.0, 1.0, 0.0, 1.0],
        },
    ]
    cfg = dict(env=env_cfg, scene=scene_cfg, robots=[robot0_cfg], objects=object_cfg)

    # Create the environment
    env = og.Environment(configs=cfg)
    env.reset()
    # update viewer camera pose
    og.sim.viewer_camera.set_position_orientation([-0.22, 0.99, 1.09], [-0.14, 0.47, 0.84, -0.23])

    # Start vrsys 
    robot = env.robots[0]
    vrsys = VRSystem(robot=robot, system="OpenXR", show_controller=True, disable_display_output=True, align_anchor_to_robot_base=True)
    vrsys.start()
    # tracker variable of whether the robot is attached to the VR system
    prev_robot_attached = False
    # get the target object
    target = env.scene.object_registry("name", "target")
    # main simulation loop
    for _ in range(10000):
        if og.sim.is_playing():
            vrsys.update()
            if vrsys.teleop_data["robot_attached"] == True and prev_robot_attached == False:
                # The user just pressed the grip, so snap the VR right controller to the robot's right arm
                if robot.model_name == "Tiago":
                    # Tiago's default arm is the left arm
                    robot_eef_pos = robot.links[robot.eef_link_names["right"]].get_position()
                else:
                    robot_eef_pos = robot.links[robot.eef_link_names[robot.default_arm]].get_position()
                base_orn = robot.get_orientation()
                vrsys.reset_transform_mapping(robot_eef_pos=robot_eef_pos, robot_base_orn=base_orn)
            else:
                action = vrsys.teleop_data_to_action()
                env.step(action) 
            if "right" in vrsys.teleop_data["transforms"]:
                # update the target object's pose to the VR right controller's pose
                target.set_position_orientation(vrsys.teleop_data["transforms"]["right"][0], vrsys.teleop_data["transforms"]["right"][1])  
            prev_robot_attached = vrsys.teleop_data["robot_attached"]
    # Shut down the environment cleanly at the end
    vrsys.stop()
    env.close()

if __name__ == "__main__":
    main()