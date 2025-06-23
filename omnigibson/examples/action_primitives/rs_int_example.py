import os

import torch as th
import yaml

import omnigibson as og
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
    StarterSemanticActionPrimitiveSet,
)
from omnigibson.robots.tiago import Tiago
from omnigibson.utils.ui_utils import choose_from_options


def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action)


def main():
    """
    Demonstrates how to use the action primitives to pick and place an object in a crowded scene.

    It loads Rs_int with a robot, and the robot picks and places an apple.
    """
    robot_options = ["R1", "Tiago"]
    robot_type = choose_from_options(options=robot_options, name="robot options", random_selection=False)

    # Load the config
    config_filename = os.path.join(og.example_config_path, f"{robot_type.lower()}_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["scene_model"] = "Rs_int"
    config["scene"]["not_load_object_categories"] = ["ceilings", "carpet"]
    config["objects"] = [
        {
            "type": "DatasetObject",
            "name": "apple",
            "category": "apple",
            "model": "agveuv",
            "position": [1.2, 0.0, 0.75],
            "orientation": [0, 0, 0, 1],
        },
    ]

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    # Open the gripper(s) to match cuRobo's default state
    for arm_name in robot.gripper_control_idx.keys():
        grpiper_control_idx = robot.gripper_control_idx[arm_name]
        robot.set_joint_positions(th.ones_like(grpiper_control_idx), indices=grpiper_control_idx, normalized=True)
    robot.keep_still()

    for _ in range(5):
        og.sim.step()

    env.scene.update_initial_file()
    env.scene.reset()

    og.sim.viewer_camera.set_position_orientation(
        th.tensor([1.8294, -3.2502, 1.6885]), th.tensor([0.5770, 0.1719, 0.2280, 0.7652])
    )

    # Let the object settle
    for _ in range(30):
        og.sim.step()

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    controller = StarterSemanticActionPrimitives(env, robot, enable_head_tracking=isinstance(robot, Tiago))
    coffee_table = scene.object_registry("name", "coffee_table_fqluyq_0")
    apple = scene.object_registry("name", "apple")

    # Grasp apple
    print("Start executing grasp")
    execute_controller(controller.apply_ref(StarterSemanticActionPrimitiveSet.GRASP, apple), env)
    print("Finish executing grasp")

    # Place on cabinet
    print("Start executing place")
    execute_controller(controller.apply_ref(StarterSemanticActionPrimitiveSet.PLACE_ON_TOP, coffee_table), env)
    print("Finish executing place")

    og.shutdown()


if __name__ == "__main__":
    main()
