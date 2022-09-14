"""
Example script demo'ing robot manipulation control with grasping.
"""
import logging
import os
import platform
import random
import sys
import time
from collections import OrderedDict

import numpy as np

import igibson as ig
from igibson.objects import DatasetObject, PrimitiveObject
from igibson.utils.asset_utils import get_ig_avg_category_specs, get_ig_category_path, get_ig_model_path
from igibson.utils.ui_utils import choose_from_options, KeyboardRobotController

GRASPING_MODES = OrderedDict(
    sticky="Sticky Mitten - Objects are magnetized when they touch the fingers and a CLOSE command is given",
    # assisted="Assisted Grasping - Objects are fixed when they touch virtual rays cast between each finger and a CLOSE command is given",  # TODO: Not supported in OG yet
    physical="Physical Grasping - No additional grasping assistance applied",
)


def main(random_selection=False, headless=False, short_exec=False):
    """
    Robot grasping mode demo with selection
    Queries the user to select a type of grasping mode and GUI
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    # Choose type of grasping
    grasping_mode = choose_from_options(options=GRASPING_MODES, name="grasping mode", random_selection=random_selection)

    # Create environment configuration to use
    scene_cfg = OrderedDict(type="EmptyScene")
    robot0_cfg = OrderedDict(
        type="Fetch",
        obs_modalities=["rgb"],     # we're just doing a grasping demo so we don't need all observation modalities
        action_type="continuous",
        action_normalize=True,
        grasping_mode=grasping_mode,
    )

    # Compile config
    cfg = OrderedDict(scene=scene_cfg, robots=[robot0_cfg])

    # Create the environment
    env = ig.Environment(configs=cfg, action_timestep=1/60., physics_timestep=1/60.)

    # Load objects (1 table)
    objects_to_load = {
        "table_1": {
            "init_kwargs": {
                "category": "breakfast_table",
                "model": "1b4e6f9dd22a8c628ef9d976af675b86",
                "bounding_box": (0.5, 0.5, 0.8),
                "fit_avg_dim_volume": False,
                "fixed_base": True,
            },
            "init_pose": {
                "position": (0.7, -0.1, 0.6),
                "orientation": (0, 0, 0.707, 0.707),
            },
        },
        "chair_2": {
            "init_kwargs": {
                "category": "straight_chair",
                "model": "2a8d87523e23a01d5f40874aec1ee3a6",
                "bounding_box": None,
                "fit_avg_dim_volume": True,
                "fixed_base": False,
            },
            "init_pose": {
                "position": (0.45, 0.65, 0.425),
                "orientation": (0, 0, -0.9990215, -0.0442276),
            },
        },
    }

    # Load the furniture objects into the simulator
    for obj_name, obj_cfg in objects_to_load.items():
        obj = DatasetObject(
            prim_path=f"/World/{obj_name}",
            name=obj_name,
            **obj_cfg["init_kwargs"],
        )
        ig.sim.import_object(obj)
        obj.set_position_orientation(**obj_cfg["init_pose"])
        ig.sim.step_physics()

    # Now load a box on the table
    box = PrimitiveObject(
        prim_path=f"/World/box",
        name="box",
        primitive_type="Cube",
        rgba=[1.0, 0, 0, 1.0],
        size=0.05,
    )
    ig.sim.import_object(box)
    box.set_position(np.array([0.53, -0.1, 0.97]))
    ig.sim.step_physics()

    # Reset the robot
    robot = env.robots[0]
    robot.set_position([0, 0, 0])
    robot.reset()
    robot.keep_still()

    # Update the simulator's viewer camera's pose so it points towards the robot
    ig.sim.viewer_camera.set_position_orientation(
        position=np.array([-2.39951,  2.26469,  2.66227]),
        orientation=np.array([-0.23898481,  0.48475231,  0.75464013, -0.37204802]),
    )

    # Create teleop controller
    action_generator = KeyboardRobotController(robot=robot)

    # Print out relevant keyboard info if using keyboard teleop
    action_generator.print_keyboard_teleop_info()

    # Other helpful user info
    print("Running demo with grasping mode {}.".format(grasping_mode))
    print("Press ESC to quit")

    # Loop control until user quits
    max_steps = -1 if not short_exec else 100
    step = 0
    while step != max_steps:
        action = action_generator.get_random_action() if random_selection else action_generator.get_teleop_action()
        robot.apply_action(action)
        for _ in range(10):
            env.step(action=action)
            step += 1

    # Always shut down the environment cleanly at the end
    env.close()


if __name__ == "__main__":
    main()
