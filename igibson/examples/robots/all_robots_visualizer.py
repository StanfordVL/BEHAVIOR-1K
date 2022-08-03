import logging
import os

import numpy as np

import igibson
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.robots import REGISTERED_ROBOTS
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.config_utils import parse_config


def main(random_selection=False, headless=False, short_exec=False):
    """
    Robot demo
    Loads all robots in an empty scene, generate random actions
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    # Create empty scene
    settings = MeshRendererSettings(enable_shadow=False, msaa=False, texture_scale=0.5)
    s = Simulator(
        mode="gui_interactive" if not headless else "headless",
        image_width=512,
        image_height=512,
        rendering_settings=settings,
    )
    scene = EmptyScene(floor_plane_rgba=[0.6, 0.6, 0.6, 1])
    s.import_scene(scene)

    # Create one instance of each robot aligned along the y axis
    position = [0, 0, 0]
    robots = {}
    for robot_config_file in os.listdir(os.path.join(igibson.example_config_path, "robots")):
        config = parse_config(os.path.join(igibson.example_config_path, "robots", robot_config_file))
        robot_config = config["robot"]
        robot_name = robot_config.pop("name")
        robot = REGISTERED_ROBOTS[robot_name](**robot_config)
        s.import_object(robot)
        robot.set_position(position)
        robot.reset()
        robot.keep_still()
        robots[robot_name] = (robot, position[1])
        logging.info("Loaded " + robot_name)
        logging.info("Moving " + robot_name)

        if not headless:
            # Set viewer in front
            s.viewer.initial_pos = [1.6, 0, 1.3]
            s.viewer.initial_view_direction = [-0.7, 0, -0.7]
            s.viewer.reset_viewer()

        for _ in range(100):  # keep still for 10 seconds
            s.step()

        for _ in range(30):
            action = np.random.uniform(-1, 1, robot.action_dim)
            robot.apply_action(action)
            for _ in range(10):
                s.step()

        robot.keep_still()
        s.reload()
        scene = EmptyScene(floor_plane_rgba=[0.6, 0.6, 0.6, 1])
        s.import_scene(scene)

    s.disconnect()


if __name__ == "__main__":
    main()
