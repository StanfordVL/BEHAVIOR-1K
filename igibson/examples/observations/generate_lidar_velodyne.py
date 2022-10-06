import logging
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import igibson
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.robots.turtlebot import Turtlebot
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.simulator import Simulator
from igibson.utils.config_utils import parse_config


def main(random_selection=False, headless=False, short_exec=False):
    """
    Example of rendering and visualizing velodyne lidar signals
    Loads Rs (non interactive) and a robot and renders a velodyne signal from the robot's camera
    It plots the velodyne point cloud with matplotlib
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    config = parse_config(os.path.join(igibson.example_config_path, "turtlebot_static_nav.yaml"))
    settings = MeshRendererSettings(enable_shadow=False, msaa=False, texture_scale=0.01)
    s = Simulator(mode="headless", image_width=256, image_height=256, rendering_settings=settings)

    scene = StaticIndoorScene("Rs", build_graph=True)
    s.import_scene(scene)
    robot_config = config["robot"]
    robot_config.pop("name")
    turtlebot = Turtlebot(**robot_config)
    s.import_object(turtlebot)

    turtlebot.apply_action([0.1, -0.1])
    s.step()

    # Get velodyne lidar
    lidar = s.renderer.get_lidar_all()
    logging.info("Dimensions of the lidar observation: {}".format(lidar.shape))

    if not headless:
        # Visualize velodyne lidar
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(lidar[:, 0], lidar[:, 1], lidar[:, 2], s=3)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()

    s.disconnect()


if __name__ == "__main__":
    main()
