import logging
import os
from sys import platform

import matplotlib.pyplot as plt
import numpy as np
import yaml
from mpl_toolkits.mplot3d import Axes3D

import omnigibson
from omnigibson.envs.omnigibson_env import OmniGibsonEnv


def main(random_selection=False, headless=False, short_exec=False):
    """
    Example of rendering and visualizing a single 3D dense pointcloud
    Loads Rs (non interactive) and a robot and renders a dense panorama depth map from the robot's camera
    It plots the point cloud with matplotlib, colored with the RGB values
    It also generates semantic segmentation
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    # Create OmniGibsonEnvironment with the Fetch rearrangement config
    mode = "headless"
    config = os.path.join(omnigibson.example_config_path, "fetch_rearrangement.yaml")
    config_data = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    # Reduce texture scale for Mac.
    if platform == "darwin":
        config_data["texture_scale"] = 0.5
    scene_id = "Rs_int"
    nav_env = OmniGibsonEnv(
        config_file=config_data, mode=mode, scene_id=scene_id, action_timestep=1.0 / 120.0, physics_timestep=1.0 / 120.0
    )

    # Generate data ####################################
    # First set the robot in a random point in the scene
    point = nav_env.scene.get_random_point()[1]
    nav_env.robots[0].set_position(point)
    nav_env.simulator.sync()

    # Then generate RGB, semantic segmentation and 3D/depth, in panorama format
    pano_rgb = nav_env.simulator.renderer.get_equi(mode="rgb", use_robot_camera=True)
    pano_seg = nav_env.simulator.renderer.get_equi(mode="seg", use_robot_camera=True)
    pano_3d = nav_env.simulator.renderer.get_equi(mode="3d", use_robot_camera=True)
    depth = np.linalg.norm(pano_3d[:, :, :3], axis=2)
    theta = -np.arange(-np.pi / 2, np.pi / 2, np.pi / 128.0)[:, None]
    phi = np.arange(-np.pi, np.pi, 2 * np.pi / 256)[None, :]
    # depth = depth / np.cos(theta)
    x = np.cos(theta) * np.sin(phi) * depth
    y = np.sin(theta) * depth
    z = np.cos(theta) * np.cos(phi) * depth

    # Select only a few points
    mask = np.random.uniform(size=(128, 256)) > 0.5

    pts = (np.stack([x[mask], y[mask], z[mask]]).T).astype(np.float32)
    color = (pano_rgb[mask][:, :3]).astype(np.float32)
    label = (pano_seg[mask][:, 0]).astype(np.int32)

    assert len(pts) == len(label)

    # Create visualization: 3D points with RGB color
    if not headless:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], s=3, c=color[:, :3])
        plt.show()


if __name__ == "__main__":
    main()
