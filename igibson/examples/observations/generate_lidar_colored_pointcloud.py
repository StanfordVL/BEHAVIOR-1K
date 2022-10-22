import logging
import os
from sys import platform

import matplotlib.pyplot as plt
import numpy as np
import yaml
from mpl_toolkits.mplot3d import Axes3D

import omnigibson
from omnigibson.envs.omnigibson_env import OmniGibsonEnv


def get_lidar_sampling_pattern():
    lidar_vertical_low = -15 / 180.0 * np.pi
    lidar_vertical_high = 15 / 180.0 * np.pi
    lidar_vertical_n_beams = 16
    lidar_vertical_beams = np.arange(
        lidar_vertical_low,
        lidar_vertical_high + (lidar_vertical_high - lidar_vertical_low) / (lidar_vertical_n_beams - 1),
        (lidar_vertical_high - lidar_vertical_low) / (lidar_vertical_n_beams - 1),
    )

    lidar_horizontal_low = -45 / 180.0 * np.pi
    lidar_horizontal_high = 45 / 180.0 * np.pi
    lidar_horizontal_n_beams = 468
    lidar_horizontal_beams = np.arange(
        lidar_horizontal_low,
        lidar_horizontal_high,
        (lidar_horizontal_high - lidar_horizontal_low) / (lidar_horizontal_n_beams),
    )

    xx, yy = np.meshgrid(lidar_vertical_beams, lidar_horizontal_beams)
    xx = xx.flatten()
    yy = yy.flatten()

    height = 128

    x_samples = (np.tan(xx) / np.cos(yy) * height // 2 + height // 2).astype(np.int)
    y_samples = (np.tan(yy) * height // 2 + height // 2).astype(np.int)

    x_samples = x_samples.flatten()
    y_samples = y_samples.flatten()
    return x_samples, y_samples


x_samples, y_samples = get_lidar_sampling_pattern()


def generate_data_lidar(nav_env, num_samples=3):

    rgb_all = []
    lidar_all = []
    lidar_all_2 = []
    label_all = []

    # Set initial point (not used)
    point = nav_env.scene.get_random_point()[1]

    for _ in range(num_samples):
        # Sample a new point
        new_point = nav_env.scene.get_random_point()[1]
        # If it is too far away, sample again
        while np.linalg.norm(new_point - point) > 1:
            new_point = nav_env.scene.get_random_point()[1]

        # Get vector of distance and change to (y, z, x)
        delta_pos = new_point - point
        delta_pos = np.array([delta_pos[1], delta_pos[2], delta_pos[0]])

        # Set new robot's position
        nav_env.robots[0].set_position(new_point)

        # Get observations (panorama RGB, 3D/Depth and semantic segmentation)
        pano_rgb = nav_env.simulator.renderer.get_cube(mode="rgb", use_robot_camera=True)
        pano_3d = nav_env.simulator.renderer.get_cube(mode="3d", use_robot_camera=True)
        pano_seg = nav_env.simulator.renderer.get_cube(mode="seg", use_robot_camera=True)

        r3 = np.array(
            [[np.cos(-np.pi / 2), 0, -np.sin(-np.pi / 2)], [0, 1, 0], [np.sin(-np.pi / 2), 0, np.cos(-np.pi / 2)]]
        )
        transformatiom_matrix = np.eye(3)

        for i in range(4):
            lidar_all.append(pano_3d[i][:, :, :3].dot(transformatiom_matrix)[x_samples, y_samples] - delta_pos[None, :])
            rgb_all.append(pano_rgb[i][:, :, :3][x_samples, y_samples])
            label_all.append(pano_seg[i][:, :, 0][x_samples, y_samples] * 255.0)
            lidar_all_2.append(
                pano_3d[i][:, :, :3].dot(transformatiom_matrix)[x_samples, y_samples] * 0.9 - delta_pos[None, :]
            )
            transformatiom_matrix = r3.dot(transformatiom_matrix)

    lidar_all = np.concatenate(lidar_all, 0).astype(np.float32)
    lidar_all_2 = np.concatenate(lidar_all_2, 0).astype(np.float32)
    rgb_all = np.concatenate(rgb_all, 0).astype(np.float32)
    label_all = np.concatenate(label_all, 0).astype(np.int32)

    assert len(label_all) == len(label_all)

    direction = lidar_all - lidar_all_2
    direction = direction / (np.linalg.norm(direction, axis=1)[:, None] + 1e-5)

    return lidar_all, direction, rgb_all, label_all


def main(random_selection=False, headless=False, short_exec=False):
    """
    Example of rendering and visualizing a single lidar-like pointcloud
    Loads Rs (non interactive) and a robot and renders a dense panorama depth map from the robot's camera
    Samples the depth map with a lidar-like pattern
    It plots the point cloud with matplotlib, colored with the RGB values
    It also generates segmentation and "direction"
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    # Create environment
    mode = "headless"
    scene_id = "Rs_int"
    config = os.path.join(omnigibson.example_config_path, "fetch_rearrangement.yaml")
    config_data = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    # Reduce texture scale for Mac.
    if platform == "darwin":
        config_data["texture_scale"] = 0.5
    nav_env = OmniGibsonEnv(
        config_file=config_data, mode=mode, scene_id=scene_id, action_timestep=1.0 / 120.0, physics_timestep=1.0 / 120.0
    )

    # Generate data
    pts, direction, color, label = generate_data_lidar(nav_env)

    if not headless:
        # Create visualization: 3D points with RGB color
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], s=3, c=color[:, :3])
        plt.show()


if __name__ == "__main__":
    main()
