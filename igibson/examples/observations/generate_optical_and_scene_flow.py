import logging
import os
from sys import platform

import cv2
import numpy as np
import yaml

import omnigibson
from omnigibson.envs.omnigibson_env import OmniGibsonEnv
from omnigibson.objects.ycb_object import YCBObject
from omnigibson.render.profiler import Profiler
from omnigibson.utils.constants import MAX_CLASS_COUNT, MAX_INSTANCE_COUNT
from omnigibson.utils.vision_utils import randomize_colors, segmentation_to_rgb

FLOW_SCALING_FACTOR = 500


def optical_flow_to_visualization(image):
    hsv = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(image[..., 0], image[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = np.clip(mag * FLOW_SCALING_FACTOR, 0, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def scene_flow_to_visualization(image):
    return np.clip((image * FLOW_SCALING_FACTOR + 127), 0, 255).astype(np.uint8)


def main(random_selection=False, headless=True, short_exec=True):
    """
    Example of rendering additional sensor modalities
    Loads Rs_int (interactive) with some objects and and renders depth, normals, semantic and instance segmentation
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    config_filename = os.path.join(omnigibson.example_config_path, "turtlebot_static_nav.yaml")
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config_data["vertical_fov"] = 90
    config_data["output"] = ["task_obs", "rgb", "optical_flow", "scene_flow"]
    config_data["optimized_renderer"] = False
    config_data["enable_shadow"] = False
    # Reduce texture scale for Mac.
    if platform == "darwin":
        config_data["texture_scale"] = 0.5
    env = OmniGibsonEnv(config_file=config_data, mode="gui_interactive" if not headless else "headless")

    if not headless:
        # Set a better viewing direction
        env.simulator.viewer.initial_pos = [0, 1.1, 1.5]
        env.simulator.viewer.initial_view_direction = [0, 1, 0.1]
        env.simulator.viewer.reset_viewer()

    added_objects = []

    for _ in range(10):
        obj = YCBObject("003_cracker_box")
        env.simulator.import_object(obj)
        obj.set_position_orientation(np.append(np.random.uniform(low=0, high=2, size=2), [1.8]), [0, 0, 0, 1])
        added_objects.append(obj)

    if short_exec:
        episodes = 1
    else:
        episodes = 10

    for j in range(episodes):
        logging.info("Resetting environment")
        env.reset()
        env.robots[0].set_position_orientation([0, 0, 0], [0, 0, 0, 1])

        for obj in added_objects:
            obj.set_position_orientation(np.append(np.random.uniform(low=0, high=2, size=2), [1.8]), [0, 0, 0, 1])
        for i in range(50):
            with Profiler("Environment action step"):
                # let the robot stay static, and objects will fall around the robot
                state, reward, done, info = env.step([0, 0])

                optical_flow = state["optical_flow"]
                scene_flow = state["scene_flow"]

                if not headless:
                    cv2.imshow("Optical Flow", optical_flow_to_visualization(optical_flow))
                    cv2.imshow("Scene Flow", scene_flow_to_visualization(scene_flow))

    env.close()


if __name__ == "__main__":
    main()
