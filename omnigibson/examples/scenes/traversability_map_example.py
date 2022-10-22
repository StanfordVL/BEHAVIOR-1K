import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from omnigibson.utils.asset_utils import get_scene_path


def main(random_selection=False, headless=False, short_exec=False):
    """
    Traversable map demo
    Loads the floor plan and obstacles for the Rs scene, and overlays them in a visual figure such that the
    highlighted area reflects the traversable (free-space) area
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    scene_id = "Rs"
    trav_map_size = 200
    trav_map_erosion = 2

    with open(os.path.join(get_scene_path(scene_id), "floors.txt"), "r") as f:
        floors = sorted(list(map(float, f.readlines())))
        logging.info("Floor heights: {}".format(floors))

    for f in range(len(floors)):
        trav_map = Image.open(os.path.join(get_scene_path(scene_id), "floor_trav_{}.png".format(f)))
        obstacle_map = Image.open(os.path.join(get_scene_path(scene_id), "floor_{}.png".format(f)))
        trav_map = np.array(trav_map.resize((trav_map_size, trav_map_size)))
        obstacle_map = np.array(obstacle_map.resize((trav_map_size, trav_map_size)))
        trav_map[obstacle_map == 0] = 0
        trav_map = cv2.erode(trav_map, np.ones((trav_map_erosion, trav_map_erosion)))

        if not headless:
            plt.figure(f, figsize=(12, 12))
            plt.imshow(trav_map)
            plt.title("Traversable area of {} scene".format(scene_id))

    if not headless:
        plt.show()


if __name__ == "__main__":
    main()
