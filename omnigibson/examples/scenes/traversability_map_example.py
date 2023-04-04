import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import omnigibson as og
from omnigibson.utils.asset_utils import get_og_scene_path, get_available_og_scenes
from omnigibson.utils.ui_utils import choose_from_options


def main(random_selection=False, headless=False, short_exec=False):
    """
    Traversable map demo
    Loads the floor plan and obstacles for the requested scene, and overlays them in a visual figure such that the
    highlighted area reflects the traversable (free-space) area
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    scenes = get_available_og_scenes()
    scene_model = choose_from_options(options=scenes, name="scene model", random_selection=random_selection)
    print(f"Generating traversability map for scene {scene_model}")

    trav_map_size = 200
    trav_map_erosion = 2

    trav_map = Image.open(os.path.join(get_og_scene_path(scene_model), "layout", "floor_trav_0.png"))
    trav_map = np.array(trav_map.resize((trav_map_size, trav_map_size)))
    trav_map = cv2.erode(trav_map, np.ones((trav_map_erosion, trav_map_erosion)))

    if not headless:
        plt.figure(figsize=(12, 12))
        plt.imshow(trav_map)
        plt.title(f"Traversable area of {scene_model} scene")

    if not headless:
        plt.show()

    # Shut down omnigibson at the end
    og.shutdown()


if __name__ == "__main__":
    main()
