import colorsys
import logging

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.scenes.interactive_traversable_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
from igibson.utils.constants import MAX_CLASS_COUNT, MAX_INSTANCE_COUNT
from igibson.utils.vision_utils import segmentation_to_rgb


def main(random_selection=False, headless=False, short_exec=False):
    """
    Example of generating a topdown semantic segmentation map
    Loads Rs_int (interactive) with all or some objects (code can be commented)
    and and renders a semantic segmentation image top-down
    This is also an example of how to load a scene without the ceiling to facilitate creating visualizations
    Quit with q
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    settings = MeshRendererSettings(enable_shadow=False, msaa=False, texture_scale=0.01)
    s = Simulator(mode="headless", image_width=512, image_height=512, device_idx=0, rendering_settings=settings)
    scene = InteractiveIndoorScene(
        "Rs_int",
        texture_randomization=False,
        object_randomization=False,
        # load_object_categories=[
        #     "breakfast_table",
        #     "carpet",
        #     "sofa",
        #     "bottom_cabinet",
        #     "sink",
        #     "stove",
        #     "fridge",
        # ],
        not_load_object_categories=["ceilings"],
    )
    s.import_scene(scene)
    camera_pose = np.array([0, 0, 6.0])
    view_direction = np.array([0, 0, -1.0])
    s.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 1, 0])
    seg = s.renderer.render(modes=("seg"))[0]
    seg_int = np.round(seg[:, :, 0] * MAX_CLASS_COUNT).astype(np.int32)
    seg_color = segmentation_to_rgb(seg_int, MAX_CLASS_COUNT)

    if not headless:
        max_steps = -1 if not short_exec else 1000
        step = 0
        while step != max_steps:
            cv2.imshow("Topdown Semantic Segmentation", seg_color)
            q = cv2.waitKey(1)
            if q == ord("q"):
                break
            step += 1

    s.disconnect()


if __name__ == "__main__":
    main()
