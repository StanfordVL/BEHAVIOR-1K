import logging
import os
import sys

import cv2
import numpy as np

from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from igibson.utils.asset_utils import get_scene_path


def main(random_selection=False, headless=False, short_exec=False):
    """
    Minimal example of use of the renderer. Loads Rs_int, renders one set of images (RGB, normals,
    3D points (as depth)), shows them.
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    # If a model is given, we load it, otherwise we load Rs mesh (non interactive)
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = os.path.join(get_scene_path("Rs"), "mesh_z_up.obj")

    renderer = MeshRenderer(width=512, height=512)
    renderer.load_object(model_path)
    renderer.add_instance_group([0])
    camera_pose = np.array([0, 0, 1.2])
    view_direction = np.array([1, 0, 0])
    renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
    renderer.set_fov(90)
    frames = renderer.render(modes=("rgb", "normal", "3d"))

    # Render 3d points as depth map
    depth = np.linalg.norm(frames[2][:, :, :3], axis=2)
    depth /= depth.max()
    frames[2][:, :, :3] = depth[..., None]

    if not headless:
        frames = cv2.cvtColor(np.concatenate(frames, axis=1), cv2.COLOR_RGB2BGR)
        cv2.imshow("image", frames)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
