import logging
import os
import platform
import sys

import matplotlib.pyplot as plt
import numpy as np

from igibson.render.mesh_renderer.mesh_renderer_tensor import MeshRendererG2G
from igibson.render.profiler import Profiler
from igibson.utils.asset_utils import get_scene_path


def main(random_selection=False, headless=False, short_exec=False):
    """
    Example of usage of the direct GPU rendering, tensor-to-tensor
    ONLY ON LINUX
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    if platform.system() != "Linux":
        logging.error("Rendering to pytorch tensor is only available on Linux.")
        sys.exit(0)

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = os.path.join(get_scene_path("Rs"), "mesh_z_up.obj")

    renderer = MeshRendererG2G(width=512, height=512, device_idx=0)
    renderer.load_object(model_path)
    renderer.add_instance_group([0])

    logging.info("visual objects {}, instances {}".format(renderer.visual_objects, renderer.instances))
    logging.info("{} {}".format(renderer.material_idx_to_material_instance_mapping, renderer.shape_material_idx))
    camera_pose = np.array([0, 0, 1.2])
    view_direction = np.array([1, 0, 0])
    renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
    renderer.set_fov(90)

    max_steps = 3000 if not short_exec else 100
    for i in range(max_steps):
        with Profiler("Render"):
            frame = renderer.render(modes=("rgb", "normal", "3d"))

    logging.info(frame)
    img_np = frame[0].flip(0).data.cpu().numpy().reshape(renderer.height, renderer.width, 4)
    normal_np = frame[1].flip(0).data.cpu().numpy().reshape(renderer.height, renderer.width, 4)

    if not headless:
        plt.imshow(np.concatenate([img_np, normal_np], axis=1))
        plt.show()


if __name__ == "__main__":
    main()
