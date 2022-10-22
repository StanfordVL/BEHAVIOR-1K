import logging
import os
import sys

import cv2
import numpy as np

from omnigibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from omnigibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from omnigibson.render.profiler import Profiler
from omnigibson.utils.asset_utils import get_scene_path


def main(random_selection=False, headless=False, short_exec=False):
    """
    Creates renderer and renders panorama images in Rs (no interactive). No physics.
    The camera view can be controlled.
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    global _mouse_ix, _mouse_iy, down, view_direction

    # If a model is given, we load it, otherwise we load Rs mesh (non interactive)
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = os.path.join(get_scene_path("Rs"), "mesh_z_up.obj")

    # Create renderer object and load the scene model
    settings = MeshRendererSettings(enable_pbr=False)
    renderer = MeshRenderer(width=512, height=512, rendering_settings=settings)
    renderer.load_object(model_path)
    renderer.add_instance_group([0])

    # Print some information about the loaded model
    logging.info("visual objects {}, instances {}".format(renderer.visual_objects, renderer.instances))
    logging.info("{} {}".format(renderer.material_idx_to_material_instance_mapping, renderer.shape_material_idx))

    # Create a simple viewer with OpenCV and a keyboard navigation
    px = 0
    py = 0.2
    camera_pose = np.array([px, py, 0.5])
    view_direction = np.array([0, -1, -0.3])
    renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
    renderer.set_fov(90)
    _mouse_ix, _mouse_iy = -1, -1
    down = False

    # Define the function callback for OpenCV events on the window
    def change_dir(event, x, y, flags, param):
        global _mouse_ix, _mouse_iy, down, view_direction
        if event == cv2.EVENT_LBUTTONDOWN:
            _mouse_ix, _mouse_iy = x, y
            down = True
        if event == cv2.EVENT_MOUSEMOVE:
            if down:
                dx = (x - _mouse_ix) / 100.0
                dy = (y - _mouse_iy) / 100.0
                _mouse_ix = x
                _mouse_iy = y
                r1 = np.array([[np.cos(dy), 0, np.sin(dy)], [0, 1, 0], [-np.sin(dy), 0, np.cos(dy)]])
                r2 = np.array([[np.cos(-dx), -np.sin(-dx), 0], [np.sin(-dx), np.cos(-dx), 0], [0, 0, 1]])
                view_direction = r1.dot(r2).dot(view_direction)
        elif event == cv2.EVENT_LBUTTONUP:
            down = False

    if not headless:
        cv2.namedWindow("Viewer")
        cv2.setMouseCallback("Viewer", change_dir)
        cv2.namedWindow("Panorama Viewer")

    # Move camera and render normal and panorama images
    max_steps = -1 if not short_exec else 1000
    step = 0
    while step != max_steps:
        with Profiler("Render"):
            frame = renderer.render(modes=("rgb"))
            # Actual panorama image stuff
            img = renderer.get_equi()

        if not headless:
            cv2.imshow("Viewer", cv2.cvtColor(np.concatenate(frame, axis=1), cv2.COLOR_RGB2BGR))
            cv2.imshow("Panorama Viewer", img)

            q = cv2.waitKey(1)
            if q == ord("w"):
                px += 0.01
            elif q == ord("s"):
                px -= 0.01
            elif q == ord("a"):
                py += 0.01
            elif q == ord("d"):
                py -= 0.01
            elif q == ord("q"):
                break
        camera_pose = np.array([px, py, 0.5])
        renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])

        step += 1

    # Cleanup
    renderer.release()


if __name__ == "__main__":
    main()
