import logging
import os
import sys

import cv2
import numpy as np

import omnigibson
from omnigibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from omnigibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from omnigibson.render.profiler import Profiler


def load_obj_np(filename_obj, normalization=False, texture_size=4, texture_wrapping="REPEAT", use_bilinear=True):
    """
    Load Wavefront OBJ file into numpy array
    This function only supports vertices (v x x x) and faces (f x x x).
    """
    # load vertices
    vertices = []
    with open(filename_obj) as f:
        lines = f.readlines()

    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == "v":
            vertices.append([float(v) for v in line.split()[1:4]])
    vertices = np.vstack(vertices).astype(np.float32)

    # load faces
    faces = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == "f":
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split("/")[0])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split("/")[0])
                v2 = int(vs[i + 2].split("/")[0])
                faces.append((v0, v1, v2))
    faces = np.vstack(faces).astype(np.int32) - 1
    assert normalization is False
    return vertices, faces


def main(random_selection=False, headless=False, short_exec=False):
    """
    Minimal example of use of the PBR renderer. Loads Rs_int (interactive) or one object.
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    global _mouse_ix, _mouse_iy, down, view_direction

    # Assuming that if random_selection=True, headless=True, short_exec=True, we are calling it from tests
    if len(sys.argv) > 1 and not (random_selection and headless and short_exec):
        model_path = sys.argv[1]
    else:
        model_path = os.path.join(omnigibson.og_dataset_path, "scenes", "Rs_int", "shape", "visual")
    settings = MeshRendererSettings(msaa=True, enable_shadow=True)
    renderer = MeshRenderer(width=512, height=512, vertical_fov=70, rendering_settings=settings)
    renderer.set_light_position_direction([0, 0, 10], [0, 0, 0])

    i = 0
    v = []

    # If a model path is given and it is an OBJ file we load it
    for fn in os.listdir(model_path):
        if fn.endswith("obj"):
            vertices, faces = load_obj_np(os.path.join(model_path, fn))
            v.append(vertices)

    v = np.vstack(v)
    logging.info("Number of vertices {}".format(v.shape))
    xlen = np.max(v[:, 0]) - np.min(v[:, 0])
    ylen = np.max(v[:, 1]) - np.min(v[:, 1])
    scale = 2.0 / (max(xlen, ylen))

    # If a model path is given and it is an OBJ file we load it
    for fn in os.listdir(model_path):
        if fn.endswith("obj"):
            renderer.load_object(os.path.join(model_path, fn), scale=[scale, scale, scale])
            renderer.add_instance_group([i])
            i += 1

    # Logging some info
    logging.info("renderer.visual_objects {}".format(renderer.visual_objects))
    logging.info("renderer.instances {}".format(renderer.instances))
    logging.info(
        "renderer.material_idx_to_material_instance_mapping {}".format(
            renderer.material_idx_to_material_instance_mapping
        )
    )
    logging.info("renderer.shape_material_idx {}".format(renderer.shape_material_idx))

    # Create a simple viewer with OpenCV
    px = 1
    py = 1
    pz = 1
    camera_pose = np.array([px, py, pz])
    view_direction = np.array([-1, -1, -1])
    renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
    _mouse_ix, _mouse_iy = -1, -1
    down = False

    # Callback function for OpenCV viewer
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

    # Move camera and render
    max_steps = -1 if not short_exec else 1000
    step = 0
    while step != max_steps:
        with Profiler("Render"):
            frame = renderer.render(modes=("rgb", "normal", "seg", "ins_seg"))

        if not headless:
            cv2.imshow("Viewer", cv2.cvtColor(np.concatenate(frame, axis=1), cv2.COLOR_RGB2BGR))

            q = cv2.waitKey(1)
            if q == ord("w"):
                px += 0.1
            elif q == ord("s"):
                px -= 0.1
            elif q == ord("a"):
                py += 0.1
            elif q == ord("d"):
                py -= 0.1
            elif q == ord("q"):
                break

        camera_pose = np.array([px, py, 1])
        renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])

        step += 1

    renderer.release()


if __name__ == "__main__":
    main()
