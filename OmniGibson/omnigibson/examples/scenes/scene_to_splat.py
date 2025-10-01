import json
import math
import os
import random
import sys

import numpy as np
import tqdm
import omnigibson as og
from omnigibson.macros import gm
import omnigibson.utils.transform_utils as T
from omnigibson.utils.camera_utils import convert_camera_frame_orientation_convention

import torch as th
from scipy.spatial.transform import Rotation as R
from PIL import Image

# Configure macros for maximum performance
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True
gm.ENABLE_OBJECT_STATES = False
gm.ENABLE_TRANSITION_RULES = False
gm.HEADLESS = True

DEG2RAD = math.pi / 180.0


def main():
    """
    Prompts the user to select any available interactive scene and loads a turtlebot into it.
    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    cfg = {
        "render": {
            "viewer_width": 1280,
            "viewer_height": 720,
        },
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "dataset_name": "behavior-1k-assets",
            "scene_instance": "Rs_int_with_clutter",
            # "load_object_categories": [
            #     "floors",
            #     "walls",
            #     "ceilings",
            #     "lawn",
            #     "driveway",
            #     "roof",
            #     "rail_fence",
            # ],
            "use_skybox": False,
        },
    }

    # Load the environment
    env = og.Environment(configs=cfg)

    import omnigibson.lazy as lazy

    # lazy.omni.replicator.core.settings.set_render_pathtraced(samples_per_pixel=64)
    lazy.carb.settings.get_settings().set_string("/rtx/rendermode", "PathTracing")
    lazy.carb.settings.get_settings().set_bool("/rtx/reflections/enabled", False)
    lazy.carb.settings.get_settings().set_bool("/rtx/useViewLightingMode", True)
    lazy.carb.settings.get_settings().set_bool("/rtx/post/histogram/enabled", True)
    lazy.carb.settings.get_settings().set_float("/rtx/post/histogram/whiteScale", 5.0)

    # Do 100 steps of rendering
    for _ in range(5):
        og.sim.render()

    index = 0

    TOTAL_IMAGES = 5000

    output_dir = sys.argv[1]

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)

    segmentation_dir = os.path.join(output_dir, "seg")
    os.makedirs(segmentation_dir, exist_ok=True)

    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    # Warm up.
    for _ in range(5):
        og.sim.render()

    # Record the camera intrinsics
    K = og.sim.viewer_camera.intrinsic_matrix.cpu().numpy()
    np.save(os.path.join(output_dir, "camera_intrinsics.npy"), K)

    # Calculate scene AABB
    scene_aabb_min, scene_aabb_max = None, None
    for obj in env.scene.objects:
        aabb_min, aabb_max = obj.aabb
        if scene_aabb_min is None:
            scene_aabb_min = aabb_min
            scene_aabb_max = aabb_max
        else:
            scene_aabb_min = th.minimum(scene_aabb_min, aabb_min)
            scene_aabb_max = th.maximum(scene_aabb_max, aabb_max)

    poses_so_far = []
    with tqdm.tqdm(total=TOTAL_IMAGES, desc="Collecting images") as pbar:
        while index < TOTAL_IMAGES:
            # Pick a random point in the scene AABB
            camera_point = scene_aabb_min + th.rand(3) * (scene_aabb_max - scene_aabb_min)

            # Pick a random height
            camera_point[2] = random.uniform(1.5, 2.0)  # Height in meters

            # Now iterate through the camera orientations.
            # for yaw in range(0, 360, 45):
            yaw = random.uniform(0, 360)  # Randomly pick a yaw angle between 0 and 360 degrees

            # Randomly pick a pitch angle between -45 and 0 degrees
            pitch = random.uniform(45, 0)

            # Compute the camera rotation
            rotation = T.euler2quat(th.tensor([0, pitch * DEG2RAD, yaw * DEG2RAD], dtype=th.float32))
            rotation = convert_camera_frame_orientation_convention(rotation, "world", "opengl")

            # Set the camera pose
            og.sim.viewer_camera.set_position_orientation(position=camera_point, orientation=rotation)

            # Render 5 times to ensure the camera is stable
            for _ in range(5):
                og.sim.render()

            # Get the observation from the viewer camera sensor
            obs, obs_info = og.sim.viewer_camera.get_obs()
            rgb = obs["rgb"].detach().cpu().numpy()
            depth = obs["depth_linear"].detach().cpu().numpy()
            seg = obs["seg_instance"].detach().cpu().numpy()

            # Check that in any given image at least 3 different objects are visible by at least 1% of the total area
            unique_objects, counts = np.unique(seg.flatten(), return_counts=True)
            if len(unique_objects) < 3:
                continue
            counts = counts[unique_objects > 0]  # Ignore background
            object_areas = counts / (rgb.shape[0] * rgb.shape[1])
            # if np.sum(object_areas > 0.01) < 3:
            #     continue

            # Check that no object takes up more than 90% of the image area
            if np.any(object_areas > 0.9):
                continue

            # Check that the average depth is not less than a constant.
            if np.mean(depth) < 1:
                continue

            # Check if we are too close to any particular pose (within 5cm and 15 degrees)
            too_close = False
            rot_obj = R.from_quat(rotation.numpy())
            inv_rot_obj = rot_obj.inv()
            for prev_pos, prev_rot in poses_so_far:
                pos_difference = th.norm(prev_pos - camera_point).item()
                orn_difference = (inv_rot_obj * prev_rot).magnitude()
                if pos_difference < 0.05 and orn_difference < 15 * DEG2RAD:
                    too_close = True
                    break
            if too_close:
                continue

            # Save the image to a file
            image_file = f"{index:05d}.png"
            Image.fromarray(rgb).save(os.path.join(images_dir, image_file))

            # Save the depth and segmentation as numpy arrays
            np.save(os.path.join(depth_dir, image_file.replace(".png", ".npy")), depth)
            np.save(os.path.join(segmentation_dir, image_file.replace(".png", ".npy")), seg)

            # Save other metadata
            camera_pose = T.pose2mat(og.sim.viewer_camera.get_position_orientation()).cpu().numpy()
            metadata = {"pose": camera_pose.tolist(), "yaw": yaw, "pitch": pitch}
            with open(os.path.join(metadata_dir, image_file.replace(".png", ".json")), "w") as f:
                json.dump(metadata, f, indent=4)

            # Record the pose for checking later
            poses_so_far.append((camera_point, rot_obj))

            index += 1
            pbar.update(1)

            # Overwrite the segmentation keys
            with open(os.path.join(output_dir, "segmentation_keys.json"), "w") as f:
                json.dump(obs_info["seg_instance"], f)

    # Always close the environment at the end
    og.clear()


if __name__ == "__main__":
    main()
