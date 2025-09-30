import json
import math
import os
import random
import sys
from typing import Literal

import numpy as np
import tqdm
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_available_g_scenes, get_available_og_scenes
import omnigibson.utils.transform_utils as T
from omnigibson.utils.ui_utils import choose_from_options
from omnigibson.utils.camera_utils import convert_camera_frame_orientation_convention

import h5py
import torch as th
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
            "viewer_width": 1600,
            "viewer_height": 1200,
        },
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "102817200",
            "dataset_type": "hssd",
            "scene_instance": "102817200_best",
            # "load_object_categories": [
            #     "floors",
            #     "walls",
            #     "ceilings",
            #     "lawn",
            #     "driveway",
            #     "roof",
            #     "rail_fence",
            # ],
            "default_erosion_radius": 0.5,  # Erosion radius for the traversable map
            "use_skybox": False,
        },
    }

    # Load the environment
    env = og.Environment(configs=cfg)

    import omnigibson.lazy as lazy

    # lazy.omni.replicator.core.settings.set_render_pathtraced(samples_per_pixel=64)
    lazy.carb.settings.get_settings().set_string("/rtx/rendermode", "PathTracing")
    lazy.carb.settings.get_settings().set_bool("/rtx/useViewLightingMode", True)
    lazy.carb.settings.get_settings().set_bool("/rtx/autoExposure/enabled", True)

    # Do 100 steps of rendering
    for _ in range(5):
        og.sim.render()

    # Get the rooms in the scene and the eroded traversable map
    # rooms = list(env.scene.seg_map.room_ins_name_to_ins_id.keys())
    # trav_map = th.clone(env.scene.trav_map.floor_map[0])
    # trav_map = env.scene.trav_map._erode_trav_map(trav_map)

    index = 0

    TOTAL_IMAGES = 1000
    FLUSH_EVERY = 10
    HEIGHT = og.sim.viewer_height
    WIDTH = og.sim.viewer_width

    output_dir = sys.argv[1]
    hdf5_path = os.path.join(output_dir, "camera_data.hdf5")
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("rgb", shape=(TOTAL_IMAGES, HEIGHT, WIDTH, 4), dtype="uint8")
        f.create_dataset("depth", shape=(TOTAL_IMAGES, HEIGHT, WIDTH), dtype="float32")
        f.create_dataset("segmentation", shape=(TOTAL_IMAGES, HEIGHT, WIDTH), dtype="int32")

        f.create_dataset("camera_pose", shape=(TOTAL_IMAGES, 4, 4), dtype="float32")

        # Warm up.
        for _ in range(5):
            og.sim.render()

        # Record the camera intrinsics
        K = og.sim.viewer_camera.intrinsic_matrix.cpu().numpy()
        f.attrs["camera_intrinsics"] = json.dumps(K.tolist())

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

        # with tqdm.tqdm(total=TOTAL_IMAGES, desc="Collecting images") as pbar:
        while index < TOTAL_IMAGES:
            # Pick a room from the scene, uniformly
            # room_name = random.choice(rooms)
            # if room_name in ("garden_0", "sauna_0", "garage_0"):
            #     continue

            # _, camera_point = env.scene.seg_map.get_random_point_by_room_instance(room_name)
            # if camera_point is None:
            #     continue

            # Check if the camera point is valid (e.g. not within 10cm of a wall)
            # map_coords = env.scene.trav_map.world_to_map(camera_point[:2])
            # if trav_map[map_coords[0], map_coords[1]] == 0:
            #     continue
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

            # Get the rotation as a quaternion
            for jitter_idx in range(1):
                jittered_position = camera_point.clone()
                jittered_rotation = rotation.clone()

                if jitter_idx > 0:
                    # Jitter the camera position slightly by up to 10cm in any direction
                    position_jitter = th.rand(3) * 0.2 - 0.1  # Random jitter in [-0.1, 0.1] meters
                    jittered_position += position_jitter

                    # Jitter the orientation too. Randomly sample an axis and a small angle.
                    axis = th.rand(3)
                    axis /= th.norm(axis)
                    angle = th.deg2rad(th.rand(1) * 20 - 10)  # Random angle of up to 10 degrees
                    rotation_jitter = T.axisangle2quat(axis * angle)
                    jittered_rotation = T.quat_multiply(jittered_rotation, rotation_jitter)

                # Set the camera pose
                og.sim.viewer_camera.set_position_orientation(position=jittered_position, orientation=jittered_rotation)

                # Render 5 times to ensure the camera is stable
                for _ in range(5):
                    og.sim.render()

                # Get the observation from the viewer camera sensor
                rgb = og.sim.viewer_camera.get_obs()[0]["rgb"].detach().clone().cpu().numpy()
                depth = og.sim.viewer_camera.get_obs()[0]["depth_linear"].detach().clone().cpu().numpy()
                seg = og.sim.viewer_camera.get_obs()[0]["seg_instance"].detach().clone().cpu().numpy()

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
                if np.mean(depth) < 2:
                    continue

                # Save the image to a file
                image_file = f"camera_{index:04d}_yaw_{yaw}_pitch_{int(pitch)}.png"
                Image.fromarray(rgb).save(os.path.join(images_dir, image_file))

                # Save the data into hdf5
                f["rgb"][index] = rgb
                f["depth"][index] = depth
                f["segmentation"][index] = seg

                f["camera_pose"][index] = T.pose2mat(og.sim.viewer_camera.get_position_orientation()).cpu().numpy()

                # Flush every N entries
                if index % FLUSH_EVERY == 0:
                    f.flush()

                index += 1
                print(index)

                if index >= TOTAL_IMAGES:
                    break

        # Record the segmentation keys
        f.attrs["segmentation_labels"] = json.dumps(og.sim.viewer_camera.get_obs()[1]["seg_instance"])

    # Always close the environment at the end
    og.clear()


if __name__ == "__main__":
    main()
