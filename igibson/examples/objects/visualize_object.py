import argparse
import logging

import numpy as np

import igibson as ig
from igibson.objects import USDObject, LightObject
from igibson.utils.asset_utils import (
    get_all_object_categories,
    get_object_models_of_category,
)
from igibson.utils.ui_utils import choose_from_options
import igibson.utils.transform_utils as T


def main(random_selection=False, headless=False, short_exec=False):
    """
    Visualizes object as specified by its USD path, @usd_path. If None if specified, will instead
    result in an object selection from iGibson's object dataset
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    # Assuming that if random_selection=True, headless=True, short_exec=True, we are calling it from tests and we
    # do not want to parse args (it would fail because the calling function is pytest "testfile.py")
    usd_path = None
    if not (random_selection and headless and short_exec):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--usd_path",
            default=None,
            help="USD Model to load",
        )
        args = parser.parse_args()
        usd_path = args.usd_path

    # Create the scene config to load -- empty scene
    cfg = {
        "scene": {
            "type": "EmptyScene",
            # "floor_plane_visible": False,
        }
    }

    # Create the environment
    env = ig.Environment(configs=cfg, action_timestep=1/60., physics_timestep=1/60.)

    # Set camera to appropriate viewing pose
    ig.sim.viewer_camera.set_position_orientation(
        position=np.array([-0.00913503, -1.95750906,  1.36407314]),
        orientation=np.array([0.63468727, 0.02012955, 0.02448817, 0.77211864]),
    )

    # Create a light object
    light0 = LightObject(
        prim_path="/World/sphere_light0",
        light_type="Sphere",
        name="sphere_light0",
        radius=0.01,
        intensity=1e5,
    )
    ig.sim.import_object(light0)
    light0.set_position(np.array([-2.0, -2.0, 2.0]))

    light1 = LightObject(
        prim_path="/World/sphere_light1",
        light_type="Sphere",
        name="sphere_light1",
        radius=0.01,
        intensity=1e5,
    )
    ig.sim.import_object(light1)
    light1.set_position(np.array([-2.0, 2.0, 2.0]))

    # Make sure we have a valid usd path
    if usd_path is None:
        # Select a category to load
        available_obj_categories = get_all_object_categories()
        obj_category = choose_from_options(options=available_obj_categories, name="object category", random_selection=random_selection)

        # Select a model to load
        available_obj_models = get_object_models_of_category(obj_category)
        obj_model = choose_from_options(options=available_obj_models, name="object model", random_selection=random_selection)

        usd_path = f"{ig.ig_dataset_path}/objects/{obj_category}/{obj_model}/usd/{obj_model}.usd"

    # Import the desired object
    obj = USDObject(
        prim_path="/World/obj",
        name="obj",
        usd_path=usd_path,
        visual_only=True,
    )
    ig.sim.import_object(obj)

    # Standardize the scale of the object so it fits in a [1,1,1] box
    extents = obj.aabb_extent
    obj.scale = (np.ones(3) / extents).min()
    env.step(np.array([]))

    # Move the object so that its center is at [0, 0, 1]
    center_offset = -obj.aabb_center + np.array([0, 0, 1.0])
    obj.set_position(center_offset)

    # Allow the user to easily move the camera around
    ig.sim.enable_viewer_camera_teleoperation()

    # Rotate the object in place
    max_steps = 100 if short_exec else 10000
    for i in range(max_steps):
        z_angle = (2 * np.pi * (i % 200) / 200)
        quat = T.euler2quat(np.array([0, 0, z_angle]))
        pos = T.quat2mat(quat) @ center_offset
        obj.set_position_orientation(position=pos, orientation=quat)
        env.step(np.array([]))


if __name__ == "__main__":
    main()
