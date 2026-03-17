import argparse
import math

import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.utils.asset_utils import get_all_object_categories, get_all_object_category_models
from omnigibson.utils.ui_utils import choose_from_options


def main(random_selection=False, headless=False, short_exec=False):
    """
    Visualizes object as specified by its USD path, @usd_path. If None if specified, will instead
    result in an object selection from OmniGibson's object dataset
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

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

    # Define objects to load
    light0_cfg = dict(
        type="LightObject",
        light_type="Sphere",
        name="sphere_light0",
        radius=0.01,
        intensity=1e5,
        position=[-2.0, -2.0, 2.0],
    )

    light1_cfg = dict(
        type="LightObject",
        light_type="Sphere",
        name="sphere_light1",
        radius=0.01,
        intensity=1e5,
        position=[-2.0, 2.0, 2.0],
    )

    # Make sure we have a valid usd path
    if usd_path is None:
        # Select a category to load
        available_obj_categories = get_all_object_categories()
        obj_category = choose_from_options(
            options=available_obj_categories, name="object category", random_selection=random_selection
        )

        # Select a model to load
        available_obj_models = get_all_object_category_models(obj_category)
        obj_model = choose_from_options(
            options=available_obj_models, name="object model", random_selection=random_selection
        )

        kwargs = {
            "type": "DatasetObject",
            "category": obj_category,
            "model": obj_model,
        }
    else:
        kwargs = {
            "type": "USDObject",
            "usd_path": usd_path,
        }

    # Import the desired object
    obj_cfg = dict(
        **kwargs,
        name="obj",
        usd_path=usd_path,
        visual_only=True,
        position=[0, 0, 10.0],
    )

    # Create the scene config to load -- empty scene
    cfg = {
        "scene": {
            "type": "Scene",
        },
        "objects": [light0_cfg, light1_cfg, obj_cfg],
    }

    # Create the environment
    env = og.Environment(configs=cfg)

    # Set camera to appropriate viewing pose
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([-0.00913503, -1.95750906, 1.36407314]),
        orientation=th.tensor([0.6350064, 0.0, 0.0, 0.77250687]),
    )

    # Grab the object references
    obj = env.scene.object_registry("name", "obj")

    # Standardize the scale of the object so it fits in a [1,1,1] box -- note that we have to stop the simulator
    # in order to set the scale
    extents = obj.aabb_extent
    og.sim.stop()
    obj.scale = (th.ones(3) / extents).min()
    og.sim.play()
    env.step(th.empty(0))

    # Move the object so that its center is at [0, 0, 1]
    center_offset = obj.get_position_orientation()[0] - obj.aabb_center + th.tensor([0, 0, 1.0])
    obj.set_position_orientation(position=center_offset)

    # Allow the user to easily move the camera around
    og.sim.enable_viewer_camera_teleoperation()

    # Rotate the object in place
    steps_per_rotate = 360
    steps_per_joint = steps_per_rotate / 10
    max_steps = 100 if short_exec else 10000
    for i in range(max_steps):
        z_angle = 2 * math.pi * (i % steps_per_rotate) / steps_per_rotate
        quat = T.euler2quat(th.tensor([0, 0, z_angle]))
        pos = T.quat2mat(quat) @ center_offset
        if obj.n_dof > 0:
            frac = (i % steps_per_joint) / steps_per_joint
            j_frac = -1.0 + 2.0 * frac if (i // steps_per_joint) % 2 == 0 else 1.0 - 2.0 * frac
            obj.set_joint_positions(positions=j_frac * th.ones(obj.n_dof), normalized=True, drive=False)
            obj.keep_still()
        obj.set_position_orientation(position=pos, orientation=quat)
        env.step(th.empty(0))

    # Shut down at the end
    og.shutdown()


if __name__ == "__main__":
    main()
