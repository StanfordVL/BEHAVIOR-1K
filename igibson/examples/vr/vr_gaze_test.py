""" VR demo to test that eye tracking is working by visualizing a gaze marker where
the user is looking.
"""

import os

import numpy as np

import pybullet_data

import igibson
from igibson.objects.articulated_object import ArticulatedObject
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.robots.behavior_robot import BehaviorRobot
from igibson.scenes.interactive_traversable_scene import InteractiveIndoorScene
from igibson.simulator import Simulator

# HDR files for PBR rendering
hdr_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_02.hdr")
hdr_texture2 = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_03.hdr")
light_modulation_map_filename = os.path.join(
    igibson.ig_dataset_path, "scenes", "Rs_int", "layout", "floor_lighttype_0.png"
)
background_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "urban_street_01.jpg")


def main(random_selection=False, headless=False, short_exec=False):
    # VR rendering settings
    vr_rendering_settings = MeshRendererSettings(
        optimized=True,
        fullscreen=False,
        env_texture_filename=hdr_texture,
        env_texture_filename2=hdr_texture2,
        env_texture_filename3=background_texture,
        light_modulation_map_filename=light_modulation_map_filename,
        enable_shadow=True,
        enable_pbr=True,
        msaa=True,
        light_dimming_factor=1.0,
    )
    s = SimulatorVR(mode="vr", rendering_settings=vr_rendering_settings, vr_settings=VrSettings(use_vr=True))

    scene = InteractiveIndoorScene(
        "Rs_int", load_object_categories=["walls", "floors", "ceilings"], load_room_types=["kitchen"]
    )
    s.import_scene(scene)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    objects = [
        ("jenga/jenga.urdf", (1.300000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107)),
        ("jenga/jenga.urdf", (1.200000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107)),
        ("jenga/jenga.urdf", (1.100000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107)),
        ("jenga/jenga.urdf", (1.000000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107)),
        ("jenga/jenga.urdf", (0.900000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107)),
        ("jenga/jenga.urdf", (0.800000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107)),
        ("table/table.urdf", (1.000000, -0.200000, 0.000000), (0.000000, 0.000000, 0.707107, 0.707107)),
        ("duck_vhacd.urdf", (1.050000, -0.500000, 0.700000), (0.000000, 0.000000, 0.707107, 0.707107)),
        ("duck_vhacd.urdf", (0.950000, -0.100000, 0.700000), (0.000000, 0.000000, 0.707107, 0.707107)),
        ("sphere_small.urdf", (0.850000, -0.400000, 0.700000), (0.000000, 0.000000, 0.707107, 0.707107)),
        ("duck_vhacd.urdf", (0.850000, -0.400000, 1.00000), (0.000000, 0.000000, 0.707107, 0.707107)),
    ]

    for item in objects:
        fpath = item[0]
        pos = item[1]
        orn = item[2]
        item_ob = ArticulatedObject(fpath, scale=1, renderer_params={"use_pbr": False, "use_pbr_mapping": False})
        s.import_object(item_ob)
        item_ob.set_position(pos)
        item_ob.set_orientation(orn)

    obj = ArticulatedObject(
        os.path.join(
            igibson.ig_dataset_path,
            "objects",
            "basket",
            "e3bae8da192ab3d4a17ae19fa77775ff",
            "e3bae8da192ab3d4a17ae19fa77775ff.urdf",
        ),
        scale=2,
    )
    s.import_object(obj)
    obj.set_position_orientation([1.1, 0.300000, 1.0], [0, 0, 0, 1])

    bvr_robot = BehaviorRobot()
    s.import_object(bvr_robot)
    s.register_main_vr_robot(bvr_robot)
    bvr_robot.set_position_orientation([0, 0, 1.5], [0, 0, 0, 1])

    # Represents gaze
    eye_marker = ArticulatedObject(
        "sphere_small.urdf", scale=2, renderer_params={"use_pbr": False, "use_pbr_mapping": False}
    )
    s.import_object(eye_marker)
    gaze_max_dist = 1.5

    # Main simulation loop
    while True:
        # Make sure eye marker never goes to sleep so it is always ready to track gaze
        eye_marker.force_wakeup()
        s.step()

        # Update VR agent using action data from simulator
        bvr_robot.update(s.gen_vr_robot_action())

        # Update gaze marker
        is_valid, origin, dir, _, _ = s.get_eye_tracking_data()
        print("Data validity: {}".format(is_valid))
        if is_valid:
            new_pos = list(np.array(origin) + np.array(dir) * gaze_max_dist)
            eye_marker.set_position(new_pos)

    s.disconnect()


if __name__ == "__main__":
    main()
