"""
Demo to show a sample VR HUD (heads-up-display), constructed using
VR overlay classes (found in render/mesh_renderer/mesh_renderer_vr.py) and Text (found in render/mesh_renderer/text.py)
"""
import os
import time


import pybullet_data

import igibson
from igibson.objects.articulated_object import ArticulatedObject
from igibson.objects.ycb_object import YCBObject
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.robots.behavior_robot import BehaviorRobot
from igibson.scenes.interactive_traversable_scene import InteractiveIndoorScene
from igibson.simulator import Simulator

# Objects in the benchmark - corresponds to Rs kitchen environment, for range of items and
# transferability to the real world
# Note: the scene will automatically load in walls/ceilings/floors in addition to these objects
benchmark_names = [
    "door_54",
    "trash_can_25",
    "counter_26",
    "bottom_cabinet_39",
    "fridge_40",
    "bottom_cabinet_41",
    "sink_42",
    "microwave_43",
    "dishwasher_44",
    "oven_45",
    "bottom_cabinet_46",
    "top_cabinet_47",
    "top_cabinet_48",
    "top_cabinet_49",
    "top_cabinet_50",
    "top_cabinet_51",
]

# Set to true to print Simulator step() statistics
PRINT_STATS = True
# Set to true to use gripper instead of VR hands
USE_GRIPPER = False

# HDR files for PBR rendering
hdr_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_02.hdr")
hdr_texture2 = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_03.hdr")
light_modulation_map_filename = os.path.join(
    igibson.ig_dataset_path, "scenes", "Rs_int", "layout", "floor_lighttype_0.png"
)
background_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "urban_street_01.jpg")


def main():
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

    s = Simulator(mode="vr", rendering_settings=vr_rendering_settings)
    scene = InteractiveIndoorScene("Rs_int")
    scene._set_obj_names_to_load(benchmark_names)
    s.import_scene(scene)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    vr_agent = BehaviorRobot(use_gripper=USE_GRIPPER)
    # Move VR agent to the middle of the kitchen
    s.set_vr_start_pos(start_pos=[0, 2.1, 0], vr_height_offset=-0.02)

    # Mass values to use for each object type - len(masses) objects will be created of each type
    masses = [1, 5, 10]

    # List of objects to load with name: filename, type, scale, base orientation, start position, spacing vector and spacing value
    obj_to_load = {
        "mustard": ("006_mustard_bottle", "ycb", 1, (0.0, 0.0, 0.0, 1.0), (0.0, 1.6, 1.18), (-1, 0, 0), 0.15),
        "marker": ("040_large_marker", "ycb", 1, (0.0, 0.0, 0.0, 1.0), (1.5, 2.6, 0.92), (0, -1, 0), 0.15),
        "can": ("005_tomato_soup_can", "ycb", 1, (0.0, 0.0, 0.0, 1.0), (1.7, 2.6, 0.95), (0, -1, 0), 0.15),
        "drill": ("035_power_drill", "ycb", 1, (0.0, 0.0, 0.0, 1.0), (1.5, 2.2, 1.15), (0, -1, 0), 0.2),
        "small_jenga": (
            "jenga/jenga.urdf",
            "pb",
            1,
            (0.000000, 0.707107, 0.000000, 0.707107),
            (-0.9, 1.6, 1.18),
            (-1, 0, 0),
            0.1,
        ),
        "large_jenga": (
            "jenga/jenga.urdf",
            "pb",
            2,
            (0.000000, 0.707107, 0.000000, 0.707107),
            (-1.3, 1.6, 1.31),
            (-1, 0, 0),
            0.15,
        ),
        "small_duck": (
            "duck_vhacd.urdf",
            "pb",
            1,
            (0.000000, 0.000000, 0.707107, 0.707107),
            (-1.8, 1.95, 1.12),
            (1, 0, 0),
            0.15,
        ),
        "large_duck": (
            "duck_vhacd.urdf",
            "pb",
            2,
            (0.000000, 0.000000, 0.707107, 0.707107),
            (-1.95, 2.2, 1.2),
            (1, 0, 0),
            0.2,
        ),
        "small_sphere": (
            "sphere_small.urdf",
            "pb",
            1,
            (0.000000, 0.000000, 0.707107, 0.707107),
            (-0.5, 1.63, 1.15),
            (-1, 0, 0),
            0.15,
        ),
        "large_sphere": (
            "sphere_small.urdf",
            "pb",
            2,
            (0.000000, 0.000000, 0.707107, 0.707107),
            (-0.5, 1.47, 1.15),
            (-1, 0, 0),
            0.15,
        ),
    }

    for name in obj_to_load:
        fpath, obj_type, scale, orn, pos, space_vec, space_val = obj_to_load[name]
        for i in range(len(masses)):
            if obj_type == "ycb":
                handle = YCBObject(fpath, scale=scale, renderer_params={"use_pbr": False, "use_pbr_mapping": False})
            elif obj_type == "pb":
                handle = ArticulatedObject(
                    fpath, scale=scale, renderer_params={"use_pbr": False, "use_pbr_mapping": False}
                )

            s.import_object(handle)
            # Calculate new position along spacing vector
            new_pos = (
                pos[0] + space_vec[0] * space_val * i,
                pos[1] + space_vec[1] * space_val * i,
                pos[2] + space_vec[2] * space_val * i,
            )
            handle.set_position(new_pos)
            handle.set_orientation(orn)
            body_ids = handle.get_body_ids()
            assert len(body_ids) == 1, "Object is expected to be single-body."
            body_id = body_ids[0]
            p.changeDynamics(body_id, -1, mass=masses[i])

    title = s.add_vr_overlay_text(
        text_data="Welcome to iGibson VR!", font_size=85, font_style="Bold", color=[0, 0, 0.5], pos=[150, 900]
    )
    sample_condition_1 = s.add_vr_overlay_text(
        text_data="for box in cupboard:\n1) find box\n2) pick up box",
        font_size=40,
        font_style="Regular",
        color=[0, 0, 0],
        pos=[800, 600],
    )
    sample_condition_2 = s.add_vr_overlay_text(
        text_data="for all ducks:\n1) pet the duck", font_size=40, font_style="Regular", color=[0, 0, 0], pos=[800, 400]
    )
    # Start off by hiding sample condition 2
    sample_condition_2.set_show_state(False)
    timing_text = s.add_vr_overlay_text(
        text_data="Current time: {}".format(time.time()),
        font_size=60,
        font_style="Italic",
        color=[1, 0, 0],
        pos=[100, 100],
    )

    # Main simulation loop
    while True:
        s.step(print_stats=PRINT_STATS)

        timing_text.set_text("Current time: {}".format(time.time()))

        # Show/hide the sample conditions with button press
        if s.query_vr_event("right_controller", "overlay_toggle"):
            sample_condition_1.set_show_state(not sample_condition_1.get_show_state())
            sample_condition_2.set_show_state(not sample_condition_2.get_show_state())

        vr_agent.apply_action()

    s.disconnect()


if __name__ == "__main__":
    main()
