""" This VR hand dexterity benchmark allows the user to interact with many types of objects
and interactive objects, and provides a good way to qualitatively measure the dexterity of a VR hand.

You can use the left and right controllers to start/stop/reset the timer,
as well as show/hide its display. The "overlay toggle" action and its
corresponding button index mapping can be found in the vr_config.yaml file in the igibson folder.
"""
import os

import numpy as np

import pybullet_data

import igibson
from igibson.objects.usd_object import ArticulatedObject
from igibson.objects.ycb_object import YCBObject
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.robots.behavior_robot import BehaviorRobot
from igibson.scenes.interactive_traversable_scene import InteractiveIndoorScene
from igibson.simulator import Simulator

# Objects in the benchmark - corresponds to Rs kitchen environment, for range of items and
# transferability to the real world
# Note: the scene will automatically load in walls/ceilings/floors in addition to these objects
benchmark_names = [
    "bottom_cabinet",
    "countertop",
    "dishwasher",
    "door",
    "fridge",
    "microwave",
    "oven",
    "sink",
    "top_cabinet",
    "trash_can",
]

# Set to true to print Simulator step() statistics
PRINT_STATS = False
# Set to true to use gripper instead of VR hands
USE_GRIPPER = False

# HDR files for PBR rendering
hdr_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_02.hdr")
hdr_texture2 = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_03.hdr")
light_modulation_map_filename = os.path.join(
    igibson.ig_dataset_path, "scenes", "Rs_int", "layout", "floor_lighttype_0.png"
)
background_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "urban_street_01.jpg")

SYMBOL_LIST = [l for l in ".,:;!?()+-=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"]


def gen_rand_string():
    """
    Generates random string of random length to test wrapping and scrolling.
    """
    letter_num = np.random.randint(500)
    text = "".join(np.random.choice(SYMBOL_LIST) for i in range(letter_num))
    return text


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

    scene = InteractiveIndoorScene(
        "Rs_int", load_object_categories=benchmark_names, load_room_types=["kitchen", "lobby"]
    )
    # scene.load_object_categories(benchmark_names)

    s.import_scene(scene)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    vr_agent = BehaviorRobot(use_gripper=USE_GRIPPER, normal_color=True)
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

    # Text position/size is described in percentage of axes in screen space
    wrap_scroll_text = s.add_vr_overlay_text(
        text_data=gen_rand_string(), font_size=70, font_style="Bold", color=[0, 0, 0]
    )

    # Main simulation loop
    while True:
        s.step(print_stats=PRINT_STATS)

        r_toggle = s.query_vr_event("right_controller", "overlay_toggle")
        l_toggle = s.query_vr_event("left_controller", "overlay_toggle")
        # Overlay toggle action on right controller is used to start/stop timer
        if r_toggle:
            wrap_scroll_text.set_text(gen_rand_string())
        if l_toggle:
            s.set_hud_show_state(not s.get_hud_show_state())

        scroll_dir = s.get_scroll_input()
        if scroll_dir > -1:
            wrap_scroll_text.scroll_text(up=scroll_dir)

        # Update VR agent
        vr_agent.apply_action()

    s.disconnect()


if __name__ == "__main__":
    main()
