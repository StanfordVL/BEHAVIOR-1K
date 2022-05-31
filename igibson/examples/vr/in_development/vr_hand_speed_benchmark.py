""" This demo can be used to benchmark how speedily the VR hand 
can be used. The aim is to put all the objects into the box on the left
side of the table.

You can use the left and right controllers to start/stop/reset the timer,
as well as show/hide its display. The "overlay toggle" action and its
corresponding button index mapping can be found in the vr_config.yaml file in the igibson folder.
"""
import os


import pybullet_data

import igibson
from igibson.objects.usd_object import ArticulatedObject
from igibson.objects.ycb_object import YCBObject
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.robots.behavior_robot import BehaviorRobot
from igibson.scenes.interactive_traversable_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
from igibson.utils.vr_utils import VrTimer

# Set to true to use viewer manipulation instead of VR
# Set to false by default so this benchmark task can be performed in VR
VIEWER_MANIP = False
# Set to true to print out render, physics and overall frame FPS
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
    vr_settings = VrSettings()

    if VIEWER_MANIP:
        s = Simulator(
            mode="gui_interactive",
            image_width=512,
            image_height=512,
            rendering_settings=vr_rendering_settings,
        )
        vr_settings.turn_off_vr_mode()
    s = Simulator(mode="vr", rendering_settings=vr_rendering_settings, vr_settings=vr_settings)

    scene = InteractiveIndoorScene("Rs_int")
    scene._set_first_n_objects(2)
    s.import_scene(scene)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    if not VIEWER_MANIP:
        vr_agent = BehaviorRobot(use_gripper=USE_GRIPPER)

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

    for i in range(3):
        obj = YCBObject("003_cracker_box")
        s.import_object(obj)
        obj.set_position_orientation([1.100000 + 0.12 * i, -0.300000, 0.750000], [0, 0, 0, 1])

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

    # Time how long demo takes
    time_text = s.add_vr_overlay_text(
        text_data="Current time: NOT STARTED", font_size=100, font_style="Bold", color=[0, 0, 0], pos=[100, 100]
    )
    timer = VrTimer()

    # Main simulation loop
    while True:
        s.step(print_stats=PRINT_STATS)

        if not VIEWER_MANIP:
            # Events that manage timer functionality
            r_toggle = s.query_vr_event("right_controller", "overlay_toggle")
            l_toggle = s.query_vr_event("left_controller", "overlay_toggle")
            # Overlay toggle action on right controller is used to start/stop timer
            if r_toggle and not l_toggle:
                if timer.is_timer_running():
                    timer.stop_timer()
                else:
                    timer.start_timer()
            # Overlay toggle action on left controller is used to show/hide timer
            elif l_toggle and not r_toggle:
                time_text.set_show_state(not time_text.get_show_state())
            # Reset timer if both toggle buttons are pressed at once
            elif r_toggle and l_toggle:
                timer.refresh_timer()

            # Update timer value
            time_text.set_text("Current time: {}".format(round(timer.get_timer_val(), 1)))

            # Update VR agent
            vr_agent.apply_action()

    s.disconnect()


if __name__ == "__main__":
    main()
