""" This is a simple object picking and placing task
that can be used to benchmark the dexterity of the VR hand.
"""
import os

import igibson
from igibson import object_states
from igibson.objects.ycb_object import YCBObject
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.robots.behavior_robot import BehaviorRobot
from igibson.scenes.interactive_traversable_scene import InteractiveIndoorScene
from igibson.simulator import Simulator

# Set to true to use viewer manipulation instead of VR
# Set to false by default so this benchmark task can be performed in VR
VIEWER_MANIP = False
# Set to true to print out render, physics and overall frame FPS
PRINT_FPS = True
# Set to true to use gripper instead of VR hands
USE_GRIPPER = False

# HDR files for PBR rendering
hdr_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_02.hdr")
hdr_texture2 = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_03.hdr")
light_modulation_map_filename = os.path.join(
    igibson.ig_dataset_path, "scenes", "Rs_int", "layout", "floor_lighttype_0.png"
)
background_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "urban_street_01.jpg")

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

if VIEWER_MANIP:
    s = Simulator(
        mode="gui_interactive",
        image_width=512,
        image_height=512,
        rendering_settings=vr_rendering_settings,
    )
else:
    vr_settings = VrSettings(use_vr=True)
    s = Simulator(mode="vr", rendering_settings=vr_rendering_settings, vr_settings=vr_settings)

scene = InteractiveIndoorScene("Rs_int", texture_randomization=False, object_randomization=False)
s.import_scene(scene)

if not VIEWER_MANIP:
    vr_agent = BehaviorRobot(use_gripper=USE_GRIPPER, normal_color=False)

block = YCBObject(name="036_wood_block", abilities={"soakable": {}, "cleaning_tool": {}})
s.import_object(block)
block.set_position([1, 1, 1.8])

# Set everything that can go dirty.
dirtyable_objects = set(
    scene.get_objects_with_state(object_states.Dusty) + scene.get_objects_with_state(object_states.Stained)
)
for obj in dirtyable_objects:
    if object_states.Dusty in obj.states:
        obj.states[object_states.Dusty].set_value(True)

    if object_states.Stained in obj.states:
        obj.states[object_states.Stained].set_value(True)

# Main simulation loop
while True:
    s.step(print_time=PRINT_FPS)

    if not VIEWER_MANIP:
        vr_agent.apply_action()

s.disconnect()
