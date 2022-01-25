""" VR embodiment demo with Fetch robot. This demos allows you to explore the Rs environment with Fetch VR"""

import os

import numpy as np


import igibson
from igibson.objects.ycb_object import YCBObject
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.scenes.interactive_traversable_scene import InteractiveIndoorScene
from igibson.simulator import Simulator

# IMPORTANT: Change this value if you have a more powerful machine
VR_FPS = 20
# Set to false to load entire Rs_int scene
LOAD_PARTIAL = True
# Set to true to print out render, physics and overall frame FPS
PRINT_FPS = False
# Set to false to just use FetchVR in non-VR mode
VR_MODE = True

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
if VR_MODE:
    s = Simulator(mode="vr", rendering_settings=vr_rendering_settings, vr_settings=VrSettings(vr_fps=VR_FPS))
else:
    s = Simulator(
        mode="gui_interactive",
        image_width=960,
        image_height=720,
        device_idx=0,
        rendering_settings=vr_rendering_settings,
    )
    s.viewer.min_cam_z = 1.0

scene = InteractiveIndoorScene("Rs_int")
# Turn this on when debugging to speed up loading
if LOAD_PARTIAL:
    scene._set_first_n_objects(10)
s.import_scene(scene)

if not VR_MODE:
    camera_pose = np.array([0, -3, 1.2])
    view_direction = np.array([0, 1, 0])
    s.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
    s.renderer.set_fov(90)

# Import FetchVR robot - the class handles importing and setup itself
fvr = FetchVR(s, [0.5, -1.5, 0], update_freq=1, use_ns_ik=True, use_gaze_marker=True)

# Objects to interact with
mass_list = [5, 10, 100, 500]
mustard_start = [-1, 1.55, 1.2]
for i in range(len(mass_list)):
    mustard = YCBObject("006_mustard_bottle", renderer_params={"use_pbr": False, "use_pbr_mapping": False})
    s.import_object(mustard)
    mustard.set_position([mustard_start[0] + i * 0.2, mustard_start[1], mustard_start[2]])
    body_ids = mustard.get_body_ids()
    assert len(body_ids) == 1, "Object is expected to be single-body."
    body_id = body_ids[0]
    p.changeDynamics(body_id, -1, mass=mass_list[i])

# Main simulation loop
while True:
    s.step()
    if VR_MODE:
        # FetchVR class handles all update logic
        fvr.update()

s.disconnect()
