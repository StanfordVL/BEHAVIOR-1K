""" VR embodiment demo with Fetch robot. This demo allows you to test out Fetch VR's grasping functionality."""

import os

import numpy as np

import pybullet_data

import igibson
from igibson.objects.articulated_object import ArticulatedObject
from igibson.objects.ycb_object import YCBObject
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.scenes.interactive_traversable_scene import InteractiveIndoorScene
from igibson.simulator import Simulator

# IMPORTANT: Change this value if you have a more powerful machine
VR_FPS = 20
# Set to true to print out render, physics and overall frame FPS
PRINT_FPS = True
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
scene._set_first_n_objects(2)
s.import_scene(scene)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

if not VR_MODE:
    camera_pose = np.array([0, -3, 1.2])
    view_direction = np.array([0, 1, 0])
    s.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
    s.renderer.set_fov(90)

# Import FetchVR robot - the class handles importing and setup itself
fvr = FetchVR(s, [0.45, 0, 0], update_freq=1, use_ns_ik=True, use_gaze_marker=True)

objects = [
    ("jenga/jenga.urdf", (1.500000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107)),
    ("jenga/jenga.urdf", (1.400000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107)),
    ("jenga/jenga.urdf", (1.300000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107)),
    ("jenga/jenga.urdf", (1.200000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107)),
    ("jenga/jenga.urdf", (1.100000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107)),
    ("jenga/jenga.urdf", (1.000000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107)),
    ("table/table.urdf", (1.300000, -0.200000, 0.000000), (0.000000, 0.000000, 0.707107, 0.707107)),
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

# Main simulation loop
while True:
    s.step()
    if VR_MODE:
        # FetchVR class handles all update logic
        fvr.update()

s.disconnect()
