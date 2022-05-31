""" VR saving/replay demo.

In this demo, we save some "mock" VR actions that are already saved by default,
but can be saved separately as actions to demonstrate the action-saving system.
During replay, we use a combination of these actions and saved values to get data
that can be used to control the physics simulation without setting every object's
transform each frame.

Usage:
python vr_sr.py --mode=[save/replay]

This demo saves to vr_logs/vr_sr.h5
Run this demo (and also change the filename) if you would like to save your own data."""

import argparse
import os

import numpy as np

import pybullet_data

import igibson
from igibson.objects.usd_object import ArticulatedObject
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.robots.behavior_robot import BehaviorRobot
from igibson.scenes.interactive_traversable_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
from igibson.utils.ig_logging import IGLogReader, IGLogWriter

# Number of frames to save
FRAMES_TO_SAVE = 600
# Set to true to print PyBullet data - can be used to check whether replay was identical to saving
PRINT_PB = True
# Modify this path to save to different files
VR_LOG_PATH = "vr_sr.h5"


def run_action_sr(mode):
    """
    Runs action save/replay. Mode can either be save or replay.
    """
    assert mode in ["save", "replay"]
    is_save = mode == "save"

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
    # VR system settings - loaded from HDF5 file
    if not is_save:
        vr_settings = VrSettings(config_str=IGLogReader.read_metadata_attr(VR_LOG_PATH, "/metadata/vr_settings"))
        vr_settings.turn_on_companion_window()
    else:
        vr_settings = VrSettings(use_vr=True)

    s = Simulator(mode="vr", rendering_settings=vr_rendering_settings, vr_settings=vr_settings)
    scene = InteractiveIndoorScene(
        "Rs_int", load_object_categories=["walls", "floors", "ceilings"], load_room_types=["kitchen"]
    )
    s.import_scene(scene)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Note: uncomment these lines during replay to see the scene from an external perspective
    # camera_pose = np.array([0, 0, 1])
    # view_direction = np.array([0, -1, 0])
    # s.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
    # s.renderer.set_fov(90)

    # Create a BehaviorRobot and it will handle all initialization and importing under-the-hood
    # Data replay uses constraints during both save and replay modes
    # Note: set show_visual_head to True upon replay to see the VR head
    bvr_robot = BehaviorRobot(show_visual_head=False)
    s.import_object(bvr_robot)
    s.register_main_vr_robot(bvr_robot)
    bvr_robot.set_position_orientation([0, 0, 1.5], [0, 0, 0, 1])

    # Objects to interact with
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
    obj.set_position_orientation([1.1, 0.300000, 1.2], [0, 0, 0, 1])

    mock_vr_action_path = "mock_vr_action"

    if mode == "save":
        # Saves every 2 seconds or so (200 / 90fps is approx 2 seconds)
        log_writer = IGLogWriter(
            s, log_filepath=VR_LOG_PATH, frames_before_write=200, store_vr=True, vr_robot=bvr_robot, log_status=False
        )

        # Save a single button press as a mock action that demonstrates action-saving capabilities.
        log_writer.register_action(mock_vr_action_path, (1,))

        # Call set_up_data_storage once all actions have been registered (in this demo we only save states so there are none)
        # Despite having no actions, we need to call this function
        log_writer.set_up_data_storage()
    else:
        # Playback faster than FPS during saving - can set emulate_save_fps to True to emulate saving FPS
        log_reader = IGLogReader(VR_LOG_PATH, log_status=False)

    if is_save:
        # Main simulation loop - run for as long as the user specified
        for i in range(FRAMES_TO_SAVE):
            s.step()

            # Example of storing a simple mock action
            log_writer.save_action(mock_vr_action_path, np.array([1]))

            # Update VR objects
            bvr_robot.update(s.gen_vr_robot_action())

            # Print debugging information
            if PRINT_PB:
                log_writer._print_pybullet_data()

            # Record this frame's data in the VRLogWriter
            log_writer.process_frame()

        # Note: always call this after the simulation is over to close the log file
        # and clean up resources used.
        log_writer.end_log_session()
    else:
        # The VR reader automatically shuts itself down and performs cleanup once the while loop has finished running
        while log_reader.get_data_left_to_read():
            s.step()

            # Set camera to be where VR headset was looking
            # Note: uncomment this (and comment in camera setting lines above) to see the scene from an external, non-VR perspective
            log_reader.set_replay_camera(s)

            # Read our mock action (but don't do anything with it for now)
            mock_action = int(log_reader.read_action(mock_vr_action_path)[0])

            # VrData that could be used for various purposes
            curr_frame_vr_data = log_reader.get_vr_data()

            # Get relevant VR action data and update VR agent
            bvr_robot.update(log_reader.get_agent_action("vr_robot"))
            # vr_agent.links['eye'].show_eye()

            # Print debugging information
            if PRINT_PB:
                log_reader._print_pybullet_data()

    s.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VR state saving and replay demo")
    parser.add_argument("--mode", default="save", help="Mode to run in: either save or replay")
    args = parser.parse_args()
    run_action_sr(mode=args.mode)
