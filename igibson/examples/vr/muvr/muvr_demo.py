""" Multi-user VR demo. Always start server running before the client.

Usage: python muvr_demo.py --mode=[server or client] --host=[localhost or ip address] --port=[valid port number]
"""

import argparse
import os
import time

import numpy as np


import igibson
from igibson import assets_path
from igibson.objects.ycb_object import YCBObject
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.robots.behavior_robot import BehaviorRobot
from igibson.scenes.interactive_traversable_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
from igibson.utils.muvr_utils import IGVRClient, IGVRServer

sample_urdf_folder = os.path.join(assets_path, "models", "sample_urdfs")

# Only load in first few objects in Rs to decrease load times
LOAD_PARTIAL = True
# Whether to print iGibson + networking FPS each frame
PRINT_FPS = True
# Whether to wait for client before simulating
WAIT_FOR_CLIENT = False

# Note: This is where the VR configuration for the MUVR experience can be changed.
RUN_SETTINGS = {"client": VrSettings(use_vr=False), "server": VrSettings(use_vr=True)}


def run_muvr(mode="server", host="localhost", port="8885"):
    """
    Sets up the iGibson environment that will be used by both server and client
    """
    print("INFO: Running MUVR {} at {}:{}".format(mode, host, port))
    # This function only runs if mode is one of server or client, so setting this bool is safe
    is_server = mode == "server"

    vr_settings = RUN_SETTINGS[mode]

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
    s = Simulator(mode="vr", rendering_settings=vr_rendering_settings, vr_settings=vr_settings)
    scene = InteractiveIndoorScene("Rs_int")
    if LOAD_PARTIAL:
        scene._set_first_n_objects(10)
    s.import_scene(scene)

    # Default camera for non-VR MUVR users
    if not vr_settings.use_vr:
        camera_pose = np.array([0, -3, 1.2])
        view_direction = np.array([0, 1, 0])
        s.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
        s.renderer.set_fov(90)

    # Spawn two agents - one for client and one for the server
    client_agent = BehaviorRobot(agent_num=1)
    server_agent = BehaviorRobot(agent_num=2)

    # Objects to interact with
    mass_list = [5, 10, 20, 30]
    mustard_start = [-1, 1.55, 1.2]
    for i in range(len(mass_list)):
        mustard = YCBObject("006_mustard_bottle", renderer_params={"use_pbr": False, "use_pbr_mapping": False})
        s.import_object(mustard)
        mustard.set_position([mustard_start[0] + i * 0.2, mustard_start[1], mustard_start[2]])
        body_ids = mustard.get_body_ids()
        assert len(body_ids) == 1, "Object is expected to be single-body."
        body_id = body_ids[0]
        p.changeDynamics(body_id, -1, mass=mass_list[i])

    # Start the two agents at different points so they don't collide upon entering the scene
    if vr_settings.use_vr:
        s.set_vr_start_pos([0.5, 0 if is_server else -1.5, 0], vr_height_offset=-0.1)

    # Setup client/server
    if is_server:
        vr_server = IGVRServer(localaddr=(host, port))
        vr_server.register_data(s, client_agent)
    else:
        vr_client = IGVRClient(host, port)
        vr_client.register_data(s, client_agent)
        # Disconnect pybullet since the client only renders
        s.disconnect(release_renderer=False)

    # Main networking loop
    while True:
        frame_start = time.time()
        if is_server:
            # Update iGibson with latest vr data from client
            vr_server.ingest_vr_data()

            if not WAIT_FOR_CLIENT or vr_server.client_connected():
                # Server is the one that steps the physics simulation, not the client
                s.step()
                if s.vr_settings.use_vr:
                    server_agent.apply_action()
                if vr_server.latest_vr_data:
                    client_agent.apply_action(vr_server.latest_vr_data)

            # Generate and send latest rendering data to client
            vr_server.gen_frame_data()
            vr_server.send_frame_data()
            vr_server.Refresh()
        else:
            # Update client renderer with frame data received from server, and then render it
            vr_client.ingest_frame_data()
            vr_client.client_step()

            # Generate and send client's VR data so it can be used to process client physics on the server
            # This does not generate or send data when the client is in non-VR mode - all error checking is
            # handled by IGVRClient internally
            vr_client.gen_vr_data()
            vr_client.send_vr_data()
            vr_client.Refresh()

        frame_dur = time.time() - frame_start
        if PRINT_FPS:
            print("Frame duration: {:.3f} ms".format(frame_dur / 0.001))

    # Disconnect at end of server session
    if is_server:
        s.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-user VR demo that can be run in server and client mode.")
    parser.add_argument("--mode", default="server", help="Mode to run in: either server or client")
    parser.add_argument("--host", default="localhost", help="Host to connect to - eg. localhost or an IP address")
    parser.add_argument("--port", default="8885", help="Port to connect to - eg. 8887")
    args = parser.parse_args()
    try:
        port = int(args.port)
    except ValueError as ve:
        print("ERROR: Non-integer port-number supplied!")
    if args.mode in ["server", "client"]:
        run_muvr(mode=args.mode, host=args.host, port=port)
    else:
        print("ERROR: mode {} is not supported!".format(args.mode))
