import argparse
import datetime
import logging
import os
import pprint

import h5py
import numpy as np

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.utils.git_utils import project_git_info
from igibson.utils.log_utils import IGLogReader, IGLogWriter
from igibson.utils.config_utils import parse_config, parse_str_config


def main(random_selection=False, headless=False, short_exec=False):
    """
    Example of how to replay a previously recorded demo of a task
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    # Assuming that if random_selection=True, headless=True, short_exec=True, we are calling it from tests and we
    # do not want to parse args (it would fail because the calling function is pytest "testfile.py")
    if not (random_selection and headless and short_exec):
        args = parse_args()
        replay_demo(
            args.log_path,
            out_log_path=args.replay_log_path,
            disable_save=args.disable_save,
            frame_save_path=args.frame_save_path,
            mode=args.mode,
            profile=args.profile,
            config_file=args.config,
        )
    else:
        replay_demo(
            os.path.join("/", "tmp", "demo.hdf5"),
        )


def verify_determinism(in_log_path, out_log_path):
    is_deterministic = True
    with h5py.File(in_log_path) as original_file, h5py.File(out_log_path) as new_file:
        for obj in original_file["physics_data"]:
            for attribute in original_file["physics_data"][obj]:
                is_close = np.isclose(
                    original_file["physics_data"][obj][attribute], new_file["physics_data"][obj][attribute]
                )
                is_deterministic = is_deterministic and is_close.all()
                if not is_close.all():
                    logging.info(
                        "Mismatch for obj {} with mismatched attribute {} starting at timestep {}".format(
                            obj, attribute, np.where(is_close == False)[0][0]
                        )
                    )
    return bool(is_deterministic)


def parse_args():
    parser = argparse.ArgumentParser(description="Run and collect an ATUS demo")
    parser.add_argument("--log_path", type=str, required=True, help="Path (and filename) of log to replay")
    parser.add_argument(
        "--replay_log_path", type=str, help="Path (and filename) of file to save replay to (for debugging)"
    )
    parser.add_argument(
        "--frame_save_path",
        type=str,
        help="Path to save frames (frame number added automatically, as well as .jpg extension)",
    )
    parser.add_argument(
        "--disable_save",
        action="store_true",
        help="Whether to disable saving log of replayed trajectory, used for validation.",
    )
    parser.add_argument("--profile", action="store_true", help="Whether to print profiling data.")
    parser.add_argument(
        "--mode",
        type=str,
        default="gui_non_interactive",
        choices=["headless", "headless_tensor", "vr", "gui_non_interactive"],
        help="Mode to run simulator in",
    )
    parser.add_argument(
        "--config",
        help="which config file to use [default: use yaml files in examples/configs]",
        default=os.path.join(igibson.example_config_path, "behavior_vr.yaml"),
    )
    return parser.parse_args()


def replay_demo(
    in_log_path,
    out_log_path=None,
    disable_save=False,
    frame_save_path=None,
    verbose=True,
    mode="headless",
    config_file=os.path.join(igibson.example_config_path, "behavior_vr.yaml"),
    start_callbacks=[],
    step_callbacks=[],
    end_callbacks=[],
    profile=False,
    image_size=(1280, 720),
    use_pb_gui=False,
):
    """
    Replay a demo of a task.

    Note that this returns, but does not check for determinism. Use safe_replay_demo to assert for determinism
    when using in scenarios where determinism is important.

    @param in_log_path: the path of the BEHAVIOR demo log to replay.
    @param out_log_path: the path of the new BEHAVIOR demo log to save from the replay.
    @param disable_save: Whether saving the replay as a BEHAVIOR demo log should be disabled.
    @param frame_save_path: the path to save frame images to. None to disable frame image saving.
    @param verbose: Whether to print out git diff in detail
    @param mode: which rendering mode ("headless", "headless_tensor", "gui_non_interactive", "vr"). In gui_non_interactive
        mode, the demo will be replayed with simple robot view.
    @param config_file: environment config file
    @param start_callback: A callback function that will be called immediately before starting to replay steps. Should
        take two arguments: iGibsonEnv and IGLogReader
    @param step_callback: A callback function that will be called immediately following each replayed step. Should
        take two arguments: iGibsonEnv and IGLogReader
    @param end_callback: A callback function that will be called when replay has finished. Should
        take two arguments: iGibsonEnv and IGLogReader
    @param profile: Whether the replay should be profiled, with profiler output to stdout.
    @param image_size: The image size that should be used by the renderer.
    @param use_pb_gui: display the interactive pybullet gui (for debugging)
    @return if disable_save is True, returns None. Otherwise, returns a boolean indicating if replay was deterministic.
    """
    # HDR files for PBR rendering
    hdr_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_02.hdr")
    hdr_texture2 = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_03.hdr")
    light_modulation_map_filename = os.path.join(
        igibson.ig_dataset_path, "scenes", "Rs_int", "layout", "floor_lighttype_0.png"
    )
    background_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "urban_street_01.jpg")

    # VR rendering settings
    rendering_setting = MeshRendererSettings(
        optimized=True,
        fullscreen=False,
        env_texture_filename=hdr_texture,
        env_texture_filename2=hdr_texture2,
        env_texture_filename3=background_texture,
        light_modulation_map_filename=light_modulation_map_filename,
        enable_shadow=True,
        enable_pbr=True,
        msaa=False,
        light_dimming_factor=1.0,
    )

    # Check mode
    assert mode in ["headless", "headless_tensor", "vr", "gui_non_interactive"]

    # Initialize settings to save action replay frames
    vr_settings = VrSettings(config_str=IGLogReader.read_metadata_attr(in_log_path, "/metadata/vr_settings"))
    vr_settings.set_frame_save_path(frame_save_path)

    task = IGLogReader.read_metadata_attr(in_log_path, "/metadata/atus_activity")
    task_id = IGLogReader.read_metadata_attr(in_log_path, "/metadata/activity_definition")
    scene = IGLogReader.read_metadata_attr(in_log_path, "/metadata/scene_id")
    physics_timestep = IGLogReader.read_metadata_attr(in_log_path, "/metadata/physics_timestep")
    render_timestep = IGLogReader.read_metadata_attr(in_log_path, "/metadata/render_timestep")
    filter_objects = IGLogReader.read_metadata_attr(in_log_path, "/metadata/filter_objects")
    instance_id = IGLogReader.read_metadata_attr(in_log_path, "/metadata/instance_id")
    urdf_file = IGLogReader.read_metadata_attr(in_log_path, "/metadata/urdf_file")

    if urdf_file is None:
        urdf_file = "{}_task_{}_{}_0_fixed_furniture".format(scene, task, task_id)

    if instance_id is None:
        instance_id = 0

    logged_git_info = IGLogReader.read_metadata_attr(in_log_path, "/metadata/git_info")
    logged_git_info = parse_str_config(logged_git_info)
    git_info = project_git_info()
    pp = pprint.PrettyPrinter(indent=4)

    for key in logged_git_info.keys():
        if key not in git_info:
            print(
                "Warning: {} not present in current git info. It might be installed through PyPI, "
                "so its version cannot be validated.".format(key)
            )
            continue

        logged_git_info[key].pop("directory", None)
        git_info[key].pop("directory", None)
        if logged_git_info[key] != git_info[key] and verbose:
            print("Warning, difference in git commits for repo: {}. This may impact deterministic replay".format(key))
            print("Logged git info:\n")
            pp.pprint(logged_git_info[key])
            print("Current git info:\n")
            pp.pprint(git_info[key])

    config = parse_config(config_file)
    config["task"] = task
    config["task_id"] = task_id
    config["scene_id"] = scene
    config["instance_id"] = instance_id
    config["urdf_file"] = urdf_file
    config["image_width"] = image_size[0]
    config["image_height"] = image_size[1]
    config["online_sampling"] = False

    env = iGibsonEnv(
        config_file=config,
        mode=mode,
        action_timestep=render_timestep,
        physics_timestep=physics_timestep,
        rendering_settings=rendering_setting,
        vr_settings=vr_settings,
        use_pb_gui=use_pb_gui,
    )
    env.reset()
    robot = env.robots[0]

    log_reader = IGLogReader(in_log_path, log_status=False)
    log_writer = None
    if not disable_save:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if out_log_path is None:
            out_log_path = "{}_{}_{}_{}_{}_replay.hdf5".format(task, task_id, scene, instance_id, timestamp)

        log_writer = IGLogWriter(
            env.simulator,
            log_filepath=out_log_path,
            task=env.task,
            store_vr=False,
            vr_robot=robot,
            profiling_mode=profile,
            filter_objects=filter_objects,
        )
        log_writer.set_up_data_storage()

    try:
        for callback in start_callbacks:
            callback(env, log_reader)

        task_done = False

        while log_reader.get_data_left_to_read():
            env.step(log_reader.get_agent_action("vr_robot"))
            task_done |= env.task.check_success()[0]

            # Set camera each frame
            if mode == "vr":
                log_reader.set_replay_camera(env.simulator)

            for callback in step_callbacks:
                callback(env, log_reader)

            if not disable_save:
                log_writer.process_frame()

        logging.info("Demo was successfully completed: {}".format(task_done))

        demo_statistics = {}
        for callback in end_callbacks:
            callback(env, log_reader)
    finally:
        env.close()
        if not disable_save:
            log_writer.end_log_session()

    is_deterministic = None
    if not disable_save:
        is_deterministic = verify_determinism(in_log_path, out_log_path)
        logging.info("Demo was deterministic: {}".format(is_deterministic))

    demo_statistics = {
        "deterministic": is_deterministic,
        "task": task,
        "task_id": int(task_id),
        "scene": scene,
        "task_done": task_done,
        "total_frame_num": log_reader.total_frame_num,
    }
    return demo_statistics


def safe_replay_demo(*args, **kwargs):
    """Replays a demo, asserting that it was deterministic."""
    demo_statistics = replay_demo(*args, **kwargs)
    assert (
        demo_statistics["deterministic"] == True
    ), "Replay was not deterministic (or was executed with disable_save=True)."


if __name__ == "__main__":
    main()
