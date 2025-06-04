import argparse
import omnigibson as og
import os
import sys
import torch as th
from omnigibson.envs import DataPlaybackWrapper
from omnigibson.macros import gm


class CustomDataPlaybackWrapper(DataPlaybackWrapper):
    def _process_obs(self, obs, info):
        robot = self.env.robots[0]
        base_link_pose = th.concatenate(robot.get_position_orientation())
        left_cam_pose = th.concatenate(
            robot.sensors["robot_r1:left_realsense_link:Camera:0"].get_position_orientation()
        )
        right_cam_pose = th.concatenate(
            robot.sensors["robot_r1:right_realsense_link:Camera:0"].get_position_orientation()
        )
        external_cam_pose = th.concatenate(self.env.external_sensors["external_sensor0"].get_position_orientation())
        # store the poses to obs
        obs["robot_r1::robot_base_link_pose"] = base_link_pose
        obs["robot_r1::left_cam_pose"] = left_cam_pose
        obs["robot_r1::right_cam_pose"] = right_cam_pose
        obs["robot_r1::external_cam_pose"] = external_cam_pose
        return obs


def replay_hdf5_file(hdf_input_path, write_video=False):
    """
    Replays a single HDF5 file and saves videos to a new folder

    Args:
        hdf_input_path: Path to the HDF5 file to replay
    """
    # get processed folder path
    processed_dir = os.path.join(os.path.dirname(os.path.dirname(hdf_input_path)), "rgbd")
    os.makedirs(processed_dir, exist_ok=True)
    base_name = os.path.basename(hdf_input_path)
    folder_name = os.path.splitext(base_name)[0]

    # Define output paths
    hdf_output_path = os.path.join(processed_dir, f"{folder_name}.hdf5")
    video_dir = processed_dir

    # Define resolution for consistency
    RESOLUTION = 240

    # This flag is needed to run data playback wrapper
    gm.ENABLE_TRANSITION_RULES = False

    # Robot sensor configuration
    robot_sensor_config = {
        "VisionSensor": {
            "modalities": ["rgb", "depth_linear"],
            "sensor_kwargs": {
                "image_height": RESOLUTION,
                "image_width": RESOLUTION,
            },
        },
    }

    # Replace normal head camera with custom config
    external_sensors_config = []
    external_sensors_config.append(
        {
            "sensor_type": "VisionSensor",
            "name": "external_sensor0",
            "relative_prim_path": "/controllable__r1pro__robot_r1/zed_link/external_sensor0",
            "modalities": ["rgb", "depth_linear"],
            "sensor_kwargs": {
                "image_height": RESOLUTION,
                "image_width": RESOLUTION,
                "horizontal_aperture": 40.0,
            },
            "position": th.tensor([0.06, 0.0, 0.01], dtype=th.float32),
            "orientation": th.tensor([-1.0, 0.0, 0.0, 0.0], dtype=th.float32),
            "pose_frame": "parent",
        }
    )

    # Create the environment
    additional_wrapper_configs = []

    env = CustomDataPlaybackWrapper.create_from_hdf5(
        input_path=hdf_input_path,
        output_path=hdf_output_path,
        robot_obs_modalities=["rgb", "depth_linear", "proprio"],
        robot_sensor_config=robot_sensor_config,
        external_sensors_config=external_sensors_config,
        n_render_iterations=1,
        only_successes=False,
        additional_wrapper_configs=additional_wrapper_configs,
    )

    # Create a list to store video writers and RGB keys
    video_writers = []
    video_keys = []

    if write_video:
        # Create video writer for robot cameras
        robot_camera_names = [
            "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
            "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
            "robot_r1::robot_r1:zed_link:Camera:0::rgb",
        ]

        for camera_name in robot_camera_names:
            video_writers.append(env.create_video_writer(fpath=f"{video_dir}/{camera_name}.mp4"))
            video_keys.append(camera_name)

    breakpoint()

    # Playback the dataset with all video writers
    # We avoid calling playback_dataset and call playback_episode individually in order to manually
    # aggregate per-episode metrics
    for episode_id in range(env.input_hdf5["data"].attrs["n_episodes"]):
        print(f" >>> Replaying episode {episode_id}")

        env.playback_episode(
            episode_id=episode_id,
            record_data=True,
            video_writers=video_writers,
            video_rgb_keys=video_keys,
        )

    # Close all video writers
    for writer in video_writers:
        writer.close()

    env.save_data()

    # Always clear the environment to free resources
    og.clear()

    print(f"Successfully processed {hdf_input_path}")


def main():
    parser = argparse.ArgumentParser(description="Replay HDF5 files and save videos")
    parser.add_argument("--dir", help="Directory containing HDF5 files to process")
    parser.add_argument("--files", nargs="*", help="Individual HDF5 file(s) to process")
    parser.add_argument(
        "--write_video", action="store_true", help="Include this flag to write RGB and depth video files"
    )

    args = parser.parse_args()

    if args.dir and os.path.isdir(args.dir):
        # the directory must ends with raw
        assert args.dir.endswith("raw"), "Directory must end with 'raw' to process HDF5 files"
        # Process all HDF5 files in the directory (non-recursively)
        hdf_files = [
            os.path.join(args.dir, f)
            for f in os.listdir(args.dir)
            if f.lower().endswith(".hdf5") and os.path.isfile(os.path.join(args.dir, f))
        ]

        if not hdf_files:
            print(f"No HDF5 files found in directory: {args.dir}")
        else:
            print(f"Found {len(hdf_files)} HDF5 files to process")
    elif args.files:
        # Process individual files specified
        hdf_files = args.files
    else:
        parser.print_help()
        print("\nError: Either --dir or --files must be specified", file=sys.stderr)
        return

    # Process each file
    for hdf_file in hdf_files:
        if not os.path.exists(hdf_file):
            print(f"Error: File {hdf_file} does not exist", file=sys.stderr)
            continue

        replay_hdf5_file(hdf_file, write_video=args.write_video)

    print("All done!")


if __name__ == "__main__":
    main()
