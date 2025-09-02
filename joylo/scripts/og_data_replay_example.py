from omnigibson.envs import DataPlaybackWrapper
from omnigibson.utils.config_utils import TorchEncoder
import torch as th
import os
import omnigibson as og
from omnigibson.macros import gm
import argparse
import sys
import json
from gello.robots.sim_robot.og_teleop_utils import optimize_sim_settings
from gello.utils.qa_utils import *
from gello.utils.b1k_utils import ALL_QA_METRICS, COMMON_QA_METRICS, TASK_QA_METRICS
import inspect

RUN_QA = True

gm.RENDER_VIEWER_CAMERA = False
gm.DEFAULT_VIEWER_WIDTH = 128
gm.DEFAULT_VIEWER_HEIGHT = 128


def extract_arg_names(func):
    return list(inspect.signature(func).parameters.keys())


def replay_hdf5_file(hdf_input_path):
    """
    Replays a single HDF5 file and saves videos to a new folder
    
    Args:
        hdf_input_path: Path to the HDF5 file to replay
    """
    # Create folder with same name as HDF5 file (without extension)
    base_name = os.path.basename(hdf_input_path)
    folder_name = os.path.splitext(base_name)[0]
    folder_path = os.path.join(os.path.dirname(hdf_input_path), folder_name)
    
    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    
    # Define output paths
    hdf_output_path = os.path.join(folder_path, f"{folder_name}_replay.hdf5")
    video_dir = folder_path

    # Metrics path
    metrics_output_path = os.path.join(folder_path, f"qa_metrics.json")

    # Move original HDF5 file to the new folder
    new_hdf_input_path = os.path.join(folder_path, base_name)
    if hdf_input_path != new_hdf_input_path:  # Avoid copying if already in target folder
        os.rename(hdf_input_path, new_hdf_input_path)
        hdf_input_path = new_hdf_input_path
    
    # Define resolution for consistency
    RESOLUTION_DEFAULT = 560
    RESOLUTION_WRIST = 240
    
    # This flag is needed to run data playback wrapper
    gm.ENABLE_TRANSITION_RULES = False
    
    # Define external camera positions and orientations
    external_camera_poses = [
        # Camera 1
        [[-0.4, 0, 2.0], [0.2706, -0.2706, -0.6533,  0.6533]],
        # # Camera 2
        # [[-0.2, 0.6, 2.0], [-0.1930, 0.4163, 0.8062, -0.3734]],
        # # Camera 3
        # [[-0.2, -0.6, 2.0], [0.4164, -0.1929, -0.3737, 0.8060]]
    ]
    
    # Robot sensor configuration
    robot_sensor_config = {
        "VisionSensor": {
            "modalities": ["rgb"],
            "sensor_kwargs": {
                "image_height": RESOLUTION_WRIST,
                "image_width": RESOLUTION_WRIST,
            },
        },
    }
    
    # Generate external sensors config automatically
    external_sensors_config = []
    for i, (position, orientation) in enumerate(external_camera_poses):
        external_sensors_config.append({
            "sensor_type": "VisionSensor",
            "name": f"external_sensor{i}",
            "relative_prim_path": f"/controllable__r1pro__robot_r1/base_link/external_sensor{i}",
            "modalities": ["rgb"],
            "sensor_kwargs": {
                "image_height": RESOLUTION_DEFAULT,
                "image_width": RESOLUTION_DEFAULT,
                "horizontal_aperture": 40.0,
            },
            "position": th.tensor(position, dtype=th.float32),
            "orientation": th.tensor(orientation, dtype=th.float32),
            "pose_frame": "parent",
        })

    # Replace normal head camera with custom config
    idx = len(external_sensors_config)
    external_sensors_config.append({
        "sensor_type": "VisionSensor",
        "name": f"external_sensor{idx}",
        "relative_prim_path": f"/controllable__r1pro__robot_r1/zed_link/external_sensor{idx}",
        "modalities": ["rgb", "seg_instance_id"],
        "sensor_kwargs": {
            "image_height": RESOLUTION_DEFAULT,
            "image_width": RESOLUTION_DEFAULT,
            "horizontal_aperture": 40.0,
        },
        "position": th.tensor([0.06, 0.0, 0.01], dtype=th.float32),
        "orientation": th.tensor([-1.0, 0.0, 0.0, 0.0], dtype=th.float32),
        "pose_frame": "parent",
    })

    # Create the environment
    additional_wrapper_configs = []
    if RUN_QA:
        additional_wrapper_configs.append({
            "type": "MetricsWrapper",
        })
    env = DataPlaybackWrapper.create_from_hdf5(
        input_path=hdf_input_path,
        output_path=hdf_output_path,
        robot_obs_modalities=["rgb"],
        robot_sensor_config=robot_sensor_config,
        external_sensors_config=external_sensors_config,
        exclude_sensor_names=["zed"],
        n_render_iterations=1,
        only_successes=False,
        additional_wrapper_configs=additional_wrapper_configs,
        include_task=True,
        include_task_obs=False,
        include_robot_control=False,
        include_contacts=True,
    )

    # Optimize rendering for faster speeds
    og.sim.add_callback_on_play("optimize_rendering", optimize_sim_settings)

    if RUN_QA:
        # Add QA metrics
        metric_kwargs = dict(
            step_dt=1/30,
            vel_threshold=0.001,
            color_arms=False,           # For ghost hand
            default_color=(0.8235, 0.8235, 1.0000),
            head_camera=env.external_sensors[f"external_sensor{len(env.external_sensors) - 1}"],
            head_camera_link_name="torso_link4",
            navigation_window=3.0,
            translation_threshold=0.1,
            rotation_threshold=0.05,
            camera_tilt_threshold=0.4,
            gripper_link_paths={
                "left":
                    set([
                        '/World/scene_0/controllable__r1pro__robot_r1/left_realsense_link/visuals',
                        '/World/scene_0/controllable__r1pro__robot_r1/left_gripper_link/visuals',
                        '/World/scene_0/controllable__r1pro__robot_r1/left_gripper_finger_link1/visuals',
                        '/World/scene_0/controllable__r1pro__robot_r1/left_gripper_finger_link2/visuals'
                    ]),
                "right":
                    set([
                        '/World/scene_0/controllable__r1pro__robot_r1/right_realsense_link/visuals',
                        '/World/scene_0/controllable__r1pro__robot_r1/right_gripper_link/visuals',
                        '/World/scene_0/controllable__r1pro__robot_r1/right_gripper_finger_link1/visuals',
                        '/World/scene_0/controllable__r1pro__robot_r1/right_gripper_finger_link2/visuals'
                    ])
            },
        )
        active_metrics_info = {metric_name: ALL_QA_METRICS[metric_name] for metric_name in COMMON_QA_METRICS}
        for metric_name, metric_info in active_metrics_info.items():
            create_fcn = metric_info["cls"] if metric_info["init"] is None else metric_info["init"]
            init_kwargs = {arg: metric_kwargs[arg] for arg in extract_arg_names(create_fcn)}
            metric = create_fcn(**init_kwargs)
            env.add_metric(name=metric_name, metric=metric)
        env.reset()

    # Create a list to store video writers and RGB keys
    video_writers = []
    video_rgb_keys = []
    
    # Create video writer for robot cameras
    robot_camera_names = ['robot_r1::robot_r1:left_realsense_link:Camera:0::rgb', 
                        'robot_r1::robot_r1:right_realsense_link:Camera:0::rgb']
    for robot_camera_name in robot_camera_names:
        video_writers.append(env.create_video_writer(
            fpath=f"{video_dir}/{robot_camera_name}.mp4",
            resolution=(RESOLUTION_WRIST, RESOLUTION_WRIST),    
        ))
        video_rgb_keys.append(robot_camera_name)
    # Create video writers for external cameras
    for i in range(len(external_sensors_config)):
        camera_name = f"external_sensor{i}"
        video_writers.append(env.create_pyav_writer(
            fpath=f"{video_dir}/{camera_name}.mp4",
            resolution=(RESOLUTION_DEFAULT, RESOLUTION_DEFAULT),    
        ))
        video_rgb_keys.append(f"external::{camera_name}::rgb")
    # Playback the dataset with all video writers
    # We avoid calling playback_dataset and call playback_episode individually in order to manually
    # aggregate per-episode metrics
    metrics = dict()
    for episode_id in range(env.input_hdf5["data"].attrs["n_episodes"]):
        env.playback_episode(
            episode_id=episode_id,
            record_data=False,
            video_writers=video_writers,
            video_keys=video_rgb_keys,
        )
        if RUN_QA:
            episode_metrics = env.aggregate_metrics(flatten=True)
            for k, v in episode_metrics.items():
                print(f"Metric [{k}]: {v}")
            metrics[f"episode_{episode_id}"] = episode_metrics
    
    # Close all video writers
    for container, stream in video_writers:
        # Flush any remaining packets
        for packet in stream.encode():
            container.mux(packet)
        # Close the container
        container.close()

    env.save_data()

    # Save metrics
    with open(metrics_output_path, "w+") as f:
        json.dump(metrics, f, cls=TorchEncoder, indent=4)

    # Always clear the environment to free resources
    og.clear()
        
    print(f"Successfully processed {hdf_input_path}")


def main():
    parser = argparse.ArgumentParser(description="Replay HDF5 files and save videos")
    parser.add_argument("--dir", help="Directory containing HDF5 files to process")
    parser.add_argument("--files", nargs="*", help="Individual HDF5 file(s) to process")
    
    args = parser.parse_args()
    
    if args.dir and os.path.isdir(args.dir):
        # Process all HDF5 files in the directory (non-recursively)
        hdf_files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) 
                    if f.lower().endswith('.hdf5') and os.path.isfile(os.path.join(args.dir, f))]
        
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
            
        replay_hdf5_file(hdf_file)

    og.shutdown()


if __name__ == "__main__":
    main()
