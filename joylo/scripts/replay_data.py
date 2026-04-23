import argparse
import csv
import inspect
import json
import numpy as np
import omnigibson as og
import os
import torch as th
import yaml
from omnigibson.envs import DataPlaybackWrapper
from omnigibson.learning.utils.obs_utils import create_video_writer, write_video
from omnigibson.macros import gm
from omnigibson.utils.config_utils import TorchEncoder

from gello.utils.qa_utils import (
    ALL_QA_METRICS,
    ACTIVE_QA_METRICS,
    aggregate_episode_validation,
)


gm.RENDER_VIEWER_CAMERA = False
gm.DEFAULT_VIEWER_WIDTH = 128
gm.DEFAULT_VIEWER_HEIGHT = 128

OBS_CAMERA_RESOLUTION = 480


class VideoPlaybackWrapper(DataPlaybackWrapper):
    def _create_video_writers(self, video_keys):
        """
        Create a single video writer that will concatenate all video keys into one video.
        """
        fpath = os.path.join(self.video_output_dir, f"{video_keys['aggregated']}.mp4")
        container, stream = create_video_writer(
            fpath=fpath, 
            resolution=(OBS_CAMERA_RESOLUTION * 3, OBS_CAMERA_RESOLUTION * 3), 
            rate=self.fps, 
            stream_options={"crf": "30"},
        )
        self.video_writers.append((container, stream, "video"))

    def _write_video_frames(self):
        """Write current observation to all video writers."""
        container, stream, _ = self.video_writers[0]
        left_wrist_frame = self._extract_frame_from_obs(
            self.current_obs, "robot::robot:left_realsense_link:Camera:0::rgb"
        )
        right_wrist_frame = self._extract_frame_from_obs(
            self.current_obs, "robot::robot:right_realsense_link:Camera:0::rgb"
        )
        head_frame = self._extract_frame_from_obs(
            self.current_obs, "robot::robot:zed_link:Camera:0::rgb"
        )
        external_frame = self._extract_frame_from_obs(
            self.current_obs, "external::external_sensor0::rgb"
        )

        top_row = np.concatenate(
            [left_wrist_frame, head_frame, right_wrist_frame], axis=1
        )
        obs = np.concatenate([top_row, external_frame], axis=0)
        obs = obs[np.newaxis, ...]

        write_video(obs, (container, stream), mode="rgb")

    def create_dataset(self, output_path, env, overwrite=True):
        # default is no-op
        pass

    def close_dataset(self):
        # default is no-op
        pass


def extract_arg_names(func):
    return list(inspect.signature(func).parameters.keys())


def replay_hdf5_to_video(
    input_path: str,
    task_name: str,
    flush_every_n_steps: int,
    run_qa: bool = False,
) -> int:
    """
    Replays a single HDF5 file and generates videos.

    Args:
        input_path: Path to the HDF5 file
        task_name: Name of the task (also used for QA validation if run_qa is True)
        flush_every_n_steps: Number of steps to flush the data after
        run_qa: Whether to run QA metrics

    Returns:
        episode_id: ID of the episode
    """
    # get the hdf5 file name without extension
    input_filename = os.path.splitext(os.path.basename(input_path))[0]

    gm.ENABLE_TRANSITION_RULES = False

    robot_sensor_config = {
        "VisionSensor": {
            "sensor_kwargs": {
                "image_height": OBS_CAMERA_RESOLUTION,
                "image_width": OBS_CAMERA_RESOLUTION,
            },
        },
        "zed_link:Camera:0": {
            "sensor_kwargs": {
                "horizontal_aperture": 40.0,
                "image_height": OBS_CAMERA_RESOLUTION,
                "image_width": OBS_CAMERA_RESOLUTION,
            },
            "modalities": ["rgb"],
        },
    }
    available_tasks = {}
    # 2026 task instances has precedence over 2025. 
    with open(
        f"{gm.DATA_PATH}/2025-challenge-task-instances/metadata/available_tasks.yaml",
        "r",
    ) as f:
        available_tasks.update(yaml.safe_load(f))
    with open(
        f"{gm.DATA_PATH}/2026-challenge-task-instances/metadata/available_tasks.yaml",
        "r",
    ) as f:
        available_tasks.update(yaml.safe_load(f))
    scene_model = available_tasks[task_name][0]["scene_model"]
    full_scene_file = None
    for year in ("2026", "2025"):
        task_scene_file_folder = os.path.join(
            gm.DATA_PATH, f"{year}-challenge-task-instances", "scenes", scene_model, "json"
        )
        if not os.path.isdir(task_scene_file_folder):
            continue
        for file in os.listdir(task_scene_file_folder):
            if task_name in file and file.endswith(".json") and "partial_rooms" not in file:
                full_scene_file = os.path.join(task_scene_file_folder, file)
                break
        if full_scene_file is not None:
            break
    if full_scene_file is None:
        raise FileNotFoundError(
            f"No full scene file found for task '{task_name}' in either 2026 or 2025 scene directories"
        )

    load_room_instances = None
    try:
        with open(
            f"{gm.DATA_PATH}/2026-challenge-task-instances/metadata/B100_task_misc.csv",
            newline="",
            encoding="utf-8",
        ) as f:
            task_misc_csv = csv.reader(f, delimiter=",", quotechar='"')
            for row in task_misc_csv:
                if task_name in row[1]:
                    load_room_instances = row[2].strip().split("\n")
                    break
    except FileNotFoundError as e:
        raise e
    assert load_room_instances is not None, "load room instance not found!"

    additional_wrapper_configs = []
    if run_qa:
        additional_wrapper_configs.append({"type": "MetricsWrapper"})

    external_sensors_config = [
        {
            "sensor_type": "VisionSensor",
            "name": "external_sensor0",
            "relative_prim_path": f"/controllable__r1pro__robot/base_link/external_sensor0",
            "modalities": ["rgb"],
            "sensor_kwargs": {
                "image_height": OBS_CAMERA_RESOLUTION * 2,
                "image_width": OBS_CAMERA_RESOLUTION * 3,
                "horizontal_aperture": 40.0,
            },
            "position": th.tensor([-0.4, 0, 2.0], dtype=th.float32),
            "orientation": th.tensor(
                [0.2706, -0.2706, -0.6533, 0.6533], dtype=th.float32
            ),
            "pose_frame": "parent",
        }
    ]

    video_keys = {"aggregated": f"{input_filename}_video"}

    kwargs = dict(
        input_path=input_path,
        full_scene_file=full_scene_file,
        load_room_instances=load_room_instances,
        additional_wrapper_configs=additional_wrapper_configs,
        robot_sensor_config=robot_sensor_config,
        n_render_iterations=1,
        flush_every_n_steps=flush_every_n_steps,
        flush_every_n_traj=1,
        include_robot_control=False,
        output_path=input_path, # store outputs in the same folder as the input
        robot_obs_modalities=["rgb"],
        external_sensors_config=external_sensors_config,
        include_task=True,
        include_task_obs=False,
        include_contacts=True,
    )
    env = VideoPlaybackWrapper.create_from_hdf5(**kwargs)
    # add seg_instance_id to robot head camera
    env.robots[0].sensors["robot:zed_link:Camera:0"].add_modality("seg_instance_id")
    env.load_observation_space()
    # Set robot base mass to 250kg to match data collection for r1/r1pro
    if env.robots[0].model in ("r1", "r1pro"):
        with og.sim.stopped():
            env.robots[0].base_footprint_link.mass = 250.0

    if run_qa:
        metric_kwargs = dict(
            step_dt=1 / 30,
            vel_threshold=0.001,
            color_arms=False,
            default_color=(0.8235, 0.8235, 1.0000),
            head_camera=env.robots[0].sensors["robot:zed_link:Camera:0"],
            head_camera_link_name="torso_link4",
            navigation_window=3.0,
            translation_threshold=0.1,
            rotation_threshold=0.05,
            camera_tilt_threshold=0.4,
            gripper_link_paths={
                "left": {
                    "/World/scene_0/controllable__r1pro__robot/left_realsense_link/visuals",
                    "/World/scene_0/controllable__r1pro__robot/left_gripper_link/visuals",
                    "/World/scene_0/controllable__r1pro__robot/left_gripper_finger_link1/visuals",
                    "/World/scene_0/controllable__r1pro__robot/left_gripper_finger_link2/visuals",
                },
                "right": {
                    "/World/scene_0/controllable__r1pro__robot/right_realsense_link/visuals",
                    "/World/scene_0/controllable__r1pro__robot/right_gripper_link/visuals",
                    "/World/scene_0/controllable__r1pro__robot/right_gripper_finger_link1/visuals",
                    "/World/scene_0/controllable__r1pro__robot/right_gripper_finger_link2/visuals",
                },
            },
        )
        for metric_name in ACTIVE_QA_METRICS:
            metric_info = ALL_QA_METRICS[metric_name]
            create_fcn = (
                metric_info["cls"]
                if metric_info["init"] is None
                else metric_info["init"]
            )
            init_kwargs = {
                arg: metric_kwargs[arg]
                for arg in extract_arg_names(create_fcn)
                if arg in metric_kwargs
            }
            metric = create_fcn(**init_kwargs)
            env.add_metric(name=metric_name, metric=metric)

    num_samples = [
        env.input_hdf5["data"][key].attrs["num_samples"]
        for key in env.input_hdf5["data"].keys()
    ]
    episode_id = num_samples.index(max(num_samples))
    print(f" >>> Replaying episode {episode_id} with {num_samples[episode_id]} steps")

    env.playback_episode(
        episode_id=episode_id,
        record_data=False,
        video_keys=video_keys,
    )

    print("Playback complete. Saving data...")
    env.save_data()

    if run_qa:
        episode_metrics = env.aggregate_metrics(flatten=True)
        success, results = aggregate_episode_validation(
            task=task_name, all_episode_metrics=episode_metrics
        )
        print(f"QA Validation: {'SUCCESS' if success else 'FAILURE'}")
        results_path = os.path.join(os.path.dirname(input_path), f"{input_filename}_qa_results.json")
        with open(results_path, "w+") as f:
            json.dump(
                {"success": success, "results": results, "metrics": episode_metrics},
                f,
                cls=TorchEncoder,
                indent=4,
            )
        print(f"QA results saved to {results_path}")

    print(f"Successfully processed {os.path.basename(input_path)}")
    return episode_id


def main():
    parser = argparse.ArgumentParser(
        description="Replay HDF5 files and generate videos"
    )
    parser.add_argument("input", type=str, help="Path to the HDF5 file")
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        required=True,
        help="Task name (e.g., opening, placing_book_on_shelf)",
    )
    parser.add_argument(
        "--flush_every_n_steps", type=int, default=1000, help="Flush data every N steps"
    )
    parser.add_argument(
        "--qa", action="store_true", help="Run QA metrics during replay"
    )

    args = parser.parse_args()

    _ = replay_hdf5_to_video(
        input_path=args.input,
        task_name=args.task,
        flush_every_n_steps=args.flush_every_n_steps,
        run_qa=args.qa,
    )

    print("All done!")
    og.shutdown()


if __name__ == "__main__":
    main()
