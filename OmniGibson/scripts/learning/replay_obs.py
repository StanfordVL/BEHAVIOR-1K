import argparse
import csv
import gspread
import h5py
import json
import numpy as np
import omnigibson as og
import os
import omnigibson.utils.transform_utils as T
import pandas as pd
import time
import torch as th
import torch.nn.functional as F
import yaml
from omnigibson.envs import DataPlaybackWrapper
from omnigibson.sensors import VisionSensor
from omnigibson.learning.utils.dataset_utils import update_google_sheet, makedirs_with_mode
from omnigibson.learning.utils.eval_utils import (
    PROPRIOCEPTION_INDICES,
    TASK_NAMES_TO_INDICES,
    TASK_INDICES_TO_NAMES,
    ROBOT_CAMERA_NAMES,
    CAMERA_INTRINSICS,
    HEAD_RESOLUTION,
    WRIST_RESOLUTION,
)
from omnigibson.learning.utils.obs_utils import (
    create_video_writer,
    process_fused_point_cloud,
    write_video,
    instance_id_to_instance,
    instance_to_bbox,
    rgbd_vid_to_pcd,
)
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import create_module_logger
from typing import Dict, Tuple


# Create module logger
log = create_module_logger(module_name="replay_obs")
# set level to be info
log.setLevel(20)

gm.RENDER_VIEWER_CAMERA = False
gm.DEFAULT_VIEWER_WIDTH = 128
gm.DEFAULT_VIEWER_HEIGHT = 128

FLUSH_EVERY_N_STEPS = 500


class BehaviorDataPlaybackWrapper(DataPlaybackWrapper):
    def _process_obs(self, obs, info):
        robot = self.env.robots[0]
        base_pose = robot.get_position_orientation()
        cam_rel_poses = []
        for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
            assert camera_name.split("::")[1] in robot.sensors, f"Camera {camera_name} not found in robot sensors"
            if f"{camera_name}::seg_semantic" in obs:
                # remove seg semantic map (Alternatively, change this line to store seg semantic instead)
                obs.pop(f"{camera_name}::seg_semantic")
            if f"{camera_name}::seg_instance_id" in obs:
                # move seg instance maps to cpu
                obs[f"{camera_name}::seg_instance_id"] = obs[f"{camera_name}::seg_instance_id"].cpu()
            # store camera pose
            cam_pose = robot.sensors[camera_name.split("::")[1]].get_position_orientation()
            cam_rel_poses.append(th.cat(T.relative_pose_transform(*cam_pose, *base_pose)))
        obs["robot_r1::cam_rel_poses"] = th.cat(cam_rel_poses, axis=-1)
        return obs

    def postprocess_traj_group(self, traj_grp):
        """
        Runs any necessary postprocessing on the given trajectory group @traj_grp. This should be an
        in-place operation!

        Args:
            traj_grp (h5py.Group): Trajectory group to postprocess
        """
        log.info(f"Postprocessing trajectory group {traj_grp.name}")
        traj_grp.attrs["robot_type"] = "R1Pro"
        # Add the list of task obs keys as attrs (this is list of strs)
        traj_grp.attrs["task_obs_keys"] = self.env.task.low_dim_obs_keys
        # add instance mapping keys as attrs
        traj_grp.attrs["ins_id_mapping"] = json.dumps(VisionSensor.INSTANCE_ID_REGISTRY)

        camera_names = set(ROBOT_CAMERA_NAMES["R1Pro"].values())
        for name in self.env.robots[0].sensors:
            if f"robot_r1::{name}" in camera_names:
                # add unique instance ids as attrs
                unique_ins_ids = set()
                # batch process to avoid memory issues
                for i in range(0, traj_grp["obs"][f"robot_r1::{name}::seg_instance_id"].shape[0], FLUSH_EVERY_N_STEPS):
                    unique_ins_ids.update(
                        th.unique(
                            th.from_numpy(
                                traj_grp["obs"][f"robot_r1::{name}::seg_instance_id"][i : i + FLUSH_EVERY_N_STEPS]
                            )
                        )
                        .to(th.uint32)
                        .tolist()
                    )
                traj_grp.attrs[f"robot_r1::{name}::unique_ins_ids"] = list(unique_ins_ids)
        log.info(f"Postprocessing trajectory group {traj_grp.name} done")


def replay_hdf5_file(
    data_folder: str,
    task_id: int,
    demo_id: int,
    camera_names: Dict[str, str] = ROBOT_CAMERA_NAMES["R1Pro"],
    generate_rgbd: bool = False,
    generate_seg: bool = False,
    generate_bbox: bool = False,
    flush_every_n_steps: int = 500,
    offline_rgbd: bool = False,
) -> None:
    """
    Replays a single HDF5 file and saves videos to a new folder

    Args:
        data_folder: data folder
        task_id: ID of the task to replay
        demo_id: ID of the demo to replay
        camera_names: Dict of camera names to process
        generate_rgbd: If True, generates RGBD videos from the replayed data
        generate_seg: If True, generates segmentation data from the replayed data
        generate_bbox: If True, generates bounding box data from the replayed data
        flush_every_n_steps: Number of steps to flush the data after
        offline_rgbd: If True, generates RGBD videos from h5 after replaying
    Returns:
        episode_id: ID of the episode
    """
    if generate_bbox:
        assert generate_rgbd and generate_seg, "Bounding box data requires rgb and segmentation data"
    # get processed folder path
    task_name = TASK_INDICES_TO_NAMES[task_id]
    replay_dir = os.path.join(data_folder, "replayed")
    makedirs_with_mode(replay_dir)

    # This flag is needed to run data playback wrapper
    gm.ENABLE_TRANSITION_RULES = False

    modalities = []
    if generate_rgbd:
        modalities += ["rgb", "depth_linear"]
    if generate_seg:
        modalities += ["seg_semantic", "seg_instance_id"]
    # Robot sensor configuration
    robot_sensor_config = {
        "VisionSensor": {
            "modalities": modalities,
            "sensor_kwargs": {
                "image_height": WRIST_RESOLUTION[0],
                "image_width": WRIST_RESOLUTION[1],
            },
        },
    }

    # Create the environment
    additional_wrapper_configs = []

    # get full scene file
    task_scene_file_folder = os.path.join(
        os.path.dirname(os.path.dirname(og.__path__[0])), "joylo", "sampled_task", task_name
    )
    full_scene_file = None
    for file in os.listdir(task_scene_file_folder):
        if file.endswith(".json") and "partial_rooms" not in file:
            full_scene_file = os.path.join(task_scene_file_folder, file)
    assert full_scene_file is not None, f"No full scene file found in {task_scene_file_folder}"
    # optimizations for data replay
    load_room_instances = None
    try:
        with open(
            f"{gm.DATA_PATH}/2025-challenge-task-instances/metadata/B50_task_misc.csv", newline="", encoding="utf-8"
        ) as f:
            task_misc_csv = csv.reader(f, delimiter=",", quotechar='"')
            for row in task_misc_csv:
                if task_name in row[1]:
                    load_room_instances = row[2].strip().split("\n")
                    break
    except FileNotFoundError as e:
        log.error(
            "No B50_task_misc.csv file found in 2025-challenge-task-instances/metadata folder. Please ensure the dataset is up to date."
        )
        raise e
    assert load_room_instances is not None, "load room instance not found!"
    env = BehaviorDataPlaybackWrapper.create_from_hdf5(
        input_path=f"{data_folder}/2025-challenge-rawdata/task-{task_id:04d}/episode_{demo_id:08d}.hdf5",
        output_path=os.path.join(replay_dir, f"episode_{demo_id:08d}.hdf5"),
        compression={"compression": "lzf"},
        robot_obs_modalities=["proprio"],
        robot_proprio_keys=list(PROPRIOCEPTION_INDICES["R1Pro"].keys()),
        robot_sensor_config=robot_sensor_config,
        external_sensors_config=dict(),
        n_render_iterations=3,
        flush_every_n_traj=1,
        flush_every_n_steps=flush_every_n_steps,
        additional_wrapper_configs=additional_wrapper_configs,
        full_scene_file=full_scene_file,
        include_robot_control=False,
        include_contacts=False,
        load_room_instances=load_room_instances,
    )

    # Modify head camera
    if generate_rgbd:
        env.robots[0].sensors["robot_r1:zed_link:Camera:0"].horizontal_aperture = 40.0
        env.robots[0].sensors["robot_r1:zed_link:Camera:0"].image_height = HEAD_RESOLUTION[0]
        env.robots[0].sensors["robot_r1:zed_link:Camera:0"].image_width = HEAD_RESOLUTION[1]
        # reload observation space
        env.load_observation_space()

    # Replay the dataset
    # get the episode id with the largest number of samples
    num_samples = [env.input_hdf5["data"][key].attrs["num_samples"] for key in env.input_hdf5["data"].keys()]
    episode_id = num_samples.index(max(num_samples))
    log.info(f" >>> Replaying episode {episode_id}")
    # initialize video writers if required
    video_writers = dict()
    if generate_rgbd:
        for camera_id, camera_name in camera_names.items():
            rgb_dir = os.path.join(
                data_folder,
                "2025-challenge-demos",
                "videos",
                f"task-{task_id:04d}",
                f"observation.images.rgb.{camera_id}",
            )
            depth_dir = os.path.join(
                data_folder,
                "2025-challenge-demos",
                "videos",
                f"task-{task_id:04d}",
                f"observation.images.depth.{camera_id}",
            )
            makedirs_with_mode(rgb_dir)
            makedirs_with_mode(depth_dir)
            resolution = HEAD_RESOLUTION if "zed" in camera_name else WRIST_RESOLUTION
            # RGB video writer
            video_writers[f"{camera_name}::rgb"] = create_video_writer(
                fpath=f"{rgb_dir}/episode_{demo_id:08d}.mp4",
                resolution=resolution,
                codec_name="libx265",
                pix_fmt="yuv420p",
                stream_options={"x265-params": "log-level=none"},
            )
            # Depth video writer
            video_writers[f"{camera_name}::depth_linear"] = create_video_writer(
                fpath=f"{depth_dir}/episode_{demo_id:08d}.mp4",
                resolution=resolution,
                codec_name="libx265",
                pix_fmt="yuv420p10le",
                stream_options={"x265-params": "lossless=1:log-level=none"},
            )

    env.playback_episode(
        episode_id=episode_id,
        record_data=True,
        video_writers=video_writers if not offline_rgbd else None,
    )

    # now store obs as videos
    for camera_id, camera_name in camera_names.items():
        resolution = HEAD_RESOLUTION if "zed" in camera_name else WRIST_RESOLUTION
        if generate_rgbd and offline_rgbd:
            write_video(
                env.hdf5_file[f"data/demo_{episode_id}/obs/{camera_name}::rgb"],
                video_writer=video_writers[f"{camera_name}::rgb"],
                batch_size=flush_every_n_steps,
                mode="rgb",
            )
            log.info(f"Saved rgb video for {camera_name}")
            write_video(
                env.hdf5_file[f"data/demo_{episode_id}/obs/{camera_name}::depth_linear"],
                video_writer=video_writers[f"{camera_name}::depth_linear"],
                batch_size=flush_every_n_steps,
                mode="depth",
            )
            log.info(f"Saved depth video for {camera_name}")
        if generate_seg:
            seg_dir = os.path.join(
                data_folder,
                "2025-challenge-demos",
                "videos",
                f"task-{task_id:04d}",
                f"observation.images.seg_instance_id.{camera_id}",
            )
            makedirs_with_mode(seg_dir)
            video_writers[f"{camera_name}::seg_instance_id"] = create_video_writer(
                fpath=f"{seg_dir}/episode_{demo_id:08d}.mp4",
                resolution=resolution,
                codec_name="libx265",
                pix_fmt="yuv420p",
                stream_options={"x265-params": "log-level=none"},
            )
            ins_id_ids = env.hdf5_file[f"data/demo_{episode_id}"].attrs[f"{camera_name}::unique_ins_ids"]
            write_video(
                env.hdf5_file[f"data/demo_{episode_id}/obs/{camera_name}::seg_instance_id"],
                video_writer=video_writers[f"{camera_name}::seg_instance_id"],
                batch_size=flush_every_n_steps,
                mode="seg",
                seg_ids=ins_id_ids,
            )
            log.info(f"Saved seg video for {camera_name}")
        if generate_bbox:
            # We only generate bbox for head camera
            if "zed" in camera_name:
                bbox_dir = os.path.join(
                    data_folder,
                    "2025-challenge-demos",
                    "videos",
                    f"task-{task_id:04d}",
                    f"observation.images.bbox.{camera_id}",
                )
                makedirs_with_mode(bbox_dir)
                video_writers[f"{camera_name}::bbox"] = create_video_writer(
                    fpath=f"{bbox_dir}/episode_{demo_id:08d}.mp4",
                    resolution=resolution,
                    codec_name="libx265",
                    pix_fmt="yuv420p",
                    stream_options={"x265-params": "log-level=none"},
                )
                # read task relevant objects from google sheet
                task_relevant_objs = None
                with open(
                    f"{gm.DATA_PATH}/2025-challenge-task-instances/metadata/B50_object_instance_ID.csv",
                    newline="",
                    encoding="utf-8",
                ) as f:
                    oi_csv = csv.reader(f, delimiter=",", quotechar='"')
                    for row in oi_csv:
                        if row[0] == task_name:
                            task_relevant_objs = row[1] + row[2]
                            break
                assert task_relevant_objs is not None, "Task relevant objects not found!"
                instance_id_mapping = json.loads(env.hdf5_file[f"data/demo_{episode_id}"].attrs["ins_id_mapping"])
                instance_id_mapping = {int(k): v for k, v in instance_id_mapping.items()}
                unique_ins_ids = env.hdf5_file[f"data/demo_{episode_id}"].attrs[f"{camera_name}::unique_ins_ids"]
                for i in range(
                    0,
                    env.hdf5_file[f"data/demo_{episode_id}/obs/{camera_name}::seg_instance_id"].shape[0],
                    flush_every_n_steps,
                ):
                    instance_seg, instance_mapping = instance_id_to_instance(
                        th.from_numpy(
                            env.hdf5_file[f"data/demo_{episode_id}/obs/{camera_name}::seg_instance_id"][
                                i : i + flush_every_n_steps
                            ]
                        ),
                        instance_id_mapping,
                        unique_ins_ids,
                    )
                    instance_mapping = {k: v for k, v in instance_mapping.items() if v in task_relevant_objs}
                    bbox = instance_to_bbox(instance_seg, instance_mapping, set(instance_mapping.keys()))
                    write_video(
                        th.from_numpy(
                            env.hdf5_file[f"data/demo_{episode_id}/obs/{camera_name}::rgb"][i : i + flush_every_n_steps]
                        ),
                        video_writer=video_writers[f"{camera_name}::bbox"],
                        batch_size=flush_every_n_steps,
                        mode="bbox",
                        bbox=bbox,
                        instance_mapping=instance_mapping,
                        task_relevant_objects=task_relevant_objs,
                    )
                log.info(f"Saved bbox video for {camera_name}")
    # Close all video writers
    for container, stream in video_writers.values():
        # Flush any remaining packets
        for packet in stream.encode():
            container.mux(packet)
        # Close the container
        container.close()

    log.info("Playback complete. Saving data...")
    env.save_data()

    log.info(f"Successfully processed episode_{demo_id:08d}")
    return episode_id


def generate_low_dim_data(
    data_folder: str,
    task_id: int,
    demo_id: int,
    episode_id: int,
):
    """
    Post-process the replayed low-dimensional data (proprio, action, task-info, etc) to parquet.
    """
    makedirs_with_mode(f"{data_folder}/2025-challenge-demos/data/task-{task_id:04d}")
    makedirs_with_mode(f"{data_folder}/2025-challenge-demos/meta/episodes/task-{task_id:04d}")
    with h5py.File(f"{data_folder}/replayed/episode_{demo_id:08d}.hdf5", "r") as replayed_f:
        actions = np.array(replayed_f["data"][f"demo_{episode_id}"]["action"][:], dtype=np.float32)
        proprio = np.array(replayed_f["data"][f"demo_{episode_id}"]["obs"]["robot_r1::proprio"][:], dtype=np.float32)
        task_info = np.array(replayed_f["data"][f"demo_{episode_id}"]["obs"]["task::low_dim"][:], dtype=np.float32)
        cam_rel_poses = np.array(
            replayed_f["data"][f"demo_{episode_id}"]["obs"]["robot_r1::cam_rel_poses"][:], dtype=np.float32
        )
        # check if the data is valid
        assert (
            actions.shape[0] == proprio.shape[0] == task_info.shape[0]
        ), "Action, proprio, and task-info must have the same length"
        T = len(actions)

        data = {
            "index": np.arange(T, dtype=np.int64),
            "episode_index": np.zeros(T, dtype=np.int64) + demo_id,
            "task_index": np.zeros(T, dtype=np.int64) + task_id,
            "timestamp": np.arange(T, dtype=np.float64) / 30.0,  # 30 fps
            "observation.state": list(proprio),
            "observation.cam_rel_poses": list(cam_rel_poses),
            "action": list(actions),
            "observation.task_info": list(task_info),
        }
        df = pd.DataFrame(data)
        df.to_parquet(
            f"{data_folder}/2025-challenge-demos/data/task-{task_id:04d}/episode_{demo_id:08d}.parquet", index=False
        )

        task_metadata = {}
        for attr_name in replayed_f["data"].attrs:
            if isinstance(replayed_f["data"].attrs[attr_name], np.int64):
                task_metadata[attr_name] = int(replayed_f["data"].attrs[attr_name])
            elif isinstance(replayed_f["data"].attrs[attr_name], np.ndarray):
                task_metadata[attr_name] = replayed_f["data"].attrs[attr_name].tolist()
            else:
                task_metadata[attr_name] = replayed_f["data"].attrs[attr_name]
        for attr_name in replayed_f["data"][f"demo_{episode_id}"].attrs:
            if isinstance(replayed_f["data"][f"demo_{episode_id}"].attrs[attr_name], np.int64):
                task_metadata[attr_name] = int(replayed_f["data"][f"demo_{episode_id}"].attrs[attr_name])
            elif isinstance(replayed_f["data"][f"demo_{episode_id}"].attrs[attr_name], np.ndarray):
                task_metadata[attr_name] = replayed_f["data"][f"demo_{episode_id}"].attrs[attr_name].tolist()
            else:
                task_metadata[attr_name] = replayed_f["data"][f"demo_{episode_id}"].attrs[attr_name]
        with open(
            f"{data_folder}/2025-challenge-demos/meta/episodes/task-{task_id:04d}/episode_{demo_id:08d}.json", "w"
        ) as f:
            json.dump(task_metadata, f, indent=4)
    log.info(f"Successfully processed {data_folder}/replayed/episode_{demo_id:08d}.hdf5")


def rgbd_gt_to_pcd(
    data_folder: str,
    task_id: int,
    demo_id: int,
    episode_id: int,
    robot_camera_names: Dict[str, str] = ROBOT_CAMERA_NAMES["R1Pro"],
    downsample_ratio: int = 4,
    pcd_range: Tuple[float, float, float, float, float, float] = (
        -0.2,
        1.0,
        -1.0,
        1.0,
        -0.2,
        1.5,
    ),  # x_min, x_max, y_min, y_max, z_min, z_max
    pcd_num_points: int = 4096,
    batch_size: int = 500,
    use_fps: bool = False,
):
    """
    Generate point cloud data from ground truth RGBD data (HDF5) in the specified task folder.
    Args:
        data_folder (str): Path to the data folder containing RGBD data.
        task_id (int): Task ID for the task being processed.
        demo_id (int): Demo ID for the episode being processed.
        episode_id (int): Episode ID for the episode being processed.
        robot_camera_names (dict): Dict of camera names to process.
        downsample_ratio (int): Downsample ratio for the camera resolution.
        pcd_range (tuple): Range of the point cloud.
        pcd_num_points (int): Number of points to sample from the point cloud.
        batch_size (int): Number of frames to process in each batch.
        use_fps (bool): Whether to use farthest point sampling for point cloud downsampling.
    """
    log.info(f"Generating point cloud data from RGBD for {demo_id:08d} in {data_folder}")
    output_dir = os.path.join(data_folder, "pcd_gt", f"task-{task_id:04d}")
    makedirs_with_mode(output_dir)

    with h5py.File(f"{data_folder}/replayed/episode_{demo_id:08d}.hdf5", "r") as in_f:
        # create a new hdf5 file to store the point cloud data
        with h5py.File(f"{output_dir}/episode_{demo_id:08d}.hdf5", "w") as out_f:
            data = in_f["data"][f"demo_{episode_id}"]["obs"]
            data_size = data["robot_r1::cam_rel_poses"].shape[0]
            fused_pcd_dset = out_f.create_dataset(
                f"data/demo_{episode_id}/robot_r1::fused_pcd",
                shape=(data_size, pcd_num_points, 6),
                compression="lzf",
            )
            # We batch process every batch_size frames
            for i in range(0, data_size, batch_size):
                log.info(f"Processing batch {i} of {data_size}...")
                obs = dict()  # to store rgbd and pass into process_fused_point_cloud
                obs["cam_rel_poses"] = th.from_numpy(data["robot_r1::cam_rel_poses"][i : i + batch_size])
                # get all camera intrinsics
                camera_intrinsics = {}
                for camera_id, robot_camera_name in robot_camera_names.items():
                    resolution = HEAD_RESOLUTION if camera_id == "head" else WRIST_RESOLUTION
                    # Calculate the downsampled camera intrinsics
                    camera_intrinsics[robot_camera_name] = (
                        th.from_numpy(CAMERA_INTRINSICS["R1Pro"][camera_id]) / downsample_ratio
                    )
                    camera_intrinsics[robot_camera_name][-1, -1] = 1.0
                    obs[f"{robot_camera_name}::rgb"] = F.interpolate(
                        th.from_numpy(data[f"{robot_camera_name}::rgb"][i : i + batch_size, :, :, :3]).movedim(-1, -3),
                        size=(resolution[0] // downsample_ratio, resolution[1] // downsample_ratio),
                        mode="nearest-exact",
                    ).movedim(-3, -1)
                    obs[f"{robot_camera_name}::depth_linear"] = F.interpolate(
                        th.from_numpy(data[f"{robot_camera_name}::depth_linear"][i : i + batch_size]).unsqueeze(0),
                        size=(resolution[0] // downsample_ratio, resolution[1] // downsample_ratio),
                        mode="nearest-exact",
                    ).squeeze(0)

                # process the fused point cloud
                pcd = process_fused_point_cloud(
                    obs=obs,
                    camera_intrinsics=camera_intrinsics,
                    pcd_range=pcd_range,
                    pcd_num_points=pcd_num_points,
                    use_fps=use_fps,
                    verbose=True,
                )
                log.info("Saving point cloud data...")
                fused_pcd_dset[i : i + batch_size] = pcd.cpu()

    log.info("Point cloud data saved!")


def main():
    parser = argparse.ArgumentParser(description="Replay HDF5 files and save videos")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the data folder")
    parser.add_argument("--data_url", type=str, default="", required=False, help="URL to raw data")
    parser.add_argument("--task_name", type=str, required=True, help="Task name to process")
    parser.add_argument("--demo_id", type=int, required=True, help="Demo ID to process")
    parser.add_argument("--low_dim", action="store_true", help="Include this flag to generate low dimensional data")
    parser.add_argument("--rgbd", action="store_true", help="Include this flag to generate rgbd videos")
    parser.add_argument("--offline_rgbd", action="store_true", help="Whether rgbd videos should be generated offline")

    parser.add_argument(
        "--pcd_gt", action="store_true", help="Include this flag to generate point cloud data from ground truth RGBD"
    )
    parser.add_argument(
        "--pcd_vid", action="store_true", help="Include this flag to generate point cloud data from RGBD videos"
    )
    parser.add_argument("--seg", action="store_true", help="Include this flag to generate segmentation maps")
    parser.add_argument("--bbox", action="store_true", help="Include this flag to generate bounding box data")
    # [Internal use only] the following arguments are for Google Sheets integration
    parser.add_argument("--update_sheet", action="store_true", help="Include this flag to update the Google Sheet")
    parser.add_argument("--row", type=int, required=False, help="Row number to update")

    args = parser.parse_args()
    task_id = TASK_NAMES_TO_INDICES[args.task_name]

    # Process each file
    if not os.path.exists(
        f"{args.data_folder}/2025-challenge-rawdata/task-{task_id:04d}/episode_{args.demo_id:08d}.hdf5"
    ):
        if args.data_url:
            from omnigibson.learning.utils.dataset_utils import download_and_extract_data

            instance_id = int((args.demo_id % 1e4) // 10)
            traj_id = int(args.demo_id % 10)
            download_and_extract_data(args.data_url, args.data_folder, args.task_name, instance_id, traj_id)
        else:
            raise FileNotFoundError(
                f"Error: File episode_{args.demo_id:08d}.hdf5 does not exists under {args.data_folder}"
            )
    if args.rgbd or args.seg or args.bbox:
        episode_id = replay_hdf5_file(
            data_folder=args.data_folder,
            task_id=task_id,
            demo_id=args.demo_id,
            generate_rgbd=args.rgbd,
            generate_seg=args.seg,
            generate_bbox=args.bbox,
            flush_every_n_steps=FLUSH_EVERY_N_STEPS,
            offline_rgbd=args.offline_rgbd,
        )
    else:
        # TODO: change this
        episode_id = 0

    if args.low_dim:
        generate_low_dim_data(
            data_folder=args.data_folder, task_id=task_id, demo_id=args.demo_id, episode_id=episode_id
        )
    if args.pcd_gt or args.pcd_vid:
        with open(
            f"{os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}/omnigibson/learning/configs/task/behavior.yaml"
        ) as f:
            pcd_range = tuple(yaml.safe_load(f)["pcd_range"])
        if args.pcd_gt:
            rgbd_gt_to_pcd(
                data_folder=args.data_folder,
                task_id=task_id,
                demo_id=args.demo_id,
                episode_id=episode_id,
                robot_camera_names=ROBOT_CAMERA_NAMES["R1Pro"],
                pcd_range=pcd_range,
                downsample_ratio=4,
                pcd_num_points=4096,
                batch_size=1000,
                use_fps=True,
            )
        if args.pcd_vid:
            rgbd_vid_to_pcd(
                data_folder=args.data_folder,
                task_id=task_id,
                demo_id=args.demo_id,
                episode_id=episode_id,
                robot_camera_names=ROBOT_CAMERA_NAMES["R1Pro"],
                pcd_range=pcd_range,
                downsample_ratio=4,
                pcd_num_points=4096,
                batch_size=1000,
                use_fps=True,
            )

    # remove replayed hdf5 to free up storage
    try:
        os.remove(f"{args.data_folder}/replayed/episode_{args.demo_id:08d}.hdf5")
    except FileNotFoundError:
        log.warning(f"File {args.data_folder}/replayed/episode_{args.demo_id:08d}.hdf5 not found")
    # Optionally update google sheet
    if args.update_sheet:
        credentials_path = f"{os.environ.get('HOME')}/Documents/credentials"
        sheet_update_success = False
        for _ in range(5):
            try:
                update_google_sheet(credentials_path, args.task_name, args.row)
                sheet_update_success = True
                break
            except gspread.exceptions.APIError as e:
                log.error(f"Failed to update Google Sheet: {e}")
                time.sleep(60)
        if not sheet_update_success:
            # update one last time
            update_google_sheet(credentials_path, args.task_name, args.row)

    log.info("All done!")
    og.shutdown()


if __name__ == "__main__":
    main()
