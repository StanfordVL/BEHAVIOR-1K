import argparse
import h5py
import json
import numpy as np
import omnigibson as og
import os
import pandas as pd
import torch as th
from omnigibson.envs import DataPlaybackWrapper
from omnigibson.sensors import VisionSensor
from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES, TASK_INDICES, ROBOT_CAMERA_NAMES
from omnigibson.learning.utils.obs_utils import (
    create_video_writer, 
    process_fused_point_cloud, 
    write_video,
    instance_id_to_instance,
    instance_to_bbox,
)
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import create_module_logger
from typing import Dict


# Create module logger
log = create_module_logger(module_name="omnigibson.learning.scripts.replay_obs")
# set level to be info
log.setLevel(20)

gm.RENDER_VIEWER_CAMERA = False
gm.DEFAULT_VIEWER_WIDTH = 128
gm.DEFAULT_VIEWER_HEIGHT = 128

FLUSH_EVERY_N_STEPS = 500

class BehaviorDataPlaybackWrapper(DataPlaybackWrapper):
    def _process_obs(self, obs, info):
        robot = self.env.robots[0]
        for camera_name in ROBOT_CAMERA_NAMES.values():
            # move seg maps to cpu
            obs[f"{camera_name}::seg_semantic"] = obs[f"{camera_name}::seg_semantic"].cpu()
            obs[f"{camera_name}::seg_instance_id"] = obs[f"{camera_name}::seg_instance_id"].cpu()
            # store camera pose
            if camera_name.split("::")[1] in robot.sensors:
                obs[f"{camera_name}::pose"] = th.concatenate(
                    robot.sensors[camera_name.split("::")[1]].get_position_orientation()
                )
        # store the poses to obs
        obs["robot_r1::robot_base_link_pose"] = th.concatenate(robot.get_position_orientation())
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
        traj_grp.attrs["task_relevant_objs"] = [
            obj.unwrapped.name for obj in self.env.task.object_scope.values() if obj.unwrapped.name != "robot_r1"
        ]
        # add instance mapping keys as attrs
        traj_grp.attrs["ins_id_mapping"] = json.dumps(VisionSensor.INSTANCE_ID_REGISTRY)
        
        camera_names = set(ROBOT_CAMERA_NAMES.values())
        for name in self.env.robots[0].sensors:
            if f"robot_r1::{name}" in camera_names:
                # add unique instance ids as attrs
                unique_ins_ids = set()
                # batch process to avoid memory issues
                for i in range(0, traj_grp["obs"][f"robot_r1::{name}::seg_instance_id"].shape[0], FLUSH_EVERY_N_STEPS):
                    unique_ins_ids.update(th.unique(
                        th.from_numpy(traj_grp["obs"][f"robot_r1::{name}::seg_instance_id"][i:i+FLUSH_EVERY_N_STEPS])
                    ).to(th.uint32).tolist())
                traj_grp.attrs[f"robot_r1::{name}::unique_ins_ids"] = list(unique_ins_ids)
        log.info(f"Postprocessing trajectory group {traj_grp.name} done")


def replay_hdf5_file(
    hdf_input_path: str, 
    task_id: int,
    camera_names: Dict[str, str] = ROBOT_CAMERA_NAMES,
    generate_rgbd: bool=False, 
    generate_seg: bool=False,   
    generate_bbox: bool=False,
    flush_every_n_steps: int=500,
) -> None:
    """
    Replays a single HDF5 file and saves videos to a new folder

    Args:
        hdf_input_path: Path to the HDF5 raw data file to replay
        camera_names: Dict of camera names to process 
        generate_rgbd: If True, generates RGBD videos from the replayed data
        generate_seg: If True, generates segmentation data from the replayed data
        generate_bbox: If True, ge nerates bounding box data from the replayed data
        flush_every_n_steps: Number of steps to flush the data after
    """
    if generate_bbox:
        assert generate_rgbd and generate_seg, "Bounding box data requires rgb and segmentation data"
    # get processed folder path
    data_folder = os.path.dirname(os.path.dirname(os.path.dirname(hdf_input_path)))
    task_name = os.path.basename(os.path.dirname(hdf_input_path))
    replay_dir = os.path.join(data_folder, "replayed")
    os.makedirs(replay_dir, exist_ok=True)
    base_name = os.path.basename(hdf_input_path)
    demo_name = os.path.splitext(base_name)[0]
    demo_id = int(demo_name.split("_")[-1]) # 3 digit demo id

    # Define resolution for consistency
    WRIST_RESOLUTION = (120, 120)
    HEAD_RESOLUTION = (180, 180)

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

    env = BehaviorDataPlaybackWrapper.create_from_hdf5(
        input_path=hdf_input_path,
        output_path=os.path.join(replay_dir, base_name),
        compression={"compression": "lzf"},
        robot_obs_modalities=["proprio"],
        robot_proprio_keys=list(PROPRIOCEPTION_INDICES["R1Pro"].keys()),
        robot_sensor_config=robot_sensor_config,
        external_sensors_config=dict(),
        n_render_iterations=1,
        flush_every_n_traj=1,
        flush_every_n_steps=flush_every_n_steps,
        additional_wrapper_configs=additional_wrapper_configs,
        full_scene_file=full_scene_file
    )

    # Modify head camera
    if generate_rgbd:
        env.robots[0].sensors["robot_r1:zed_link:Camera:0"].horizontal_aperture = 40.0
        env.robots[0].sensors["robot_r1:zed_link:Camera:0"].image_height = HEAD_RESOLUTION[0]
        env.robots[0].sensors["robot_r1:zed_link:Camera:0"].image_width = HEAD_RESOLUTION[1]
        # reload observation space
        env.load_observation_space()

    # Replay the dataset
    for episode_id in range(env.input_hdf5["data"].attrs["n_episodes"]):
        log.info(f" >>> Replaying episode {episode_id}")
        env.playback_episode(
            episode_id=episode_id,
            record_data=True,
        )

        # now store obs as videos
        video_writers = []            
        for camera_id, camera_name in camera_names.items():
            resolution = HEAD_RESOLUTION if "zed" in camera_name else WRIST_RESOLUTION
            if generate_rgbd:
                rgb_dir = os.path.join(data_folder, "videos", f"task-{task_id:04d}", f"observation.images.rgb.{camera_id}")
                depth_dir = os.path.join(data_folder, "videos", f"task-{task_id:04d}", f"observation.images.depth.{camera_id}")
                os.makedirs(rgb_dir, exist_ok=True)
                os.makedirs(depth_dir, exist_ok=True)
                # RGB video writer
                video_writers.append(create_video_writer(
                    fpath=f"{rgb_dir}/episode_{task_id:04d}{demo_id:03d}.mp4",
                    resolution=resolution,
                    codec_name="libx265",
                    pix_fmt="yuv420p",
                    stream_options={"x265-params": "log-level=none"},
                ))
                write_video(
                    env.hdf5_file[f"data/demo_{episode_id}/obs/{camera_name}::rgb"], 
                    video_writer=video_writers[-1], 
                    batch_size=flush_every_n_steps, 
                    mode="rgb",
                )
                log.info(f"Saved rgb video for {camera_name}")
                # Depth video writer
                video_writers.append(create_video_writer(
                    fpath=f"{depth_dir}/episode_{task_id:04d}{demo_id:03d}.mp4",
                    resolution=resolution,
                    codec_name="libx265",
                    pix_fmt="yuv420p10le",    
                    stream_options={"crf": "8", "x265-params": "log-level=none"},
                ))  
                write_video(
                    env.hdf5_file[f"data/demo_{episode_id}/obs/{camera_name}::depth_linear"], 
                    video_writer=video_writers[-1], 
                    batch_size=flush_every_n_steps, 
                    mode="depth",
                )
                log.info(f"Saved depth video for {camera_name}")
            if generate_seg:
                seg_dir = os.path.join(data_folder, "videos", f"task-{task_id:04d}", f"observation.images.seg_instance_id.{camera_id}")
                os.makedirs(seg_dir, exist_ok=True)
                video_writers.append(create_video_writer(
                    fpath=f"{seg_dir}/episode_{task_id:04d}{demo_id:03d}.mp4",
                    resolution=resolution,
                    codec_name="libx265",
                    pix_fmt="yuv420p", 
                    stream_options={"x265-params": "log-level=none"},
                ))
                ins_id_seg_original = env.hdf5_file[f"data/demo_{episode_id}/obs/{camera_name}::seg_instance_id"][:]
                ins_id_ids = env.hdf5_file[f"data/demo_{episode_id}"].attrs[f"{camera_name}::unique_ins_ids"]
                write_video(
                    ins_id_seg_original, 
                    video_writer=video_writers[-1], 
                    batch_size=flush_every_n_steps, 
                    mode="seg", 
                    seg_ids=ins_id_ids,
                )
                log.info(f"Saved seg video for {camera_name}")
            if generate_bbox:
                # We only generate bbox for head camera
                if "zed" in camera_name: 
                    bbox_dir = os.path.join(data_folder, "videos", f"task-{task_id:04d}", f"observation.images.bbox.{camera_id}")
                    os.makedirs(bbox_dir, exist_ok=True)
                    video_writers.append(create_video_writer(
                        fpath=f"{bbox_dir}/episode_{task_id:04d}{demo_id:03d}.mp4",
                        resolution=resolution,
                        codec_name="libx265",
                        pix_fmt="yuv420p",
                        stream_options={"x265-params": "log-level=none"},
                    ))
                    task_relevant_objs = env.hdf5_file[f"data/demo_{episode_id}"].attrs["task_relevant_objs"]
                    instance_id_mapping = json.loads(env.hdf5_file[f"data/demo_{episode_id}"].attrs["ins_id_mapping"])
                    instance_id_mapping = {int(k): v for k, v in instance_id_mapping.items()}
                    unique_ins_ids = env.hdf5_file[f"data/demo_{episode_id}"].attrs[f"{camera_name}::unique_ins_ids"]
                    for i in range(0, env.hdf5_file[f"data/demo_{episode_id}/obs/{camera_name}::seg_instance_id"].shape[0], flush_every_n_steps):
                        instance_seg, instance_mapping = instance_id_to_instance(
                            th.from_numpy(env.hdf5_file[f"data/demo_{episode_id}/obs/{camera_name}::seg_instance_id"][i:i+flush_every_n_steps]), 
                            instance_id_mapping,
                            unique_ins_ids,
                        )
                        instance_mapping = {k: v for k, v in instance_mapping.items() if v in task_relevant_objs}
                        bbox = instance_to_bbox(instance_seg, instance_mapping, set(instance_mapping.keys()))
                        write_video(
                            th.from_numpy(env.hdf5_file[f"data/demo_{episode_id}/obs/{camera_name}::rgb"][i:i+flush_every_n_steps]), 
                            video_writer=video_writers[-1], 
                            batch_size=flush_every_n_steps,
                            mode="bbox",
                            bbox=bbox,
                            instance_mapping=instance_mapping,
                            task_relevant_objects=task_relevant_objs,
                        )
                    log.info(f"Saved bbox video for {camera_name}")
        # Close all video writers
        for container, stream in video_writers:
            # Flush any remaining packets
            for packet in stream.encode():
                container.mux(packet)
            # Close the container
            container.close()

        log.info("Playback complete. Saving data...")
        env.save_data()

    log.info(f"Successfully processed {hdf_input_path}")


def generate_low_dim_data(
    data_folder: str,
    task_id: int,
    base_name: str,
    camera_names: Dict[str, str] = ROBOT_CAMERA_NAMES,
):
    """
    Post-process the replayed low-dimensional data (proprio, action, task-info, etc) to parquet.
    """
    os.makedirs(f"{data_folder}/data/task-{task_id:04d}", exist_ok=True)
    os.makedirs(f"{data_folder}/meta/episodes/task-{task_id:04d}", exist_ok=True)
    demo_id = int(base_name.split("_")[-1].split(".")[0]) # 3 digit demo id
    with h5py.File(f"{data_folder}/replayed/{base_name}", "r") as replayed_f:
        for episode_id in range(replayed_f["data"].attrs["n_episodes"]):
            actions = np.array(replayed_f["data"][f"demo_{episode_id}"]["action"][:], dtype=np.float32)
            proprio = np.array(replayed_f["data"][f"demo_{episode_id}"]["obs"]["robot_r1::proprio"][:], dtype=np.float32)
            task_info = np.array(replayed_f["data"][f"demo_{episode_id}"]["obs"]["task::low_dim"][:], dtype=np.float32) 
            # check if the data is valid
            assert actions.shape[0] == proprio.shape[0] == task_info.shape[0], \
                "Action, proprio, and task-info must have the same length"
            T = len(actions)
            
            data = {
                "index": np.arange(T, dtype=np.int64),
                "episode_index": np.zeros(T, dtype=np.int64) + episode_id,
                "task_index": np.zeros(T, dtype=np.int64),
                "timestamp": np.arange(T, dtype=np.float64) / 30.0,  # 30 fps
                "observation.state": list(proprio),
                "action": list(actions),
                "observation.task_info": list(task_info),
            }
            for camera_id, camera_name in camera_names.items():
                data.update({
                    f"observation.camera_pose.{camera_id}": \
                        list(np.array(replayed_f["data"][f"demo_{episode_id}"]["obs"][f"{camera_name}::pose"], dtype=np.float32)),
                })
            df = pd.DataFrame(data)
            df.to_parquet(f"{data_folder}/data/task-{task_id:04d}/episode_{task_id:04d}{demo_id:03d}.parquet", index=False)
            
        
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
            with open(f"{data_folder}/meta/episodes/task-{task_id:04d}/episode_{task_id:04d}{demo_id:03d}.json", "w") as f:
                json.dump(task_metadata, f, indent=4)
    log.info(f"Successfully processed {data_folder}/replayed/{base_name}")


def rgbd_to_pcd(
    task_folder: str, 
    base_name: str, 
    robot_camera_names: Dict[str, str] = ROBOT_CAMERA_NAMES,
    pcd_num_points: int=1e5,
    batch_size: int=500,
    use_fps: bool=False,
):
    """
    Generate point cloud data from RGBD data in the specified task folder.
    Args:
        task_folder (str): Path to the task folder containing RGBD data.
        base_name (str): Base name of the HDF5 file to process (without file extension).
        robot_camera_names (dict): Dict of camera names to process.
        pcd_num_points (int): Number of points to sample from the point cloud.
        batch_size (int): Number of frames to process in each batch.
    """
    log.info(f"Generating point cloud data from RGBD for {base_name} in {task_folder}")
    assert os.path.exists(task_folder), f"Task folder {task_folder} does not exist."
    output_dir = os.path.join(task_folder, "pcd")
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(f"{task_folder}/replayed/{base_name}", "r") as in_f:
        # create a new hdf5 file to store the point cloud data
        with h5py.File(f"{output_dir}/{base_name}", "w") as out_f:
            for demo_name in in_f["data"]:
                data = in_f["data"][demo_name]["obs"]
                data_size = data[f"robot_r1::robot_base_link_pose"].shape[0]
                fused_pcd_dset = out_f.create_dataset(
                    f"data/{demo_name}/robot_r1::fused_pcd", 
                    shape=(data_size, pcd_num_points, 6),
                    compression="lzf",
                )
                pcd_semantic_dset = out_f.create_dataset(
                    f"data/{demo_name}/robot_r1::pcd_semantic", 
                    shape=(data_size, pcd_num_points),
                    compression="lzf",
                )
                # We batch process every batch_size frames
                for i in range(0, data_size, batch_size):
                    log.info(f"Processing batch {i} of {data_size}...")
                    obs = dict() # to store rgbd and pass into process_fused_point_cloud
                    # get all camera intrinsics
                    camera_intrinsics = {}
                    for robot_camera_name in robot_camera_names.values():
                        camera_intrinsics[robot_camera_name] = th.from_numpy(
                            data.attrs[f"{robot_camera_name}::intrinsics"][:]
                            )
                        robot_name, camera_name = robot_camera_name.split("::")
                        obs[f"{robot_name}::robot_base_link_pose"] = th.from_numpy(
                            data[f"{robot_name}::robot_base_link_pose"][i:i+batch_size]
                        )
                        obs[f"{robot_name}::{camera_name}::rgb"] = th.from_numpy(
                            data[f"{robot_name}::{camera_name}::rgb"][i:i+batch_size]
                        )
                        obs[f"{robot_name}::{camera_name}::depth_linear"] = th.from_numpy(
                            data[f"{robot_name}::{camera_name}::depth_linear"][i:i+batch_size]
                        )
                        obs[f"{robot_name}::{camera_name}::pose"] = th.from_numpy(
                            data[f"{robot_name}::{camera_name}::pose"][i:i+batch_size]
                        )
                        obs[f"{robot_name}::{camera_name}::seg_semantic"] = th.from_numpy(
                            data[f"{robot_name}::{camera_name}::seg_semantic"][i:i+batch_size]
                        )
                    # process the fused point cloud
                    pcd, seg = process_fused_point_cloud(
                        obs=obs,
                        robot_name=robot_name,
                        camera_intrinsics=camera_intrinsics,
                        pcd_num_points=pcd_num_points,
                        use_fps=use_fps,
                    )
                    log.info("Saving point cloud data...")
                    fused_pcd_dset[i:i+batch_size] = pcd
                    pcd_semantic_dset[i:i+batch_size] = seg

    log.info(f"Point cloud data saved!")


def main():
    parser = argparse.ArgumentParser(description="Replay HDF5 files and save videos")
    parser.add_argument("--file", help="Raw HDF5 file to process")
    parser.add_argument("--low_dim", action="store_true", help="Include this flag to generate low dimensional data")
    parser.add_argument("--rgbd", action="store_true", help="Include this flag to generate rgbd videos")
    parser.add_argument("--pcd", action="store_true", help="Include this flag to generate point cloud data from RGBD")
    parser.add_argument("--seg", action="store_true", help="Include this flag to generate segmentation maps" )
    parser.add_argument("--bbox", action="store_true", help="Include this flag to generate bounding box data" )

    args = parser.parse_args()

    task_indices_to_names = {v: k for k, v in TASK_INDICES.items()}
    task_id = task_indices_to_names[os.path.basename(os.path.dirname(args.file))]

    # Process each file
    if not os.path.exists(args.file):
        log.info(f"Error: File {args.file} does not exist")
        return
    if args.rgbd or args.seg or args.bbox:
        replay_hdf5_file(
            args.file, 
            task_id=task_id,
            generate_rgbd=args.rgbd, 
            generate_seg=args.seg,
            generate_bbox=args.bbox,
            flush_every_n_steps=FLUSH_EVERY_N_STEPS,
        )

    if args.low_dim:
        generate_low_dim_data(
            data_folder=os.path.dirname(os.path.dirname(os.path.dirname(args.file))),
            task_id=task_id,
            base_name=os.path.basename(args.file),
        )
    if args.pcd:
        rgbd_to_pcd(
            task_folder=os.path.dirname(os.path.dirname(args.file)),
            base_name=os.path.basename(args.file),
            robot_camera_names=ROBOT_CAMERA_NAMES,
            pcd_num_points=61200,
            batch_size=200,
            use_fps=False,
        )

    log.info("All done!")
    og.shutdown()


if __name__ == "__main__":
    main()
