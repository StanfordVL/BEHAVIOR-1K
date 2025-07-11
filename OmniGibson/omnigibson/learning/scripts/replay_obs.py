import argparse
import h5py
import json
import omnigibson as og
import os
import sys
import torch as th
from omnigibson.envs import DataPlaybackWrapper
from omnigibson.sensors import VisionSensor
from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES
from omnigibson.learning.utils.obs_utils import (
    create_video_writer, 
    process_fused_point_cloud, 
    write_video,
)
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)
# set level to be info
log.setLevel(20)

gm.RENDER_VIEWER_CAMERA = False
gm.DEFAULT_VIEWER_WIDTH = 128
gm.DEFAULT_VIEWER_HEIGHT = 128

ROBOT_CAMERA_NAMES = [
    "robot_r1::robot_r1:left_realsense_link:Camera:0",
    "robot_r1::robot_r1:right_realsense_link:Camera:0",
    "robot_r1::robot_r1:zed_link:Camera:0",
]


class BehaviorDataPlaybackWrapper(DataPlaybackWrapper):
    def _process_obs(self, obs, info):
        robot = self.env.robots[0]
        for camera_name in ROBOT_CAMERA_NAMES:
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
        # Store observations as videos
        # Add the list of task obs keys as attrs (this is list of strs)
        traj_grp["obs"]["task::low_dim"].attrs["task_obs_keys"] = self.env.task.low_dim_obs_keys
        # add instance mapping keys as attrs
        traj_grp["obs"].attrs["ins_id_mapping"] = json.dumps(VisionSensor.INSTANCE_ID_REGISTRY)
        
        for name, sensor in self.env.robots[0].sensors.items():
            if f"robot_r1::{name}" in ROBOT_CAMERA_NAMES:
                # add camera intrinsics as attrs
                traj_grp["obs"].attrs[f"robot_r1::{name}::intrinsics"] = sensor.intrinsic_matrix


def replay_hdf5_file(
    hdf_input_path: str, 
    camera_names: list=ROBOT_CAMERA_NAMES,
    generate_rgbd: bool=False, 
    generate_seg: bool=False,
) -> None:
    """
    Replays a single HDF5 file and saves videos to a new folder

    Args:
        hdf_input_path: Path to the HDF5 file to replay
        camera_names: List of camera names to process 
        generate_rgbd: If True, generates RGBD videos from the replayed data
        generate_seg: If True, generates segmentation data from the replayed data
    """
    # get processed folder path
    replay_dir = os.path.join(os.path.dirname(os.path.dirname(hdf_input_path)), "replayed")
    os.makedirs(replay_dir, exist_ok=True)
    base_name = os.path.basename(hdf_input_path)
    demo_name = os.path.splitext(base_name)[0]

    # Define resolution for consistency
    WRIST_RESOLUTION = (480, 480)
    HEAD_RESOLUTION = (720, 720)

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
        flush_every_n_steps=500,
        additional_wrapper_configs=additional_wrapper_configs,
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
        print(f" >>> Replaying episode {episode_id}")
        env.playback_episode(
            episode_id=episode_id,
            record_data=True,
        )

        # now store obs as videos
        video_writers = []            
        for camera_name in camera_names:
            resolution = HEAD_RESOLUTION if "zed" in camera_name else WRIST_RESOLUTION
            if generate_rgbd:
                rgbd_dir = os.path.join(os.path.dirname(os.path.dirname(hdf_input_path)), "rgbd", demo_name, f"demo_{episode_id}")
                os.makedirs(rgbd_dir, exist_ok=True)
                # RGB video writer
                video_writers.append(create_video_writer(
                    fpath=f"{rgbd_dir}/{camera_name}::rgb.mp4",
                    resolution=resolution,
                    codec_name="libx265",
                    pix_fmt="yuv420p",
                ))
                write_video(env.hdf5_file[f"data/demo_{episode_id}/obs/{camera_name}::rgb"], video_writers[-1], mode="rgb")
                # Depth video writer
                video_writers.append(create_video_writer(
                    fpath=f"{rgbd_dir}/{camera_name}::depth_linear.mp4",
                    resolution=resolution,
                    codec_name="libx265",
                    pix_fmt="yuv420p10le",    
                    stream_options={"crf": "8"},
                ))  
                write_video(env.hdf5_file[f"data/demo_{episode_id}/obs/{camera_name}::depth_linear"], video_writers[-1], mode="depth")
            if generate_seg:
                seg_dir = os.path.join(os.path.dirname(os.path.dirname(hdf_input_path)), "seg", demo_name, f"demo_{episode_id}")
                os.makedirs(seg_dir, exist_ok=True)
                video_writers.append(create_video_writer(
                    fpath=f"{seg_dir}/{camera_name}::seg_instance_id.mp4",
                    resolution=resolution,
                    codec_name="libx265",
                    pix_fmt="yuv420p", 
                ))
                ins_id_seg_original = env.hdf5_file[f"data/demo_{episode_id}/obs/{camera_name}::seg_instance_id"][:]
                ins_id_ids = th.unique(th.from_numpy(ins_id_seg_original))
                write_video(ins_id_seg_original, video_writers[-1], mode="seg", seg_ids=ins_id_ids)
        # Close all video writers
        for container, stream in video_writers:
            # Flush any remaining packets
            for packet in stream.encode():
                container.mux(packet)
            # Close the container
            container.close()

        print("Playback complete. Saving data...")
        env.save_data()

    print(f"Successfully processed {hdf_input_path}")


def generate_low_dim_data(
    task_folder: str,
    base_name: str,
    camera_names: list=ROBOT_CAMERA_NAMES,
):
    """
    Post-process the replayed data.
    """
    os.makedirs(f"{task_folder}/low_dim", exist_ok=True)
    with h5py.File(f"{task_folder}/replayed/{base_name}", "r") as replayed_f:
        # First, construct low dim observations:
        with h5py.File(f"{task_folder}/low_dim/{base_name}", "w") as low_dim_f:
            # create data group
            low_dim_f.create_group("data")
            # copy attrs
            for attr_name in replayed_f["data"].attrs:
                low_dim_f["data"].attrs[attr_name] = replayed_f["data"].attrs[attr_name]
            # copy datasets
            for demo_name in replayed_f["data"]:
                # create demo group
                low_dim_f["data"].create_group(demo_name)
                # copy all attrs from replayed_f to low_dim_f
                for attr_name in replayed_f["data"][demo_name].attrs:
                    low_dim_f["data"][demo_name].attrs[attr_name] = replayed_f["data"][demo_name].attrs[attr_name]
                # copy action from replayed_f to low_dim_f
                low_dim_f.create_dataset(
                    f"data/{demo_name}/action", 
                    data=replayed_f["data"][demo_name]["action"],
                    compression="gzip",
                    compression_opts=9,
                    shuffle=True,
                )
                # create obs group
                low_dim_f["data"][demo_name].create_group("obs")
                # copy low dim obs:
                robot_name = camera_names[0].split("::")[0]
                low_dim_f.create_dataset(
                    f"data/{demo_name}/obs/task::low_dim", 
                    data=replayed_f["data"][demo_name]["obs"]["task::low_dim"],
                    compression="gzip",
                    compression_opts=9,
                    shuffle=True,
                )
                low_dim_f.create_dataset(
                    f"data/{demo_name}/obs/{robot_name}::proprio", 
                    data=replayed_f["data"][demo_name]["obs"][f"{robot_name}::proprio"],
                    compression="gzip",
                    compression_opts=9,
                    shuffle=True,
                )
                for camera_name in camera_names:
                    low_dim_f.create_dataset(
                        f"data/{demo_name}/obs/{camera_name}::pose", 
                        data=replayed_f["data"][demo_name]["obs"][f"{camera_name}::pose"],
                        compression="gzip",
                        compression_opts=9,
                        shuffle=True,
                    )
    print(f"Successfully processed {task_folder}/replayed/{base_name}.hdf5")


def rgbd_to_pcd(
    task_folder: str, 
    base_name: str, 
    robot_camera_names: list=ROBOT_CAMERA_NAMES,
    pcd_num_points: int=4096,
    batch_size: int=500,
    use_fps: bool=True,
):
    """
    Generate point cloud data from RGBD data in the specified task folder.
    Args:
        task_folder (str): Path to the task folder containing RGBD data.
        base_name (str): Base name of the HDF5 file to process (without file extension).
        robot_camera_names (list): List of camera names to process.
        pcd_num_points (int): Number of points to sample from the point cloud.
        batch_size (int): Number of frames to process in each batch.
    """
    print(f"Generating point cloud data from RGBD for {base_name} in {task_folder}")
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
                    # compression="gzip",
                    # compression_opts=9,
                )
                pcd_semantic_dset = out_f.create_dataset(
                    f"data/{demo_name}/robot_r1::pcd_semantic", 
                    shape=(data_size, pcd_num_points),
                    # compression="gzip",
                    # compression_opts=9,
                )
                # We batch process every batch_size frames
                for i in range(0, data_size, batch_size):
                    print(f"Processing batch {i} of {data_size}...")
                    obs = dict() # to store rgbd and pass into process_fused_point_cloud
                    # get all camera intrinsics
                    camera_intrinsics = {}
                    for robot_camera_name in robot_camera_names:
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
                    print("Saving point cloud data...")
                    fused_pcd_dset[i:i+batch_size] = pcd
                    pcd_semantic_dset[i:i+batch_size] = seg
                    print("i", i, "done")

    print(f"Point cloud data saved!")


def main():
    parser = argparse.ArgumentParser(description="Replay HDF5 files and save videos")
    parser.add_argument("--file", help="Raw HDF5 file to process")
    parser.add_argument("--low_dim", action="store_true", help="Include this flag to generate low dimensional data")
    parser.add_argument("--rgbd", action="store_true", help="Include this flag to generate rgbd videos")
    parser.add_argument("--pcd", action="store_true", help="Include this flag to generate point cloud data from RGBD")
    parser.add_argument("--seg", action="store_true", help="Include this flag to generate segmentation maps" )

    args = parser.parse_args()

    # Process each file
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} does not exist", file=sys.stderr)
        return
    if args.rgbd or args.seg:
        replay_hdf5_file(
            args.file, 
            generate_rgbd=args.rgbd, 
            generate_seg=args.seg,
        )

    if args.low_dim:
        generate_low_dim_data(
            task_folder=os.path.dirname(os.path.dirname(args.file)),
            base_name=os.path.basename(args.file),
        )
    if args.pcd:
        rgbd_to_pcd(
            task_folder=os.path.dirname(os.path.dirname(args.file)),
            base_name=os.path.basename(args.file),
            robot_camera_names=ROBOT_CAMERA_NAMES,
            pcd_num_points=4096,
            batch_size=200,
            use_fps=True,
        )

    print("All done!")
    og.shutdown()


if __name__ == "__main__":
    main()
