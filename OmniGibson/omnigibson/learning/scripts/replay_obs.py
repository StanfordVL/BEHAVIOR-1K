import argparse
import h5py
import json
import omnigibson as og
import os
import sys
import numpy as np
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
        traj_grp["obs"].attrs["instance_name_mapping"] = json.dumps(VisionSensor.INSTANCE_ID_REGISTRY)
        
        for name, sensor in self.env.robots[0].sensors.items():
            if f"robot_r1::{name}" in ROBOT_CAMERA_NAMES:
                # add camera intrinsics as attrs
                traj_grp["obs"].attrs[f"robot_r1::{name}::intrinsics"] = sensor.intrinsic_matrix

                # add instance mapping keys as attrs
                instance_map = th.from_numpy(traj_grp["obs"][f"robot_r1::{name}::seg_instance_id"][:])
                instance_appearances = {}
                for i in range(instance_map.shape[0]):
                    instance_appearances[i] = th.unique(instance_map[i]).tolist()
                traj_grp["obs"].attrs[f"robot_r1::{name}::instance_appearances"] = json.dumps(instance_appearances)




def replay_hdf5_file(
    hdf_input_path: str, 
    camera_names: list=ROBOT_CAMERA_NAMES,
    generate_low_dim: bool=False,
    generate_rgbd: bool=False, 
    generate_pcd: bool=False,
    generate_seg: bool=False,
    generate_bbox: bool=False,
) -> None:
    """
    Replays a single HDF5 file and saves videos to a new folder

    Args:
        hdf_input_path: Path to the HDF5 file to replay
        camera_names: List of camera names to process 
        generate_low_dim: If True, generates low dimensional data from the replayed data
        generate_rgbd: If True, generates RGBD videos from the replayed data
        generate_pcd: If True, generates point cloud data from RGBD videos
        generate_seg: If True, generates segmentation data from the replayed data
        generate_bbox: If True, generates bounding box data from the replayed data
    """
    # get processed folder path
    low_dim_dir = os.path.join(os.path.dirname(os.path.dirname(hdf_input_path)), "low_dim")
    os.makedirs(low_dim_dir, exist_ok=True)
    base_name = os.path.basename(hdf_input_path)
    demo_name = os.path.splitext(base_name)[0]

    if generate_low_dim:
        # Define resolution for consistency
        WRIST_RESOLUTION = (240, 240)
        HEAD_RESOLUTION = (240, 240)

        # This flag is needed to run data playback wrapper
        gm.ENABLE_TRANSITION_RULES = False

        modalities = []
        if generate_rgbd:
            modalities += ["rgb", "depth_linear"]
        if generate_seg:
            modalities += ["seg_semantic", "seg_instance_id"]
        if generate_bbox:
            pass
            # modalities += ["bbox_2d_tight"]
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
            output_path=os.path.join(low_dim_dir, base_name),
            # compression={"compression": "gzip", "compression_opts": 1},
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
                if generate_rgbd:
                    rgbd_dir = os.path.join(os.path.dirname(os.path.dirname(hdf_input_path)), "rgbd", demo_name, f"demo_{episode_id}")
                    os.makedirs(rgbd_dir, exist_ok=True)
                    # RGB video writer
                    video_writers.append(create_video_writer(
                        fpath=f"{rgbd_dir}/{camera_name}::rgb.mp4",
                        resolution=WRIST_RESOLUTION,
                        codec_name="libx265",
                        pix_fmt="yuv420p",
                    ))
                    write_video(env.hdf5_file[f"data/demo_{episode_id}/obs/{camera_name}::rgb"], video_writers[-1], mode="rgb")
                    # Depth video writer
                    video_writers.append(create_video_writer(
                        fpath=f"{rgbd_dir}/{camera_name}::depth_linear.mp4",
                        resolution=WRIST_RESOLUTION,
                        codec_name="libx265",
                        pix_fmt="yuv420p10le",    
                    ))  
                    write_video(env.hdf5_file[f"data/demo_{episode_id}/obs/{camera_name}::depth_linear"], video_writers[-1], mode="depth")
                if generate_seg:
                    seg_dir = os.path.join(os.path.dirname(os.path.dirname(hdf_input_path)), "seg", demo_name, f"demo_{episode_id}")
                    os.makedirs(seg_dir, exist_ok=True)
                    # Segmentation video writer
                    sem_seg_original = env.hdf5_file[f"data/demo_{episode_id}/obs/{camera_name}::seg_semantic"][:]
                    sem_ids = th.unique(th.from_numpy(sem_seg_original))
                    video_writers.append(create_video_writer(
                        fpath=f"{seg_dir}/{camera_name}::seg_semantic.mp4",
                        resolution=WRIST_RESOLUTION,
                        codec_name="libx265",
                        pix_fmt="yuv420p", 
                    ))
                    write_video(sem_seg_original, video_writers[-1], mode="seg", seg_ids=sem_ids)
                    video_writers.append(create_video_writer(
                        fpath=f"{seg_dir}/{camera_name}::seg_instance_id.mp4",
                        resolution=WRIST_RESOLUTION,
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

    # Generate point cloud data from RGBD
    if generate_pcd:
        rgbd_to_pcd(os.path.dirname(os.path.dirname(hdf_input_path)), demo_name, camera_names)

    print(f"Successfully processed {hdf_input_path}")


def rgbd_to_pcd(task_folder: str, base_name: str, robot_camera_names: list=ROBOT_CAMERA_NAMES):
    """
    Generate point cloud data from RGBD data in the specified task folder.
    Args:
        task_folder (str): Path to the task folder containing RGBD data.
        base_name (str): Base name of the HDF5 file to process (without file extension).
        robot_camera_names (list): List of camera names to process.
    """
    print(f"Generating point cloud data from RGBD for {base_name} in {task_folder}")
    assert os.path.exists(task_folder), f"Task folder {task_folder} does not exist."
    output_dir = os.path.join(task_folder, "pcd")
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(f"{task_folder}/low_dim/{base_name}.hdf5", "r") as low_dim_f:
        # create a new hdf5 file to store the point cloud data
        with h5py.File(f"{task_folder}/pcd/{base_name}.hdf5", "w") as out_f:
            for demo_name in low_dim_f["data"]:
                low_dim_data = low_dim_f["data"][demo_name]["obs"]
                data_size = low_dim_data[f"robot_r1::robot_base_link_pose"].shape[0]
                fused_pcd_dset = out_f.create_dataset(
                    f"data/{demo_name}/robot_r1::fused_pcd", 
                    shape=(data_size, 4096, 6),
                    compression="gzip",
                    compression_opts=9,
                )
                pcd_semantic_dset = out_f.create_dataset(
                    f"data/{demo_name}/robot_r1::pcd_semantic", 
                    shape=(data_size, 4096),
                    compression="gzip",
                    compression_opts=9,
                )
                # We batch process every 500 frames
                print("data_size", data_size)
                for i in range(0, data_size, 500):
                    print("i", i)
                    obs = dict() # to store rgbd and pass into process_fused_point_cloud
                    # get all camera intrinsics
                    camera_intrinsics = {}
                    for robot_camera_name in robot_camera_names:
                        camera_intrinsics[robot_camera_name] = th.from_numpy(
                            low_dim_data.attrs[f"{robot_camera_name}::intrinsics"][:]
                            ).to(device="cuda")
                        # retrieve rgbd data from videos
                        robot_name, camera_name = robot_camera_name.split("::")
                        obs[f"{robot_name}::robot_base_link_pose"] = th.from_numpy(
                            low_dim_data[f"{robot_name}::robot_base_link_pose"][i:i+500]
                        ).to(device="cuda")
                        obs[f"{robot_name}::{camera_name}::rgb"] =  th.from_numpy(
                            low_dim_data[f"{robot_name}::{camera_name}::rgb"][i:i+500]
                        ).to(device="cuda")
                        obs[f"{robot_name}::{camera_name}::depth_linear"] = th.from_numpy(
                            low_dim_data[f"{robot_name}::{camera_name}::depth_linear"][i:i+500]
                        ).to(device="cuda")
                        obs[f"{robot_name}::{camera_name}::pose"] = th.from_numpy(
                            low_dim_data[f"{robot_name}::{camera_name}::pose"][i:i+500]
                        ).to(device="cuda")
                        obs[f"{robot_name}::{camera_name}::seg_semantic"] = th.from_numpy(
                            low_dim_data[f"{robot_name}::{camera_name}::seg_semantic"][i:i+500]
                        )
                    # process the fused point cloud
                    pcd, seg = process_fused_point_cloud(
                        obs=obs,
                        robot_name=robot_name,
                        camera_intrinsics=camera_intrinsics
                    )
                    fused_pcd_dset[i:i+500] = pcd
                    pcd_semantic_dset[i:i+500] = seg
                    print("i", i, "done")

    print(f"Point cloud data saved!")


def main():
    parser = argparse.ArgumentParser(description="Replay HDF5 files and save videos")
    parser.add_argument("--dir", help="Directory containing HDF5 files to process")
    parser.add_argument("--files", nargs="*", help="Individual HDF5 file(s) to process")
    parser.add_argument("--low_dim", action="store_true", help="Include this flag to generate low dimensional data")
    parser.add_argument("--rgbd", action="store_true", help="Include this flag to generate rgbd videos")
    parser.add_argument("--pcd", action="store_true", help="Include this flag to generate point cloud data from RGBD")
    parser.add_argument("--seg", action="store_true", help="Include this flag to generate segmentation maps" )
    parser.add_argument("--bbox", action="store_true", help="Include this flag to generate bounding box data" )

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

        replay_hdf5_file(
            hdf_file, 
            generate_low_dim=args.low_dim, 
            generate_rgbd=args.rgbd, 
            generate_pcd=args.pcd, 
            generate_seg=args.seg,
            generate_bbox=args.bbox,
        )

    print("All done!")
    og.shutdown()


if __name__ == "__main__":
    main()
