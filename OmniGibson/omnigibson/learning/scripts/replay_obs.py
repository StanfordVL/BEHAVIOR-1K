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
from omnigibson.learning.utils.obs_utils import process_fused_point_cloud, VideoLoader
from omnigibson.macros import gm
from omnigibson.utils.python_utils import create_object_from_init_info, h5py_group_to_torch
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
            # pop rgbd keys from obs
            obs.pop(f"{camera_name}::rgb", None)
            obs.pop(f"{camera_name}::depth_linear", None)

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
        # Add the list of task obs keys as attrs (this is list of strs)
        traj_grp["obs"]["task::low_dim"].attrs["task_obs_keys"] = self.env.task.low_dim_obs_keys
        # add instance mapping keys as attrs
        traj_grp["obs"].attrs["instance_mapping"] = json.dumps(VisionSensor.INSTANCE_ID_REGISTRY)
        # add camera intrinsics as attrs
        for name, sensor in self.env.robots[0].sensors.items():
            if f"robot_r1::{name}" in ROBOT_CAMERA_NAMES:
                traj_grp["obs"].attrs[f"robot_r1::{name}::intrinsics"] = sensor.intrinsic_matrix

    def flush_seg_h5(self, episode_length, camera_names, start_idx, end_idx):
        # pop seg from observation
        for camera_name in camera_names:
            sem, ins = [], []
            for i in range(start_idx, end_idx):
                sem.append(self.current_traj_history[i]["obs"].pop(f"{camera_name}::seg_semantic"))
                ins.append(self.current_traj_history[i]["obs"].pop(f"{camera_name}::seg_instance_id"))
            self.sem_dset[camera_name][start_idx:end_idx] = th.cat(sem)
            self.ins_dset[camera_name][start_idx:end_idx] = th.cat(sem)

    def playback_episode(
        self, 
        episode_id, 
        record_data=True, 
        video_writers=None, 
        video_keys=None, 
        camera_names=None, 
        segmentation_output_path=None
    ):
        """
        Playback episode @episode_id, and optionally record observation data if @record is True

        Args:
            episode_id (int): Episode to playback. This should be a valid demo ID number from the inputted collected
                data hdf5 file
            record_data (bool): Whether to record data during playback or not
            video_writers (None or list of (container, stream)): If specified, writer objects that RGB frames will be written to
            video_keys (None or list of str): If specified, observation keys representing the frames to write to video.
                If @video_writers is specified, this must also be specified!
            generate_seg (bool): whether we want to generate segementation maps
            camera_names (None or list of str): If specified, camera names to retrieve segmentation maps.
            segmentation_output_path (None or str): If specified, the path to save segmentaiton mask to
        """
        generate_seg = segmentation_output_path is not None
        # Validate video_writers and video_keys
        if video_writers is not None:
            assert video_keys is not None, "If video_writers is specified, video_rgb_keys must also be specified!"
            assert len(video_writers) == len(
                video_keys
            ), "video_writers and video_rgb_keys must have the same length!"

        data_grp = self.input_hdf5["data"]
        assert f"demo_{episode_id}" in data_grp, f"No valid episode with ID {episode_id} found!"
        traj_grp = data_grp[f"demo_{episode_id}"]

        # Grab episode data
        # Skip early if found malformed data
        try:
            transitions = json.loads(traj_grp.attrs["transitions"])
            traj_grp = h5py_group_to_torch(traj_grp)
            init_metadata = traj_grp["init_metadata"]
            action = traj_grp["action"]
            state = traj_grp["state"]
            state_size = traj_grp["state_size"]
            reward = traj_grp["reward"]
            terminated = traj_grp["terminated"]
            truncated = traj_grp["truncated"]
        except KeyError as e:
            print(f"Got error when trying to load episode {episode_id}:")
            print(f"Error: {str(e)}")
            return

        # Reset environment and update this to be the new initial state
        self.scene.restore(self.scene_file, update_initial_file=True)

        # Reset object attributes from the stored metadata
        with og.sim.stopped():
            for attr, vals in init_metadata.items():
                assert len(vals) == self.scene.n_objects
            for i, obj in enumerate(self.scene.objects):
                for attr, vals in init_metadata.items():
                    val = vals[i]
                    setattr(obj, attr, val.item() if val.ndim == 0 else val)
        self.reset()

        # Restore to initial state
        og.sim.load_state(state[0, : int(state_size[0])], serialized=True)

        # If record, record initial observations
        if record_data:
            self.current_obs, _, _, _, init_info = self.env.step(action=action[0], n_render_iterations=self.n_render_iterations)
            # If writing to video, write desired frame
            if video_writers is not None:
                self.write_videos(video_writers=video_writers, video_keys=video_keys)
            step_data = {"obs": self._process_obs(obs=self.current_obs, info=init_info)}
            self.current_traj_history.append(step_data)
            if generate_seg:
                store_every_n_step = 250
                start_idx, end_idx = 0, store_every_n_step
                # create dataset with episode size:
                self.seg_hdf5 = h5py.File(segmentation_output_path, "w")
                T = action.shape[0] + 1
                self.sem_dset, self.ins_dset = dict(), dict()
                for camera_name in camera_names:
                    H, W = step_data["obs"][f"{camera_name}::seg_semantic"].shape
                    self.sem_dset[camera_name] = self.seg_hdf5.create_dataset(
                        f"data/demo_{episode_id}/obs/{camera_name}::seg_semantic", 
                        shape=(T, H, W), 
                        dtype='uint32',
                        compression="gzip",
                        compression_opts=9,
                    )
                    self.ins_dset[camera_name] = self.seg_hdf5.create_dataset(
                        f"data/demo_{episode_id}/obs/{camera_name}::seg_instance_id", 
                        shape=(T, H, W), 
                        dtype='uint32',
                        compression="gzip",
                        compression_opts=9,
                    )
                    
        for i, (a, s, ss, r, te, tr) in enumerate(
            zip(action, state[1:], state_size[1:], reward, terminated, truncated)
        ):
            if i % store_every_n_step == store_every_n_step - 1:
                if record_data and generate_seg:
                    log.info(f"Flusing segmentation maps")
                    self.flush_segmentations(camera_names=camera_names, start_idx=start_idx, end_idx=end_idx)
                    start_idx = end_idx
                    end_idx = min(end_idx + store_every_n_step, T)
                    
                log.info(f"Playing back episode {episode_id}: {i+1} / {len(action)} steps...")
            # Execute any transitions that should occur at this current step
            if str(i) in transitions:
                cur_transitions = transitions[str(i)]
                scene = og.sim.scenes[0]
                for add_sys_name in cur_transitions["systems"]["add"]:
                    scene.get_system(add_sys_name, force_init=True)
                for remove_sys_name in cur_transitions["systems"]["remove"]:
                    scene.clear_system(remove_sys_name)
                for remove_obj_name in cur_transitions["objects"]["remove"]:
                    obj = scene.object_registry("name", remove_obj_name)
                    scene.remove_object(obj)
                for j, add_obj_info in enumerate(cur_transitions["objects"]["add"]):
                    obj = create_object_from_init_info(add_obj_info)
                    scene.add_object(obj)
                    obj.set_position(th.ones(3) * 100.0 + th.ones(3) * 5 * j)
                # Step physics to initialize any new objects
                og.sim.step()

            # Restore the sim state, and take a very small step with the action to make sure physics are
            # properly propagated after the sim state update
            og.sim.load_state(s[: int(ss)], serialized=True)
            self.current_obs, _, _, _, info = self.env.step(action=a, n_render_iterations=self.n_render_iterations)

            # If writing to video, write desired frame
            if video_writers is not None:
                self.write_videos(video_writers=video_writers, video_keys=video_keys)

            # If recording, record data
            if record_data:
                step_data = self._parse_step_data(
                    action=a,
                    obs=self.current_obs,
                    reward=r,
                    terminated=te,
                    truncated=tr,
                    info=info,
                )
                self.current_traj_history.append(step_data)

            self.step_count += 1

        if record_data:
            self.flush_segmentations(camera_names=camera_names, start_idx=start_idx, end_idx=end_idx)
            self.flush_current_traj()
            # Also flush segmentation file if applicable
            if generate_seg:
                self.seg_hdf5.flush()  # Flush data to disk to avoid large memory footprint
                # Retrieve the file descriptor and use os.fsync() to flush to disk
                fd = self.seg_hdf5.id.get_vfd_handle()
                os.fsync(fd)
                log.info("Flushing segmentation hdf5")


    def flush_segmentations(self, camera_names, start_idx, end_idx):
        for camera_name in camera_names:
            seg_data, ins_data = [], []
            for i in range(start_idx, end_idx):
                seg_data.append(self.current_traj_history[i]["obs"].pop(f"{camera_name}::seg_semantic"))
                ins_data.append(self.current_traj_history[i]["obs"].pop(f"{camera_name}::seg_instance_id"))
            self.sem_dset[camera_name][start_idx:end_idx] = th.stack(seg_data).cpu().numpy()
            self.ins_dset[camera_name][start_idx:end_idx] = th.stack(ins_data).cpu().numpy()

def replay_hdf5_file(
    hdf_input_path: str, 
    camera_names: list=ROBOT_CAMERA_NAMES,
    generate_low_dim: bool=False,
    generate_rgbd: bool=False, 
    generate_pcd: bool=False,
    generate_seg: bool=False,
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
    """
    # get processed folder path
    low_dim_dir = os.path.join(os.path.dirname(os.path.dirname(hdf_input_path)), "low_dim")
    seg_dir = os.path.join(os.path.dirname(os.path.dirname(hdf_input_path)), "segmentations")
    if generate_seg:
        os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(low_dim_dir, exist_ok=True)
    base_name = os.path.basename(hdf_input_path)
    demo_name = os.path.splitext(base_name)[0]

    if generate_low_dim:
        # Define output paths
        low_dim_output_path = os.path.join(low_dim_dir, base_name)
        segmentation_output_path = os.path.join(seg_dir, base_name) if generate_seg else None

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
            output_path=low_dim_output_path,
            compression={"compression": "gzip", "compression_opts": 9},
            robot_obs_modalities=["proprio"],
            robot_proprio_keys=list(PROPRIOCEPTION_INDICES["R1Pro"].keys()),
            robot_sensor_config=robot_sensor_config,
            external_sensors_config=dict(),
            n_render_iterations=1,
            only_successes=False,
            additional_wrapper_configs=additional_wrapper_configs,
        )

        # Modify head camera
        if generate_rgbd:
            env.robots[0].sensors["robot_r1:zed_link:Camera:0"].horizontal_aperture = 40.0
            env.robots[0].sensors["robot_r1:zed_link:Camera:0"].image_height = HEAD_RESOLUTION[0]
            env.robots[0].sensors["robot_r1:zed_link:Camera:0"].image_width = HEAD_RESOLUTION[1]
            # reload observation space
            env.load_observation_space()

        # Create a list to store video writers and RGB keys
        video_writers = []
        video_keys = []

        if generate_rgbd:
            rgbd_dir = os.path.join(os.path.dirname(os.path.dirname(hdf_input_path)), "rgbd", demo_name)
            os.makedirs(rgbd_dir, exist_ok=True)
            # Create video writer for robot cameras
            for camera_name in camera_names:
                # RGB video writer
                video_writers.append(env.create_video_writer(
                    fpath=f"{rgbd_dir}/{camera_name}::rgb.mp4",
                    resolution=WRIST_RESOLUTION,
                    codec_name="libx265",
                    context_options={"crf": "18"},
                ))
                video_keys.append(f"{camera_name}::rgb")
                # Depth video writer
                video_writers.append(env.create_video_writer(
                    fpath=f"{rgbd_dir}/{camera_name}::depth_linear.mp4",
                    resolution=WRIST_RESOLUTION,
                    codec_name="libx265",
                    pix_fmt="yuv420p10le",    
                ))
                video_keys.append(f"{camera_name}::depth_linear")


        # Playback the dataset with all video writers
        # We avoid calling playback_dataset and call playback_episode individually in order to manually
        # aggregate per-episode metrics
        for episode_id in range(env.input_hdf5["data"].attrs["n_episodes"]):
            print(f" >>> Replaying episode {episode_id}")

            env.playback_episode(
                episode_id=episode_id,
                record_data=True,
                video_writers=video_writers,
                video_keys=video_keys,
                camera_names=camera_names,
                segmentation_output_path=segmentation_output_path
            )

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


def rgbd_to_pcd(task_folder: str, base_name: str, robot_canera_names: list=ROBOT_CAMERA_NAMES):
    """
    Generate point cloud data from RGBD data in the specified task folder.
    Args:
        task_folder (str): Path to the task folder containing RGBD data.
        base_name (str): Base name of the HDF5 file to process (without file extension).
        robot_canera_names (list): List of camera names to process.
    """
    print(f"Generating point cloud data from RGBD for {base_name} in {task_folder}")
    assert os.path.exists(task_folder), f"Task folder {task_folder} does not exist."
    output_dir = os.path.join(task_folder, "pcd")
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(f"{task_folder}/low_dim/{base_name}.hdf5", "r") as low_dim_f:
        with h5py.File(f"{task_folder}/segmentations/{base_name}.hdf5", "r") as seg_f:
            obs = dict() # to store rgbd and pass into process_fused_point_cloud
            # create a new hdf5 file to store the point cloud data
            with h5py.File(f"{task_folder}/pcd/{base_name}.hdf5", "w") as out_f:
                for demo_name in low_dim_f["data"]:
                    low_dim_data = low_dim_f["data"][demo_name]["obs"]
                    # get all camera intrinsics
                    camera_intrinsics = {}
                    for robot_canera_name in robot_canera_names:
                        camera_intrinsics[robot_canera_name] = low_dim_data.attrs[f"{robot_canera_name}::intrinsics"]
                        # retrieve rgbd data from videos
                        robot_name, camera_name = robot_canera_name.split("::")
                        obs[f"{robot_name}::robot_base_link_pose"] = low_dim_data[f"{robot_name}::robot_base_link_pose"]
                        obs[f"{robot_name}::{camera_name}::rgb"] = VideoLoader(
                            path=f"{task_folder}/rgbd/{base_name}/{robot_name}::{camera_name}::rgb.mp4",
                            type="rgb",
                            streaming=True
                        )
                        obs[f"{robot_name}::{camera_name}::depth_linear"] = VideoLoader(
                            path=f"{task_folder}/rgbd/{base_name}/{robot_name}::{camera_name}::depth_linear.mp4",
                            type="depth",
                            streaming=True
                        )
                        obs[f"{robot_name}::{camera_name}::pose"] = low_dim_data[f"{robot_name}::{camera_name}::pose"]
                        obs[f"{robot_name}::{camera_name}::seg_semantic"] = seg_f["data"][demo_name]["obs"][f"{robot_name}::{camera_name}::seg_semantic"]
                    # process the fused point cloud
                    pcd, seg = process_fused_point_cloud(
                        obs=obs,
                        robot_name=robot_name,
                        camera_intrinsics=camera_intrinsics
                    )
                    out_f.create_dataset(
                        f"data/{demo_name}/robot_r1::fused_pcd", 
                        data=pcd,
                        compression="gzip",
                        compression_opts=9,
                    )
                    out_f.create_dataset(
                        f"data/{demo_name}/robot_r1::pcd_semantic", 
                        data=seg,
                        compression="gzip",
                        compression_opts=9,
                    )

    print(f"Point cloud data saved!")


def main():
    parser = argparse.ArgumentParser(description="Replay HDF5 files and save videos")
    parser.add_argument("--dir", help="Directory containing HDF5 files to process")
    parser.add_argument("--files", nargs="*", help="Individual HDF5 file(s) to process")
    parser.add_argument("--low_dim", action="store_true", help="Include this flag to generate low dimensional data")
    parser.add_argument("--rgbd", action="store_true", help="Include this flag to generate rgbd videos")
    parser.add_argument("--pcd", action="store_true", help="Include this flag to generate point cloud data from RGBD")
    parser.add_argument("--seg", action="store_true", help="Include this flag to generate segmentation maps" )

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
            generate_seg=args.seg
        )

    print("All done!")
    og.shutdown()


if __name__ == "__main__":
    main()
