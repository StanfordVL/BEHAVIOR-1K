import logging
import os
import shutil
import torch as th
from lerobot.datasets import LeRobotDataset
from lerobot.datasets.io_utils import write_info
from lerobot.utils.constants import HF_LEROBOT_HOME

import omnigibson.utils.transform_utils as T
from omnigibson.envs.env_base import Environment
from omnigibson.envs.data_wrapper import DataWrapper, DataPlaybackWrapper
from omnigibson.learning.utils.obs_utils import encode_depth_frame, decode_depth_frame
from omnigibson.sensors.vision_sensor import VisionSensor
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.asset_utils import get_omnigibson_git_hash
from omnigibson.tasks.behavior_task import BehaviorTask


# Create module logger
log = create_module_logger(module_name=__name__)
log.setLevel(logging.INFO)


class LeRobotDataWrapper(DataWrapper):
    """
    Specific data wrapper for writing data to LeRobot format.
    """

    def __init__(
        self,
        env: Environment,
        output_path: str,
        root_dir: str = HF_LEROBOT_HOME,
        overwrite: bool = True,
        only_successes: bool = True,
        flush_every_n_traj: int = 10,
        robot_type: str | None = None,
        task_name: str | None = None,
    ):
        """
        Args:
            env (Environment): The environment to wrap
            output_path (str): The path to the output lerobot dataset
            root_dir (str): Root directory to store output dataset files
            overwrite (bool): If set, will overwrite any pre-existing data found at @repo_id.
                Otherwise, will load the data and append to it
            only_successes (bool): Whether to only save successful episodes
            flush_every_n_traj (int): How often to flush (write) current data to file across episodes
            robot_type (None or str): Name of the robot within this dataset. If not specified, will be inferred
                from environment
            task_name (None or str): If specified, task that will be recorded in LeRobot dataset. If not specified,
                will try to automatically infer if the wrapped environment is a BehaviorTask
        """
        self._init_lerobot_kwargs(
            repo_id=output_path,
            root_dir=root_dir,
            robot_type=robot_type,
            env=env,
            task_name=task_name,
        )

        # Run super
        super().__init__(
            env=env,
            output_path=output_path,
            overwrite=overwrite,
            only_successes=only_successes,
            flush_every_n_traj=flush_every_n_traj,
        )

    def _init_lerobot_kwargs(
        self,
        repo_id: str,
        root_dir: str,
        robot_type: str | None,
        env: Environment,
        task_name: str | None,
    ) -> None:
        self.lerobot_dataset_kwargs = {
            "repo_id": repo_id,
            "root": f"{root_dir}/{repo_id}",
            "robot_type": env.robots[0].__class__.__name__.lower() if robot_type is None else robot_type,
            "use_videos": True,
            "streaming_encoding": True,
            "depth_map_encoding_fn": encode_depth_frame,
            "depth_map_decoding_fn": decode_depth_frame,
        }
        self.dataset = None
        self.obs_mapping = None
        self.controller_action_start_idxs = None

        if task_name is None:
            if isinstance(env.task, BehaviorTask):
                task_name = env.task.activity_name.replace("_", " ")
            else:
                raise ValueError("Task name must be specified if environment task is not a BehaviorTask!")
        self.task_name = task_name

    def _parse_step_data(
        self,
        action: th.Tensor,
        obs: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> dict:
        step_data = {
            "obs": self._process_obs(obs=obs, info=info),
            "action": action,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }
        return step_data

    @classmethod
    def get_lerobot_obs_mapping(cls, env: Environment, fps: int) -> tuple[dict[str, str], dict[str, dict]]:
        obs_mapping, obs_features = dict(), dict()
        for key, gym_shape in env.observation_space.items():
            modality = key.split("::")[-1]
            info = dict()
            # Parse the relevant name to assign
            obs_name_strs = key.split("::")[-2].split(":")
            # TODO @wensi-ai: hacky, fix this
            # filter out robot if applicable
            if len(obs_name_strs) == 4:
                obs_name_strs = obs_name_strs[1:]
            # Join with "_" and make lowercase to make final name
            obs_name = "_".join(obs_name_strs).lower()
            if "rgb" in modality:
                info["dtype"] = "video"
                info["shape"] = gym_shape.shape[:-1] + (3,)
                info["names"] = ["height", "width", "channel"]
                info["info"] = {
                    "video.fps": fps,
                    "video.height": gym_shape.shape[0],
                    "video.width": gym_shape.shape[1],
                    "video.channels": 3,
                    "video.codec": "hevc",
                    "video.pix_fmt": "yuv420p",
                    "video.g": 8,
                    "video.crf": 30,
                    "video.options": {
                        "x265-params": "log-level=0:bframes=0",
                    },
                    "video.is_depth_map": False,
                    "has_audio": False,
                }
            elif "depth" in modality:
                info["dtype"] = "video"
                info["shape"] = gym_shape.shape + (1,)
                info["names"] = ["height", "width", "channel"]
                info["info"] = {
                    "video.fps": fps,
                    "video.height": gym_shape.shape[0],
                    "video.width": gym_shape.shape[1],
                    "video.channels": 1,
                    "video.codec": "hevc",
                    "video.pix_fmt": "yuv420p12le",
                    "video.g": 8,
                    "video.crf": 0,
                    "video.options": {
                        "x265-params": "log-level=0:bframes=0",
                    },
                    "video.is_depth_map": True,
                    "has_audio": False,
                }

                # We also add relative camera transforms (wrt robot egocentric frame) in case we
                # want to convert depth to point clouds
                # So we add an extra entry here
                tf_name = f"observation.robot2cam_pose.{obs_name}"
                if tf_name not in obs_features:
                    obs_features[tf_name] = {
                        "dtype": "float32",
                        "shape": (7,),
                        "names": None,
                    }

            elif "proprio" in modality or "low_dim" in modality:
                info["dtype"] = "float32"
                info["shape"] = gym_shape.shape
                info["names"] = (None,)
            else:
                raise ValueError(f"Got LeRobot-incompatible observation modality: {modality}")

            # Add this key to features, and store the obs name mapping
            lerobot_obs_name = "observation.state" if "proprio" in modality else f"observation.{modality}.{obs_name}"
            obs_features[lerobot_obs_name] = info
            obs_mapping[key] = lerobot_obs_name

        return obs_mapping, obs_features

    def _process_obs(self, obs: dict[str, th.Tensor], info: dict) -> dict[str, th.Tensor]:
        # Add tfs to flattened obs
        robot_tf_inv = T.pose_inv(T.pose2mat(self.env.robots[0].get_position_orientation()))
        for sensor_group in (self.env.external_sensors, self.env.robots[0].sensors):
            if sensor_group is None:
                continue
            for name, sensor in sensor_group.items():
                obs[f"{name}::rel_pose"] = th.cat(
                    T.mat2pose(robot_tf_inv @ T.pose2mat(sensor.get_position_orientation()))
                )

        # Compose lerobot format obs
        frame = dict()
        for name in self.env.observation_space.keys():
            cur_obs = obs[name]
            # Prune alpha channel if keeping RGB
            if "rgb" in name:
                cur_obs = cur_obs[..., :3]
            elif "depth" in name:
                # Add channel dim at the end
                cur_obs = cur_obs.unsqueeze(-1)
                # If we haven't already added the sensor pose obs, do so now
                obs_name_strs = name.split("::")[-2].split(":")
                # TODO @wensi-ai: hacky, fix this
                # filter out robot if applicable
                if len(obs_name_strs) == 4:
                    obs_name_strs = obs_name_strs[1:]
                # Join with "_" and make lowercase to make final name
                obs_name = "_".join(obs_name_strs).lower()
                tf_name = f"observation.robot2cam_pose.{obs_name}"
                if tf_name not in frame:
                    sensor_name = name.split("::")[-2]
                    frame[tf_name] = obs[f"{sensor_name}::rel_pose"]
            elif "proprio" in name:
                # Map float64 -> float32
                cur_obs = cur_obs.float()
            # Add the observation to the current frame
            frame[self.obs_mapping[name]] = cur_obs

        return frame

    def create_dataset(self, output_path: str, env: Environment, overwrite: bool = True) -> None:
        # Sanity checks
        assert (
            output_path == self.lerobot_dataset_kwargs["repo_id"]
        ), f"Expected LeRobot repo_id path ({self.lerobot_dataset_kwargs['repo_id']}) to match output_path ({output_path})!"

        abs_output_path = f"{self.lerobot_dataset_kwargs['root']}"

        resume = False
        if os.path.exists(abs_output_path):
            if overwrite:
                # Remove any data from this path
                shutil.rmtree(abs_output_path)
                log.info(f"Overwriting existing LeRobot dataset at: {abs_output_path}")
            else:
                resume = True
                log.info(f"Resuming from existing LeRobot dataset at: {abs_output_path}")

        # For now, we only support a single robot for the sake of deterministic mapping ofrobot obs
        assert len(env.robots) == 1, "Only one robot supported for LeRobot dataset storage!"
        robot = env.robots[0]

        # Create LeRobot dataset, define features to store
        # Define standard features (RL-related entries, language instructions)
        features = {
            "action": {
                "dtype": "float32",
                "shape": env.action_space[robot.name].shape,
                "names": ["action"],
            },
            # RL-specific fields
            "next.reward": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["reward"],
            },
            "next.terminated": {
                "dtype": "bool",
                "shape": (1,),
                "names": ["done"],
            },
            "next.truncated": {
                "dtype": "bool",
                "shape": (1,),
                "names": ["done"],
            },
        }

        obs_mapping, obs_features = self.get_lerobot_obs_mapping(env=env, fps=self.fps)
        self.obs_mapping = obs_mapping
        features.update(obs_features)

        if not resume:
            self.dataset = LeRobotDataset.create(
                fps=self.fps,
                features=features,
                **self.lerobot_dataset_kwargs,
            )
        else:
            self.dataset = LeRobotDataset.resume(
                fps=self.fps,
                features=features,
                **self.lerobot_dataset_kwargs,
            )

        # Add in camera K matrices
        cam_intrinsics = dict()

        for sensor_name, sensor in env.external_sensors.items():
            if isinstance(sensor, VisionSensor):
                K = sensor.intrinsic_matrix.cpu()
                cam_intrinsics[sensor_name] = K.numpy().tolist()
        for sensor_name, sensor in env.robots[0].sensors.items():
            if isinstance(sensor, VisionSensor):
                # Remove robot naming prefix
                sensor_name = "_".join(sensor_name.split(":")[1:]).lower()
                K = sensor.intrinsic_matrix.cpu()
                cam_intrinsics[sensor_name] = K.numpy().tolist()
        self.dataset.meta.info["cam_intrinsics"] = cam_intrinsics
        self.dataset.meta.info["omnigibson_git_hash"] = get_omnigibson_git_hash()
        write_info(self.dataset.meta.info, self.dataset.meta.root)

    def process_traj_to_dataset(self, traj_data: list[dict]) -> None:
        # Write to LeRobot dataset
        # The dataset length is (N_steps + 1), since the first entry only includes the env reset observations
        # LeRobot expects (s,a) tuples to be paired with rewards from the next step, so we match the obs with
        # all other entries from the proceeding (i.e.: t+1) step

        for frame_idx, traj_step in enumerate(traj_data):
            if frame_idx == 0:
                assert (
                    len(traj_step.keys()) == 1
                ), f"Expected only one key in 0th traj step, but got: {traj_step.keys()}"
                assert "obs" in traj_step, f"Expected 'obs' key in 0th traj step, but got: {traj_step.keys()}"
                continue

            # Compose frame to add to dataset
            frame = {
                "action": traj_step["action"],
                "next.reward": th.tensor([traj_step["reward"]]),
                "next.terminated": th.tensor([traj_step["terminated"]]),
                "next.truncated": th.tensor([traj_step["truncated"]]),
                **traj_data[frame_idx - 1]["obs"],
            }
            frame["task"] = self.task_name

            self.dataset.add_frame(frame=frame)

        self.dataset.save_episode()

    def flush_current_file(self) -> None:
        # Does nothing currently
        pass

    def close_dataset(self) -> None:
        self.dataset.finalize()


class LeRobotPlaybackWrapper(DataPlaybackWrapper, LeRobotDataWrapper):
    """
    An OmniGibson environment wrapper for playing back data and collecting observations to be stored in LeRobotV3 format

    NOTE: This assumes a HDF5CollectionWrapper environment has been used to collect data!
    """

    def __init__(
        self,
        env: Environment,
        input_path: str,
        output_path: str,
        n_render_iterations: int = 1,
        overwrite: bool = True,
        only_successes: bool = False,
        flush_every_n_traj: int = 10,
        flush_every_n_steps: int = 0,
        full_scene_file: str | None = None,
        load_room_instances: list[str] | None = None,
        include_robot_control: bool = True,
        include_contacts: bool = True,
        root_dir: str = HF_LEROBOT_HOME,
        robot_type: str | None = None,
        task_name: str | None = None,
    ):
        """
        Args:
            env (Environment): The environment to wrap
            input_path (str): path to input hdf5 collected data file
            output_path (str): path to the output lerobot dataset. This value is synonymous with lerobot's
                @repo_id key, and should specify the name of the repo for saving the dataset, e.g. <username>/<dataset_name>
            n_render_iterations (int): Number of rendering iterations to use when loading each stored frame from the
                recorded data
            overwrite (bool): If set, will overwrite any pre-existing data found at @output_path.
                Otherwise, will load the data and append to it
            only_successes (bool): Whether to only save successful episodes
            flush_every_n_traj (int): How often to flush (write) current data to file across episodes
            flush_every_n_steps (int): How often to flush (write) current data to file within an episode.
                This is useful when collecting very long trajectories that may have a large memory footprint before writing to disk.
                If this is greater than 0, flush_every_n_traj must be set to 1.
            full_scene_file (None or str): If specified, the full scene file to use for playback. During data collection,
                the scene file stored may be partial, and this will be used to fill in the missing scene objects from the
                full scene file.
            load_room_instances (None or str): If specified, the room instances to load for playback.
            include_robot_control (bool): Whether or not to include robot control. If False, will disable all joint control.
            include_contacts (bool): Whether or not to include (enable) contacts in the sim. If False, will set all objects to be visual_only
            root_dir (str): Root directory to store output dataset files
            robot_type (None or str): Name of the robot within this dataset. If not specified, will be inferred
                from environment
            task_name (None or str): If specified, task that will be recorded in LeRobot dataset. If not specified,
                will try to automatically infer if the wrapped environment is a BehaviorTask
        """
        # Run super
        super().__init__(
            env=env,
            input_path=input_path,
            output_path=output_path,
            n_render_iterations=n_render_iterations,
            overwrite=overwrite,
            only_successes=only_successes,
            flush_every_n_traj=flush_every_n_traj,
            flush_every_n_steps=flush_every_n_steps,
            full_scene_file=full_scene_file,
            load_room_instances=load_room_instances,
            include_robot_control=include_robot_control,
            include_contacts=include_contacts,
            root_dir=root_dir,
            robot_type=robot_type,
            task_name=task_name,
        )

    def flush_partial_traj(self, step_idx: int, total_steps: int, step_data: dict) -> None:
        """
        Flush the current trajectory data to the LeRobot dataset. This is used when flush_every_n_steps
        is greater than 0 to incrementally write trajectory data to disk during an episode.

        With streaming encoding enabled, video data is written to disk in real-time via encoder threads.
        This method adds frames to the dataset, then resets the trajectory history to free memory.

        Args:
            step_idx (int): The index of the current step in the overall trajectory.
            total_steps (int): The total number of steps in the full trajectory.
            step_data (dict): The data for one step, useful for allocating trajectory data.
        """
        log.info(f"Flushing partial trajectory at step {self.current_episode_step_count}...")
        assert self.flush_every_n_steps > 0, "flush_every_n_steps must be greater than 0 to flush partial trajectory"
        assert (
            len(self.current_traj_history) > 0
        ), "Expected non-empty trajectory history when flushing partial trajectory"
        # Add frames to the LeRobot dataset incrementally
        # Skip the first step (only has obs from reset, no action/reward)
        for frame_idx in range(1, len(self.current_traj_history)):
            traj_step = self.current_traj_history[frame_idx]

            # Compose frame to add to dataset (same format as process_traj_to_dataset)
            frame = {
                "action": traj_step["action"],
                "next.reward": th.tensor([traj_step["reward"]]),
                "next.terminated": th.tensor([traj_step["terminated"]]),
                "next.truncated": th.tensor([traj_step["truncated"]]),
                **self.current_traj_history[frame_idx - 1]["obs"],
            }
            if self.task_name:
                frame["task"] = self.task_name

            self.dataset.add_frame(frame=frame)

        # Keep the last observation for pairing with next segment's first action
        # This is needed because obs[t] pairs with action[t+1], and after reset we need
        # the previous observation to pair with the new action
        last_step = self.current_traj_history[-1]
        assert (
            "obs" in last_step
        ), f"Expected 'obs' key in last step of trajectory history to keep for next segment, but got: {last_step.keys()}"
        self.current_traj_history = [{"obs": last_step["obs"]}]
