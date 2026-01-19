from omnigibson.envs import EnvironmentWrapper, Environment
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.learning.utils.eval_utils import ROBOT_CAMERA_NAMES, HEAD_RESOLUTION, WRIST_RESOLUTION
from omnigibson.sensors.vision_sensor import VisionSensor
from omnigibson.learning.utils.eval_utils import flatten_obs_dict
import numpy as np
import torch as th
import omnigibson.utils.transform_utils as T
from typing import Tuple
import pickle


# Create module logger
logger = create_module_logger(__name__)


class TheNorthStarWrapper(EnvironmentWrapper):
    """
    Args:
        env (og.Environment): The environment to wrap.
    """

    def step(self, robot_action: th.Tensor, n_render_iterations: int) -> Tuple[dict, float, bool, bool, dict]:
        if self.action_buffer is None or self.current_action_idx >= self.horizon:
            # We just resetted or exhausted the previous action buffer
            self.obs_buffer = []
            self.horizon = robot_action.shape[0]
            self.action_buffer = robot_action
            self.current_action_idx = 0
        current_action = self.action_buffer[self.current_action_idx]

        obs, _, terminated, truncated, info = self.env.step(current_action, n_render_iterations=n_render_iterations)

        # process obs
        if self.current_action_idx + 1 == self.horizon:
            # We exhausted the action buffer, need to get new action from policy
            # activate for nav module, inactivate for efficiency
            obs["past_obs"] = self.obs_buffer
            obs["robot_rel_lin_vel"] = self.env.robots[0].get_relative_linear_velocity()
            obs["robot_rel_ang_vel"] = self.env.robots[0].get_relative_angular_velocity()
            obs["seg_id_map"] = pickle.dumps(VisionSensor.INSTANCE_ID_REGISTRY)
            obs["need_new_action"] = True
        else:
            # store obs for future policy input
            pobs = self.process_obs_for_policy(obs)
            for topic in [
                "robot_r1::robot_r1:left_realsense_link:Camera:0::seg_semantic",
                "robot_r1::robot_r1:left_realsense_link:Camera:0::seg_instance_id",
                "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
                "robot_r1::robot_r1:left_realsense_link:Camera:0::depth_linear",
                "robot_r1::robot_r1:right_realsense_link:Camera:0::seg_semantic",
                "robot_r1::robot_r1:right_realsense_link:Camera:0::seg_instance_id",
                "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
                "robot_r1::robot_r1:right_realsense_link:Camera:0::depth_linear",
                # 'robot_r1::robot_r1:zed_link:Camera:0::seg_semantic', 'robot_r1::robot_r1:zed_link:Camera:0::seg_instance_id', 'robot_r1::robot_r1:zed_link:Camera:0::rgb', 'robot_r1::robot_r1:zed_link:Camera:0::depth_linear', 'robot_r1::proprio', 'robot_r1::cam_rel_poses' \
            ]:
                pobs.pop(topic)
            if (self.current_action_idx % (self.horizon / 2)) != (self.horizon / 2 - 1):
                for topic_head_imgs in [
                    "robot_r1::robot_r1:zed_link:Camera:0::seg_semantic",
                    "robot_r1::robot_r1:zed_link:Camera:0::seg_instance_id",
                    "robot_r1::robot_r1:zed_link:Camera:0::rgb",
                    "robot_r1::robot_r1:zed_link:Camera:0::depth_linear",
                ]:
                    pobs.pop(topic_head_imgs)
            self.obs_buffer.append(pobs)
            obs = {"need_new_action": False}

        # Increment action index
        self.current_action_idx += 1

        return obs, _, terminated, truncated, info

    def process_obs_for_policy(self, obs: dict) -> dict:
        """
        Preprocess the observation dictionary before passing it to the policy.
        Args:
            obs (dict): The observation dictionary to preprocess.

        Returns:
            dict: The preprocessed observation dictionary.
        """
        obs = flatten_obs_dict(obs)
        base_pose = self.env.robots[0].get_position_orientation()
        cam_rel_poses = []
        # The first time we query for camera parameters, it will return all zeros
        # For this case, we use camera.get_position_orientation() instead.
        # The reason we are not using camera.get_position_orientation() by defualt is because it will always return the most recent camera poses
        # However, since og render is somewhat "async", it takes >= 3 render calls per step to actually get the up-to-date camera renderings
        # Since we are using n_render_iterations=1 for speed concern, we need the correct corresponding camera poses instead of the most update-to-date one.
        # Thus, we use camera parameters which are guaranteed to be in sync with the visual observations.
        for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
            camera = self.env.robots[0].sensors[camera_name.split("::")[1]]
            direct_cam_pose = camera.camera_parameters["cameraViewTransform"]
            if np.allclose(direct_cam_pose, np.zeros(16)):
                cam_rel_poses.append(
                    th.cat(T.relative_pose_transform(*(camera.get_position_orientation()), *base_pose))
                )
            else:
                cam_pose = T.mat2pose(th.tensor(np.linalg.inv(np.reshape(direct_cam_pose, [4, 4]).T), dtype=th.float32))
                cam_rel_poses.append(th.cat(T.relative_pose_transform(*cam_pose, *base_pose)))
        obs["robot_r1::cam_rel_poses"] = th.cat(cam_rel_poses, axis=-1)
        obs["seg_id_map"] = pickle.dumps(VisionSensor.INSTANCE_ID_REGISTRY)
        obs["robot_rel_lin_vel"] = self.env.robots[0].get_relative_linear_velocity()
        obs["robot_rel_ang_vel"] = self.env.robots[0].get_relative_angular_velocity()
        return obs

    def __init__(self, env: Environment):
        super().__init__(env=env)
        self.cfg = env.config
        # Note that from eval.py we only set rgb modality, here we include more (depth + seg_instance_id)
        # Here, we change the camera resolution and head camera aperture to match the one we used in data collection
        robot = env.robots[0]
        # Update robot sensors:
        for camera_id, camera_name in ROBOT_CAMERA_NAMES["R1Pro"].items():
            sensor_name = camera_name.split("::")[1]
            if camera_id == "head":
                robot.sensors[sensor_name].horizontal_aperture = 40.0
                robot.sensors[sensor_name].image_height = HEAD_RESOLUTION[0]
                robot.sensors[sensor_name].image_width = HEAD_RESOLUTION[1]
            else:
                robot.sensors[sensor_name].image_height = WRIST_RESOLUTION[0]
                robot.sensors[sensor_name].image_width = WRIST_RESOLUTION[1]
            # add depth and segmentation
            robot.sensors[sensor_name].add_modality("depth_linear")
            robot.sensors[sensor_name].add_modality("seg_semantic")
            robot.sensors[sensor_name].add_modality("seg_instance_id")
        # reload observation space
        env.load_observation_space()
        logger.info("Reloaded observation space!")

        # Initialize action buffer and related variables
        self.action_buffer = None
        self.horizon = None
        self.current_action_idx = None
        self.obs_buffer = None

    def reset(self):
        self.action_buffer = None
        self.horizon = None
        self.current_action_idx = None
        self.obs_buffer = None
        return super().reset()


WRAPPER_CLASS = TheNorthStarWrapper
