"""
OmniGibson to LeRobot Dataset Wrapper

Collects random action demonstrations from OmniGibson environments
and saves them in LeRobot dataset format to a local folder.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from omnigibson.sensors.vision_sensor import VisionSensor
from vid2scene_policy.data_collection.lerobot_datasets.datasets.lerobot_dataset import LeRobotDataset
from vid2scene_policy.data_collection.lerobot_datasets.utils.constants import ACTION, DONE, OBS_IMAGES, OBS_STATE, REWARD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OmniGibsonLeRobotConfig:
    """Configuration for OmniGibson to LeRobot wrapper."""

    repo_id: str = "omnigibson_dataset"
    root: str | Path = "./lerobot_datasets"
    task_description: str = "omnigibson_task"
    fps: int = 30
    num_episodes: int = 10
    max_steps: int = 100
    use_videos: bool = True
    image_writer_threads: int = 4
    include_depth: bool = True
    include_segmentation: bool = True


class OmniGibsonLeRobotWrapper(gym.Wrapper):
    """
    Wrapper that converts OmniGibson observations to LeRobot format
    and enables dataset recording.
    """
    
    def __init__(self, env: gym.Env, config: OmniGibsonLeRobotConfig | None = None):
        super().__init__(env)
        
        self.config = config or OmniGibsonLeRobotConfig()
        self.dataset: LeRobotDataset | None = None
        self.is_recording = False
        
        self.current_episode = 0
        self.current_step = 0
        self.episode_start_time = None
        
        self._og_obs_structure = {}
        self._image_keys = []
        self._depth_keys = []
        self._seg_keys = []
        self._obs_inferred = False
        self._lerobot_observation_space = None
        self._lerobot_action_space = None
        self._action_keys = None
        self._target_object = None
        self._support_object = None
        
    def _infer_observation_structure(self, sample_obs: dict):
        """Extract image, depth, and segmentation keys from OmniGibson robot camera sensors.

        Creates two image types per camera:
        - rgb: Standard RGB image
        - seg_depth: Combined R=target_seg, G=support_seg, B=depth (all as uint8)

        Uses simplified, consistent camera names for LeRobot compatibility:
        - eyes -> head
        - eef_link -> wrist
        """
        self._og_obs_structure = {}
        self._image_keys = []
        self._seg_depth_keys = []
        self._og_to_lerobot_key = {}  # Maps OG keys to simplified LeRobot keys

        robot_key = list(sample_obs.keys())[0]
        robot_obs = sample_obs[robot_key]

        for sensor_key, sensor_data in robot_obs.items():
            # Extract camera name and simplify
            camera_name = self._simplify_camera_name(sensor_key)

            if "rgb" in sensor_data:
                full_key = f"{robot_key}/{sensor_key}/rgb"
                lerobot_key = f"{OBS_IMAGES}.{camera_name}"
                img = sensor_data["rgb"]
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                shape = img.shape
                if len(shape) == 3 and shape[2] == 4:
                    shape = (shape[0], shape[1], 3)
                self._image_keys.append(full_key)
                self._og_obs_structure[full_key] = {"shape": shape, "dtype": np.uint8}
                self._og_to_lerobot_key[full_key] = lerobot_key

            # Combined seg_depth image: R=target, G=support, B=depth
            if self.config.include_segmentation and "seg_instance" in sensor_data:
                full_key = f"{robot_key}/{sensor_key}/seg_depth"
                lerobot_key = f"{OBS_IMAGES}.{camera_name}_seg_depth"
                seg = sensor_data["seg_instance"]
                if isinstance(seg, torch.Tensor):
                    seg = seg.cpu().numpy()
                shape = seg.shape[:2] + (3,)  # RGB format
                self._seg_depth_keys.append(full_key)
                self._og_obs_structure[full_key] = {"shape": shape, "dtype": np.uint8}
                self._og_to_lerobot_key[full_key] = lerobot_key

        logger.info(f"Image keys: {[self._og_to_lerobot_key[k] for k in self._image_keys]}")
        logger.info(f"Seg+Depth keys: {[self._og_to_lerobot_key[k] for k in self._seg_depth_keys]}")
        self._obs_inferred = True
        self._setup_lerobot_spaces()

    def _simplify_camera_name(self, sensor_key: str) -> str:
        """Convert OmniGibson sensor key to simple camera name."""
        # sensor_key format: "robot_xxx:eyes:Camera:0" or "robot_xxx:eef_link:Camera:0"
        if "eyes" in sensor_key:
            return "head"
        elif "eef_link" in sensor_key:
            return "wrist"
        else:
            # Fallback: extract middle part
            parts = sensor_key.split(":")
            if len(parts) >= 2:
                return parts[1].replace("_", "")
            return "camera"

    
    def _setup_lerobot_spaces(self):
        """Setup observation and action spaces compatible with LeRobot."""
        state_dim = len(self._extract_robot_proprioception())

        obs_spaces = {
            OBS_STATE: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        }
        for key in self._image_keys:
            lerobot_key = self._to_lerobot_image_key(key)
            obs_spaces[lerobot_key] = gym.spaces.Box(
                low=0, high=255, shape=self._og_obs_structure[key]["shape"], dtype=np.uint8
            )
        for key in self._seg_depth_keys:
            lerobot_key = self._to_lerobot_image_key(key)
            obs_spaces[lerobot_key] = gym.spaces.Box(
                low=0, high=255, shape=self._og_obs_structure[key]["shape"], dtype=np.uint8
            )
        self._lerobot_observation_space = gym.spaces.Dict(obs_spaces)

        lows, highs = [], []
        self._action_keys = []
        for key, space in self.env.action_space.spaces.items():
            lows.append(space.low.flatten())
            highs.append(space.high.flatten())
            self._action_keys.append((key, space.shape))

        # Add extra dimension for mode flag (0=navigation, 1=manipulation)
        lows.append(np.array([0.0]))
        highs.append(np.array([1.0]))

        self._og_action_dim = sum(np.prod(shape) for _, shape in self._action_keys)
        self._lerobot_action_space = gym.spaces.Box(
            low=np.concatenate(lows),
            high=np.concatenate(highs),
            dtype=np.float32
        )
            
    def _to_lerobot_image_key(self, og_key: str) -> str:
        """Convert OmniGibson image key to LeRobot format using pre-computed mapping."""
        return self._og_to_lerobot_key.get(og_key, og_key)
    
    def _extract_robot_proprioception(self) -> np.ndarray:
        """Extract robot joint positions."""
        joint_pos = self.env.robots[0]._get_proprioception_dict()['joint_qpos']
        return joint_pos.cpu().numpy().astype(np.float32)

    def _convert_observation(self, og_obs: dict) -> dict:
        """Convert OmniGibson observation to LeRobot format.

        Creates:
        - RGB images from cameras
        - seg_depth images with R=target_seg, G=support_seg, B=depth (normalized)
        """
        lerobot_obs = {}

        lerobot_obs[OBS_STATE] = self._extract_robot_proprioception()

        robot_key = list(og_obs.keys())[0]
        robot_obs = og_obs[robot_key]

        for key in self._image_keys:
            parts = key.split("/")
            sensor_key = parts[1]
            img = robot_obs[sensor_key]["rgb"]
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            if img.shape[2] == 4:
                img = img[:, :, :3]
            lerobot_obs[self._to_lerobot_image_key(key)] = img.astype(np.uint8)

        for key in self._seg_depth_keys:
            parts = key.split("/")
            sensor_key = parts[1]

            seg_instance = robot_obs[sensor_key].get("seg_instance")
            depth = robot_obs[sensor_key].get("depth_linear")

            if seg_instance is None:
                shape = self._og_obs_structure[key]["shape"]
                lerobot_obs[self._to_lerobot_image_key(key)] = np.zeros(shape, dtype=np.uint8)
                continue

            if isinstance(seg_instance, torch.Tensor):
                seg_instance = seg_instance.cpu().numpy()

            h, w = seg_instance.shape[:2]
            seg_depth_img = np.zeros((h, w, 3), dtype=np.uint8)

            # Use VisionSensor.INSTANCE_REGISTRY to map IDs to object names
            instance_registry = VisionSensor.INSTANCE_REGISTRY

            # R channel: target object segmentation (0 or 255)
            if self._target_object is not None:
                target_name = self._target_object.name
                for instance_id, obj_name in instance_registry.items():
                    if target_name in str(obj_name):
                        mask = seg_instance == instance_id
                        seg_depth_img[:, :, 0][mask] = 255

            # G channel: support object segmentation (0 or 255)
            if self._support_object is not None:
                support_name = self._support_object.name
                for instance_id, obj_name in instance_registry.items():
                    if support_name in str(obj_name):
                        mask = seg_instance == instance_id
                        seg_depth_img[:, :, 1][mask] = 255

            # B channel: depth (normalized to 0-255)
            if depth is not None:
                if isinstance(depth, torch.Tensor):
                    depth = depth.cpu().numpy()
                # Normalize depth: clip to max 10m and scale to 0-255
                depth_normalized = np.clip(depth, 0, 10.0) / 10.0 * 255
                seg_depth_img[:, :, 2] = depth_normalized.astype(np.uint8)

            lerobot_obs[self._to_lerobot_image_key(key)] = seg_depth_img

        return lerobot_obs

    def set_target_objects(self, target_object, support_object):
        """Set the target and support objects for segmentation mask generation."""
        self._target_object = target_object
        self._support_object = support_object
    
    def _convert_action_to_og(self, action: np.ndarray) -> dict | np.ndarray:
        """Convert flat action array back to OmniGibson format.

        Strips the mode flag (last dimension) before passing to OmniGibson.
        """
        if self._action_keys is None:
            return action[:self._og_action_dim] if len(action) > self._og_action_dim else action

        # Strip mode flag (last dimension)
        og_action_flat = action[:self._og_action_dim]

        og_action = {}
        idx = 0
        for key, shape in self._action_keys:
            size = int(np.prod(shape))
            og_action[key] = og_action_flat[idx:idx + size].reshape(shape)
            idx += size
        return og_action
    
    def reset_env(self, **kwargs) -> tuple[dict, dict]:
        """Reset environment and return LeRobot-formatted observation."""
        og_obs, info = self.env.reset(**kwargs)
        
        if not self._obs_inferred:
            self._infer_observation_structure(og_obs)

        lerobot_obs = self._convert_observation(og_obs)
        
        self.current_step = 0
        self.episode_start_time = time.perf_counter()

        return lerobot_obs, info
    
    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Step environment with action and return LeRobot-formatted observation."""
        og_action = self._convert_action_to_og(action)
        og_obs, reward, terminated, truncated, info = self.env.step(og_action)
        lerobot_obs = self._convert_observation(og_obs)
        
        self.current_step += 1
        
        if self.config.max_steps and self.current_step >= self.config.max_steps:
            truncated = True
            
        return lerobot_obs, reward, terminated, truncated, info
    
    def get_features(self) -> dict:
        """Get LeRobot dataset features based on observation structure."""
        if not self._obs_inferred:
            raise RuntimeError("Call reset() first to infer observation structure")

        features = {}

        action_dim = self._lerobot_action_space.shape[0]
        features[ACTION] = {"dtype": "float32", "shape": (action_dim,), "names": None}

        if OBS_STATE in self._lerobot_observation_space.spaces:
            state_space = self._lerobot_observation_space.spaces[OBS_STATE]
            features[OBS_STATE] = {"dtype": "float32", "shape": state_space.shape, "names": None}

        for key in self._image_keys:
            lerobot_key = self._to_lerobot_image_key(key)
            info = self._og_obs_structure[key]
            features[lerobot_key] = {
                "dtype": "video" if self.config.use_videos else "image",
                "shape": info["shape"],
                "names": ["height", "width", "channels"] if len(info["shape"]) == 3 else None
            }

        # seg_depth: R=target_seg, G=support_seg, B=depth
        for key in self._seg_depth_keys:
            lerobot_key = self._to_lerobot_image_key(key)
            info = self._og_obs_structure[key]
            features[lerobot_key] = {
                "dtype": "video" if self.config.use_videos else "image",
                "shape": info["shape"],
                "names": ["height", "width", "channels"]
            }

        features[REWARD] = {"dtype": "float32", "shape": (1,), "names": None}
        features[DONE] = {"dtype": "bool", "shape": (1,), "names": None}

        return features
    
    def start_recording(self):
        """Initialize dataset and start recording to local folder."""
        if not self._obs_inferred:
            raise RuntimeError("Call reset() first to infer observation structure")

        local_path = Path(self.config.root) / self.config.repo_id

        # Remove existing dataset to allow fresh start
        if local_path.exists():
            import shutil
            shutil.rmtree(local_path)
            logger.info(f"Removed existing dataset at: {local_path}")

        features = self.get_features()

        logger.info(f"Creating dataset at: {local_path}")
        logger.info(f"Features: {list(features.keys())}")
        
        self.dataset = LeRobotDataset.create(
            repo_id=self.config.repo_id,
            fps=self.config.fps,
            features=features,
            root=local_path,
            use_videos=self.config.use_videos,
            image_writer_processes=0,
            image_writer_threads=self.config.image_writer_threads,
            vcodec="h264",  # Use h264 to preserve color channels (yuv444p support)
        )
        
        self.is_recording = True
        self.current_episode = 0
        
    def stop_recording(self):
        """Stop recording and finalize dataset."""
        if self.dataset is not None:
            self.dataset.finalize()
            logger.info(f"Recording stopped. Total episodes: {self.current_episode}")
            logger.info(f"Dataset saved to: {self.dataset.root}")
        self.is_recording = False
        
    def record_frame(self, observation: dict, action: np.ndarray, reward: float, done: bool):
        """Record a single frame to the dataset."""
        if not self.is_recording or self.dataset is None:
            return
            
        frame = {
            **observation,
            ACTION: action.astype(np.float32),
            REWARD: np.array([reward], dtype=np.float32),
            DONE: np.array([done], dtype=bool),
            "task": self.config.task_description,
        }
        self.dataset.add_frame(frame)
        
    def save_episode(self, episode_metadata: dict):
        """Save the current episode buffer to disk."""
        if not self.is_recording or self.dataset is None:
            return
            
        logger.info(f"Saving episode {self.current_episode} ({self.current_step} steps)")
        self.dataset.save_episode(episode_metadata=metadata)
        self.current_episode += 1

    @property
    def lerobot_observation_space(self) -> gym.spaces.Dict:
        return self._lerobot_observation_space
    
    @property
    def lerobot_action_space(self) -> gym.spaces.Box:
        return self._lerobot_action_space


def collect_demonstrations(
    env: gym.Env,
    repo_id: str = "omnigibson_demos",
    root: str = "./lerobot_datasets",
    task: str = "exploration",
    num_episodes: int = 10,
    max_steps: int = 100,
    fps: int = 30,
    policy=None,
):
    """
    Collect demonstrations from an OmniGibson environment using a policy.

    Args:
        env: OmniGibson environment
        repo_id: Dataset name (subfolder)
        root: Root directory for saving
        task: Task description
        num_episodes: Number of episodes to collect
        max_steps: Max steps per episode
        fps: Recording FPS
        policy: Policy to use for action generation. If None, uses random actions.

    Returns:
        Path to saved dataset
    """
    config = OmniGibsonLeRobotConfig(
        repo_id=repo_id,
        root=root,
        task_description=task,
        fps=fps,
        num_episodes=num_episodes,
        max_steps=max_steps,
    )
    
    wrapped_env = OmniGibsonLeRobotWrapper(env, config)
    
    # First reset to infer observation structure
    obs, _ = wrapped_env.reset_env()
    
    # Start recording
    wrapped_env.start_recording()
    
    try:
        for ep in range(num_episodes):
            logger.info(f"Episode {ep + 1}/{num_episodes}")
            obs, _ = wrapped_env.reset_env()

            for step in range(max_steps):
                if policy is not None:
                    action = policy.get_action(env.robots[0], obs)
                else:
                    action = wrapped_env.lerobot_action_space.sample()

                next_obs, reward, terminated, truncated, _ = wrapped_env.step(action)

                wrapped_env.record_frame(obs, action, reward, terminated or truncated)
                obs = next_obs

                if terminated or truncated:
                    break

            wrapped_env.save_episode()
            
    finally:
        wrapped_env.stop_recording()
        
    return wrapped_env.dataset.root


def collect_random_demonstrations(*args, **kwargs):
    """Backward compatibility wrapper for collect_demonstrations."""
    return collect_demonstrations(*args, **kwargs)
