"""OmniGibson Gymnasium environment wrapper for LeRobot policy evaluation.

This module provides a Gymnasium-compatible environment that wraps OmniGibson
for use with LeRobot's evaluation infrastructure.
"""

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import yaml

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.objects import DatasetObject
from omnigibson.object_states import OnTop
from omnigibson.sensors.vision_sensor import VisionSensor

logger = logging.getLogger(__name__)


@dataclass
class EpisodeMetadata:
    """Metadata for a single evaluation episode.

    This matches the format produced by data collection.
    """
    dataset_name: str
    scene_name: str
    source_support_name: str
    target_support_name: str
    robot_start_x_y_z_theta: list[float]
    spawned_target_object: bool
    target_object_name: str
    spawned_target_object_position: list[float]
    spawned_target_object_orientation: list[float]
    spawned_target_object_category: str
    spawned_target_object_model: str
    spawned_target_object_dataset_name: str = "behavior-1k-assets"

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodeMetadata":
        # Filter to only include fields that exist in the dataclass
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in field_names}
        return cls(**filtered)


@dataclass
class EnvConfig:
    """Configuration for OmniGibson evaluation environment."""
    scene_model: str = "Rs_int"
    dataset_name: str = "behavior-1k-assets"
    max_steps: int = 500
    fps: int = 30
    image_height: int = 224
    image_width: int = 224
    include_depth: bool = True
    include_segmentation: bool = True
    robot_type: str = "Stretch"
    not_load_object_categories: list[str] = field(
        default_factory=lambda: ["ceilings", "armchair", "ottoman"]
    )


class OmniGibsonEvalEnv(gym.Env):
    """Gymnasium environment wrapper for OmniGibson evaluation.

    Provides 4 camera observations:
    - observation.images.head: Head camera RGB (224x224x3)
    - observation.images.wrist: Wrist camera RGB (224x224x3)
    - observation.images.head_seg_depth: Head camera seg+depth (224x224x3)
        - R: target object segmentation (0 or 255)
        - G: support object segmentation (0 or 255)
        - B: normalized depth (0-255)
    - observation.images.wrist_seg_depth: Wrist camera seg+depth (224x224x3)
    - observation.state: Robot proprioception (joint positions)

    Action space:
    - base: [lin_vel, ang_vel] (2D)
    - arm: joint position deltas (depends on robot)
    - gripper: binary open/close
    - mode: navigation (0) or manipulation (1) flag
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        config: EnvConfig | None = None,
        episode_metadata: EpisodeMetadata | None = None,
        render_mode: str | None = "rgb_array",
    ):
        super().__init__()

        self.config = config or EnvConfig()
        self.episode_metadata = episode_metadata
        self.render_mode = render_mode

        self._env = None
        self._scene = None
        self._robot = None
        self._target_object = None
        self._source_support = None
        self._target_support = None

        self._current_step = 0
        self._initialized = False

        self._image_keys = []
        self._seg_depth_keys = []

    def _max_episode_steps(self) -> int:
        """Return max episode steps (used by LeRobot)."""
        return self.config.max_steps

    def _create_environment(self):
        """Create OmniGibson environment."""
        og_config = self._build_og_config()
        self._env = og.Environment(configs=og_config)
        self._scene = self._env.scene
        self._robot = self._env.robots[0]

        # Load segmentation map
        self._scene._seg_map.load_map()
        logger.info("Loaded segmentation map: %d rooms",
                   len(self._scene._seg_map.room_ins_name_to_ins_id))

        # Apply floor friction
        self._apply_floor_friction()

        # Increase robot base mass for stability
        with og.sim.stopped():
            original_mass = self._robot.base_footprint_link.mass
            self._robot.base_footprint_link.mass = original_mass * 2.0

        # Run initial simulation steps
        for _ in range(30):
            og.sim.step()

        self._setup_spaces()
        self._initialized = True

    def _build_og_config(self) -> dict:
        """Build OmniGibson configuration dictionary."""
        config = {
            "env": {
                "action_frequency": self.config.fps,
                "physics_frequency": 120,
                "device": None,
                "automatic_reset": False,
                "flatten_action_space": False,
                "flatten_obs_space": False,
                "use_external_obs": False,
                "external_sensors": None,
            },
            "render": {
                "viewer_width": 1280,
                "viewer_height": 720,
            },
            "scene": {
                "type": "InteractiveTraversableScene",
                "scene_model": self.episode_metadata.scene_name,
                "dataset_name": self.episode_metadata.dataset_name,
                "trav_map_resolution": 0.1,
                "default_erosion_radius": 0.01,
                "trav_map_with_objects": True,
                "num_waypoints": 50,
                "waypoint_resolution": 0.1,
                "load_object_categories": None,
                "not_load_object_categories": self.config.not_load_object_categories,
                "load_room_types": None,
                "load_room_instances": None,
                "load_task_relevant_only": False,
                "seg_map_resolution": 0.1,
                "scene_source": "OG",
                "include_robots": False,
            },
            "robots": [
                {
                    "type": self.config.robot_type,
                    "obs_modalities": ["rgb", "seg_instance", "depth_linear"],
                    "include_sensor_names": None,
                    "exclude_sensor_names": None,
                    "scale": 1.0,
                    "self_collisions": True,
                    "action_normalize": False,
                    "action_type": "continuous",
                    "grasping_mode": "sticky",
                    "sensor_config": {
                        "VisionSensor": {
                            "sensor_kwargs": {
                                "image_height": self.config.image_height,
                                "image_width": self.config.image_width,
                            }
                        }
                    },
                    "controller_config": {
                        "base": {
                            "name": "DifferentialDriveController",
                            "command_input_limits": None,
                        },
                        "camera": {
                            "name": "NullJointController",
                            "motor_type": "position",
                        },
                        "arm_0": {
                            "name": "JointController",
                            "motor_type": "position",
                            "command_input_limits": None,
                            "use_delta_commands": True,
                            "use_impedances": False,
                        },
                        "gripper_0": {
                            "name": "MultiFingerGripperController",
                            "motor_type": "position",
                            "mode": "binary",
                            "closed_qpos": [-0.6, -0.6],
                            "open_qpos": [0.6, 0.6],
                        },
                    },
                }
            ],
            "objects": [],
            "task": {"type": "DummyTask"},
        }

        # Handle SPOC scenes
        if self.episode_metadata.dataset_name == "spoc":
            config["scene"]["use_floor_plane"] = True
            config["scene"]["floor_plane_visible"] = True

        # Handle non-standard dataset names
        if self.episode_metadata.dataset_name != "behavior-1k-assets":
            config["scene"]["scene_instance"] = f"{self.episode_metadata.scene_name}_best"

        return config

    def _apply_floor_friction(self):
        """Apply high friction to floor surfaces."""
        for obj in self._scene.objects:
            if getattr(obj, 'category', '') == 'floors':
                for link in obj.links.values():
                    for mesh in link.collision_meshes.values():
                        mat_name = f"{obj.name}_floor_physics_mat"
                        physics_mat = lazy.isaacsim.core.api.materials.physics_material.PhysicsMaterial(
                            prim_path=f"{obj.prim_path}/Looks/{mat_name}",
                            name=mat_name,
                            static_friction=1.0,
                            dynamic_friction=1.0,
                            restitution=0.0,
                        )
                        mesh.apply_physics_material(physics_mat)

    def _setup_spaces(self):
        """Setup observation and action spaces."""
        # Get a sample observation to infer structure
        self._robot.reset()
        for _ in range(5):
            og.sim.step()

        sample_obs, _ = self._env.get_obs()
        self._infer_observation_structure(sample_obs)

        # Observation space
        obs_spaces = {}
        state_dim = len(self._extract_proprioception())
        obs_spaces["observation.state"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        img_shape = (self.config.image_height, self.config.image_width, 3)
        for key in self._image_keys:
            lerobot_key = self._to_lerobot_key(key)
            obs_spaces[lerobot_key] = gym.spaces.Box(
                low=0, high=255, shape=img_shape, dtype=np.uint8
            )
        for key in self._seg_depth_keys:
            lerobot_key = self._to_lerobot_key(key)
            obs_spaces[lerobot_key] = gym.spaces.Box(
                low=0, high=255, shape=img_shape, dtype=np.uint8
            )

        self.observation_space = gym.spaces.Dict(obs_spaces)

        # Action space (flat array with mode flag at end)
        lows, highs = [], []
        self._action_keys = []
        for key, space in self._env.action_space.spaces.items():
            lows.append(space.low.flatten())
            highs.append(space.high.flatten())
            self._action_keys.append((key, space.shape))

        # Add mode flag dimension
        lows.append(np.array([0.0]))
        highs.append(np.array([1.0]))

        self._og_action_dim = sum(np.prod(shape) for _, shape in self._action_keys)
        self.action_space = gym.spaces.Box(
            low=np.concatenate(lows),
            high=np.concatenate(highs),
            dtype=np.float32
        )

    def _infer_observation_structure(self, sample_obs: dict):
        """Extract camera keys from observation."""
        self._image_keys = []
        self._seg_depth_keys = []
        self._og_to_lerobot_key = {}

        robot_key = list(sample_obs.keys())[0]
        robot_obs = sample_obs[robot_key]

        for sensor_key, sensor_data in robot_obs.items():
            camera_name = self._simplify_camera_name(sensor_key)

            if "rgb" in sensor_data:
                full_key = f"{robot_key}/{sensor_key}/rgb"
                lerobot_key = f"observation.images.{camera_name}"
                self._image_keys.append(full_key)
                self._og_to_lerobot_key[full_key] = lerobot_key

            if self.config.include_segmentation and "seg_instance" in sensor_data:
                full_key = f"{robot_key}/{sensor_key}/seg_depth"
                lerobot_key = f"observation.images.{camera_name}_seg_depth"
                self._seg_depth_keys.append(full_key)
                self._og_to_lerobot_key[full_key] = lerobot_key

        logger.info("Image keys: %s", [self._og_to_lerobot_key[k] for k in self._image_keys])
        logger.info("Seg+Depth keys: %s", [self._og_to_lerobot_key[k] for k in self._seg_depth_keys])

    def _simplify_camera_name(self, sensor_key: str) -> str:
        """Convert OmniGibson sensor key to simple camera name."""
        if "eyes" in sensor_key:
            return "head"
        elif "eef_link" in sensor_key:
            return "wrist"
        else:
            parts = sensor_key.split(":")
            if len(parts) >= 2:
                return parts[1].replace("_", "")
            return "camera"

    def _to_lerobot_key(self, og_key: str) -> str:
        """Convert OmniGibson key to LeRobot format."""
        return self._og_to_lerobot_key.get(og_key, og_key)

    def _extract_proprioception(self) -> np.ndarray:
        """Extract robot joint positions."""
        joint_pos = self._robot._get_proprioception_dict()['joint_qpos']
        return joint_pos.cpu().numpy().astype(np.float32)

    def _convert_observation(self, og_obs: dict) -> dict:
        """Convert OmniGibson observation to LeRobot format."""
        lerobot_obs = {}

        lerobot_obs["observation.state"] = self._extract_proprioception()

        robot_key = list(og_obs.keys())[0]
        robot_obs = og_obs[robot_key]

        # RGB images
        for key in self._image_keys:
            parts = key.split("/")
            sensor_key = parts[1]
            img = robot_obs[sensor_key]["rgb"]
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            if img.shape[2] == 4:
                img = img[:, :, :3]
            lerobot_obs[self._to_lerobot_key(key)] = img.astype(np.uint8)

        # Seg+depth images
        for key in self._seg_depth_keys:
            parts = key.split("/")
            sensor_key = parts[1]

            seg_instance = robot_obs[sensor_key].get("seg_instance")
            depth = robot_obs[sensor_key].get("depth_linear")

            if seg_instance is None:
                h, w = self.config.image_height, self.config.image_width
                lerobot_obs[self._to_lerobot_key(key)] = np.zeros((h, w, 3), dtype=np.uint8)
                continue

            if isinstance(seg_instance, torch.Tensor):
                seg_instance = seg_instance.cpu().numpy()

            h, w = seg_instance.shape[:2]
            seg_depth_img = np.zeros((h, w, 3), dtype=np.uint8)

            instance_registry = VisionSensor.INSTANCE_REGISTRY

            # R channel: target object segmentation
            if self._target_object is not None:
                target_name = self._target_object.name
                for instance_id, obj_name in instance_registry.items():
                    if target_name in str(obj_name):
                        mask = seg_instance == instance_id
                        seg_depth_img[:, :, 0][mask] = 255

            # G channel: support object segmentation (current target support)
            if self._target_support is not None:
                support_name = self._target_support.name
                for instance_id, obj_name in instance_registry.items():
                    if support_name in str(obj_name):
                        mask = seg_instance == instance_id
                        seg_depth_img[:, :, 1][mask] = 255

            # B channel: depth (normalized to 0-255)
            if depth is not None:
                if isinstance(depth, torch.Tensor):
                    depth = depth.cpu().numpy()
                depth_normalized = np.clip(depth, 0, 10.0) / 10.0 * 255
                seg_depth_img[:, :, 2] = depth_normalized.astype(np.uint8)

            lerobot_obs[self._to_lerobot_key(key)] = seg_depth_img

        return lerobot_obs

    def _convert_action_to_og(self, action: np.ndarray) -> dict:
        """Convert flat action array to OmniGibson format."""
        og_action_flat = action[:self._og_action_dim]

        og_action = {}
        idx = 0
        for key, shape in self._action_keys:
            size = int(np.prod(shape))
            og_action[key] = og_action_flat[idx:idx + size].reshape(shape)
            idx += size
        return og_action

    def set_episode_metadata(self, metadata: EpisodeMetadata):
        """Set episode metadata for evaluation setup."""
        self.episode_metadata = metadata

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None
    ) -> tuple[dict, dict]:
        """Reset environment for a new episode."""
        super().reset(seed=seed)

        if not self._initialized:
            self._create_environment()

        self._current_step = 0

        # Clear any grasped objects
        for arm in self._robot.arm_names:
            if self._robot._ag_obj_in_hand.get(arm) is not None:
                try:
                    self._robot.release_grasp_immediately(arm=arm)
                except Exception:
                    pass

        # Reset robot and scene
        self._robot.reset()
        self._scene.reset()

        for _ in range(30):
            og.sim.step()

        # Setup episode from metadata if provided
        if self.episode_metadata is not None:
            self._setup_episode_from_metadata()
        else:
            # Default: start robot at origin
            self._robot.set_position_orientation(
                position=[0, 0, 0],
                orientation=[0, 0, 0, 1]
            )

        for _ in range(10):
            og.sim.step()

        og_obs, _ = self._env.get_obs()
        obs = self._convert_observation(og_obs)
        info = {"episode_metadata": self.episode_metadata}

        return obs, info

    def _setup_episode_from_metadata(self):
        """Setup episode using metadata."""
        meta = self.episode_metadata

        # Find source and target supports
        self._source_support = None
        self._target_support = None
        for obj in self._scene.objects:
            if obj.name == meta.source_support_name:
                self._source_support = obj
            if obj.name == meta.target_support_name:
                self._target_support = obj

        if self._source_support is None:
            logger.warning("Source support '%s' not found in scene", meta.source_support_name)
        if self._target_support is None:
            logger.warning("Target support '%s' not found in scene", meta.target_support_name)

        # Position robot
        x, y, z, theta = meta.robot_start_x_y_z_theta
        quat = [0, 0, math.sin(theta / 2), math.cos(theta / 2)]
        self._robot.set_position_orientation(position=[x, y, z], orientation=quat)
        self._robot.keep_still()

        for _ in range(10):
            og.sim.step()

        # Setup target object
        if meta.spawned_target_object:
            self._target_object = self._spawn_target_object(meta)
        else:
            self._target_object = self._find_object_by_name(meta.target_object_name)

        if self._target_object is None:
            logger.warning("Target object '%s' not found/spawned", meta.target_object_name)

        # Remove all other loose objects
        for i, obj in enumerate(self._scene.objects):
            if not obj.fixed_base and obj not in (
                self._target_object, self._source_support,
                self._target_support, self._robot
            ):
                obj.set_position_orientation(position=torch.tensor([100 + i, 0, 10.0]))

    def _spawn_target_object(self, meta: EpisodeMetadata) -> DatasetObject | None:
        """Spawn target object from metadata."""
        try:
            obj = DatasetObject(
                name=meta.target_object_name,
                category=meta.spawned_target_object_category,
                model=meta.spawned_target_object_model,
                dataset_name=meta.spawned_target_object_dataset_name,
            )
            self._scene.add_object(obj)

            for _ in range(5):
                og.sim.step()

            # Set position and orientation
            obj.set_position_orientation(
                position=torch.tensor(meta.spawned_target_object_position),
                orientation=torch.tensor(meta.spawned_target_object_orientation),
            )

            for _ in range(10):
                og.sim.step()

            return obj
        except Exception as e:
            logger.error("Failed to spawn target object: %s", e)
            return None

    def _find_object_by_name(self, name: str) -> Any | None:
        """Find object in scene by name."""
        for obj in self._scene.objects:
            if obj.name == name:
                return obj
        return None

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Execute one environment step."""
        og_action = self._convert_action_to_og(action)
        og_obs, reward, terminated, truncated, info = self._env.step(og_action)

        self._current_step += 1

        # Check step limit
        if self._current_step >= self.config.max_steps:
            truncated = True

        # Compute success (object placed on target support)
        success = self._check_success()
        info["is_success"] = success

        if success:
            terminated = True
            reward = 1.0
        else:
            reward = 0.0

        obs = self._convert_observation(og_obs)

        return obs, reward, terminated, truncated, info

    def _check_success(self) -> bool:
        """Check if task is successful (object on target support)."""
        if self._target_object is None or self._target_support is None:
            return False

        # Check if object is no longer grasped
        for arm in self._robot.arm_names:
            if self._robot._ag_obj_in_hand.get(arm) == self._target_object:
                return False  # Still holding object

        # Check OnTop state
        if OnTop in self._target_object.states:
            try:
                return self._target_object.states[OnTop].get_value(self._target_support)
            except Exception:
                pass

        # Fallback: check position relative to support
        try:
            obj_pos = self._target_object.get_position_orientation()[0]
            support_aabb = self._target_support.aabb

            x_in = support_aabb[0][0] <= obj_pos[0] <= support_aabb[1][0]
            y_in = support_aabb[0][1] <= obj_pos[1] <= support_aabb[1][1]
            z_above = obj_pos[2] >= support_aabb[1][2] - 0.1
            z_below = obj_pos[2] <= support_aabb[1][2] + 0.5

            return x_in and y_in and z_above and z_below
        except Exception:
            return False

    def render(self) -> np.ndarray | None:
        """Render the environment."""
        if self.render_mode == "rgb_array":
            # Return head camera image
            og_obs, _ = self._env.get_obs()
            robot_key = list(og_obs.keys())[0]
            for sensor_key, sensor_data in og_obs[robot_key].items():
                if "eyes" in sensor_key and "rgb" in sensor_data:
                    img = sensor_data["rgb"]
                    if isinstance(img, torch.Tensor):
                        img = img.cpu().numpy()
                    if img.shape[2] == 4:
                        img = img[:, :, :3]
                    return img.astype(np.uint8)
        return None

    def close(self):
        """Clean up environment."""
        if self._env is not None:
            try:
                og.shutdown()
            except Exception:
                pass
            self._env = None
            self._initialized = False

    @property
    def unwrapped(self):
        """Return unwrapped environment."""
        return self

    @property
    def task(self) -> str:
        """Return task description."""
        return f"pick_and_place_{self.episode_metadata.scene_name}"
