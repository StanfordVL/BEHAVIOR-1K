import csv
import cv2
import json
import logging
import os
import sys
import traceback
from signal import SIGINT, signal
from typing import Any, List, Tuple

import gymnasium as gym
import numpy as np
import torch as th
from av.container import Container
from av.stream import Stream
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import omnigibson as og
import omnigibson.utils.transform_utils as T
from gello.utils.og_teleop_cfg import DISABLED_TRANSITION_RULES
from gello.utils.og_teleop_utils import (
    augment_rooms,
    generate_robot_config,
    get_task_relevant_room_types,
    load_available_tasks,
)
from omnigibson.envs.env_wrapper import EnvironmentWrapper
from omnigibson.eval.utils.eval_utils import (
    PROPRIOCEPTION_INDICES,
    ROBOT_CAMERA_NAMES,
    TASK_NAMES_TO_INDICES,
    flatten_obs_dict,
    generate_basic_environment_config,
)
from omnigibson.eval.utils.obs_utils import write_video
from omnigibson.macros import create_module_macros, gm, macros
from omnigibson.metrics import AgentMetric, MetricBase, TaskMetric
from omnigibson.robots import Robot
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.bddl_utils import is_system_bddl_inst
from omnigibson.utils.python_utils import recursively_convert_to_torch

m = create_module_macros(module_path=__file__)
m.NUM_EVAL_EPISODES = 1
m.NUM_TRAIN_INSTANCES = 200
m.NUM_EVAL_INSTANCES = 10

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True

with macros.unlocked():
    macros.robots.manipulation_robot.GRASP_WINDOW = 0.75

logger = logging.getLogger("evaluator")
logger.setLevel(logging.INFO)


class Evaluator:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        self.n_trials = 0
        self.n_success_trials = 0
        self.total_time = 0
        self.robot_action = dict()

        self.env = self.load_env(env_wrapper=self.cfg.env_wrapper)
        self.robot = self.load_robot()
        self.policy = self.load_policy()
        self.metrics = self.load_metrics()
        self.obs = None

        self.env._current_episode = 0
        self._video_writer = None

    def load_env(self, env_wrapper: DictConfig) -> EnvironmentWrapper:
        for rule in DISABLED_TRANSITION_RULES:
            rule.ENABLED = False

        available_tasks = load_available_tasks()
        task_name = self.cfg.task.name
        assert task_name in available_tasks, f"Got invalid task name: {task_name}"

        task_idx = TASK_NAMES_TO_INDICES[task_name]
        self.human_stats = {
            "length": [],
            "distance_traveled": [],
            "left_eef_displacement": [],
            "right_eef_displacement": [],
        }
        with open(os.path.join(gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "episodes.jsonl"), "r") as f:
            episodes = [json.loads(line) for line in f]
        for episode in episodes:
            if episode["episode_index"] // 1e4 == task_idx:
                for key in self.human_stats:
                    self.human_stats[key].append(episode[key])
        for key in self.human_stats:
            self.human_stats[key] = sum(self.human_stats[key]) / len(self.human_stats[key])

        task_cfg = available_tasks[task_name][0]
        robot_type = self.cfg.robot.type
        robot_model = robot_type.lower()
        assert robot_model == "r1pro", f"Got invalid robot type: {robot_type}, only R1Pro is supported."

        cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg)
        if self.cfg.partial_scene_load:
            relevant_rooms = get_task_relevant_room_types(activity_name=task_name)
            relevant_rooms = augment_rooms(relevant_rooms, task_cfg["scene_model"], task_name)
            cfg["scene"]["load_room_types"] = relevant_rooms

        cfg["robots"] = [
            generate_robot_config(
                robot_type=robot_model,
                robot_name="robot_r1",
                task_name=task_name,
                task_cfg=task_cfg,
            )
        ]
        cfg["robots"][0]["model"] = cfg["robots"][0].pop("type")
        cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb"]
        cfg["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())
        if self.cfg.robot.controllers is not None:
            cfg["robots"][0]["controller_config"].update(
                OmegaConf.to_container(self.cfg.robot.controllers, resolve=True)
            )

        if self.cfg.max_steps is None:
            logger.info(
                f"Setting timeout to be 2x the average length of human demos: {int(self.human_stats['length'] * 2)}"
            )
            cfg["task"]["termination_config"]["max_steps"] = int(self.human_stats["length"] * 2)
        else:
            logger.info(f"Setting timeout to be {self.cfg.max_steps} steps through config.")
            cfg["task"]["termination_config"]["max_steps"] = self.cfg.max_steps
        cfg["task"]["include_obs"] = False

        env = og.Environment(configs=cfg)
        return instantiate(env_wrapper, env=env)

    def load_robot(self) -> Robot:
        return self.env.scene.object_registry("name", "robot_r1")

    def load_policy(self) -> Any:
        policy = instantiate(self.cfg.model)
        if hasattr(policy, "set_action_dim"):
            policy.set_action_dim(self.robot.action_dim)
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"Loaded policy: {self.cfg.policy_name}")
        logger.info("=" * 50)
        logger.info("")
        return policy

    def load_metrics(self) -> List[MetricBase]:
        return [AgentMetric(self.human_stats), TaskMetric(self.human_stats)]

    def step(self) -> Tuple[bool, bool]:
        self.robot_action = self.policy.forward(obs=self.obs)
        obs, _, terminated, truncated, info = self.env.step(self.robot_action, n_render_iterations=1)
        self.obs = self._preprocess_obs(obs)

        if terminated or truncated:
            self.n_trials += 1
            if info["done"]["success"]:
                self.n_success_trials += 1

        for metric in self.metrics:
            metric.step(self.env, self.robot_action, obs, 0.0, terminated, truncated, info)
        return terminated, truncated

    @property
    def video_writer(self) -> Tuple[Container, Stream]:
        return self._video_writer

    @video_writer.setter
    def video_writer(self, video_writer: Tuple[Container, Stream]) -> None:
        if self._video_writer is not None:
            container, stream = self._video_writer
            for packet in stream.encode():
                container.mux(packet)
            container.close()
        self._video_writer = video_writer

    def load_task_instance(self, instance_id: int, test_hidden: bool = False) -> None:
        scene_model = self.env.task.scene_name
        tro_filename = self.env.task.get_cached_activity_scene_filename(
            scene_model=scene_model,
            activity_name=self.env.task.activity_name,
            activity_definition_id=self.env.task.activity_definition_id,
            activity_instance_id=instance_id,
        )
        if test_hidden:
            tro_file_path = os.path.join(
                gm.DATA_PATH,
                "2025-challenge-test-instances",
                self.env.task.activity_name,
                f"{tro_filename}-tro_state.json",
            )
        else:
            tro_file_path = os.path.join(
                get_task_instance_path(
                    scene_model,
                    f"{scene_model}_task_{self.env.task.activity_name}_instances/{tro_filename}-tro_state",
                )
            )

        with open(tro_file_path, "r") as f:
            tro_state = recursively_convert_to_torch(json.load(f))
        for tro_key, tro_state in tro_state.items():
            if tro_key == "robot_poses":
                presampled_robot_poses = {key.lower(): value for key, value in tro_state.items()}
                if "robot" in presampled_robot_poses:
                    available_poses = presampled_robot_poses["robot"]
                elif self.robot.model in presampled_robot_poses:
                    print("No generic presampled robot pose found, using robot-specific pose.")
                    available_poses = presampled_robot_poses[self.robot.model]
                else:
                    raise KeyError(f"No generic or model-specific presampled robot pose found for {self.robot.model}!")
                self.robot.set_position_orientation(available_poses[0]["position"], available_poses[0]["orientation"])
                self.env.scene.write_task_metadata(key=tro_key, data=tro_state)
            else:
                self.env.task.object_scope[tro_key].load_state(tro_state, serialized=False)

        og.sim.update_handles()
        for _ in range(25):
            og.sim.step_physics()
            for inst, entity in self.env.task.object_scope.items():
                if not is_system_bddl_inst(inst) and entity is not None and not isinstance(entity, Robot):
                    entity.keep_still()

        self.env.scene.update_initial_file()
        self.env.scene.reset()

    def _preprocess_obs(self, obs: dict) -> dict:
        obs = flatten_obs_dict(obs)
        base_pose = self.robot.get_position_orientation()
        cam_rel_poses = []
        for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
            camera = self.robot.sensors[camera_name.split("::")[1]]
            direct_cam_pose = camera.camera_parameters["cameraViewTransform"]
            if np.allclose(direct_cam_pose, np.zeros(16)):
                cam_rel_poses.append(
                    th.cat(T.relative_pose_transform(*(camera.get_position_orientation()), *base_pose))
                )
            else:
                cam_pose = T.mat2pose(th.tensor(np.linalg.inv(np.reshape(direct_cam_pose, [4, 4]).T), dtype=th.float32))
                cam_rel_poses.append(th.cat(T.relative_pose_transform(*cam_pose, *base_pose)))
        obs["robot_r1::cam_rel_poses"] = th.cat(cam_rel_poses, axis=-1)
        obs["task_id"] = th.tensor([TASK_NAMES_TO_INDICES[self.cfg.task.name]], dtype=th.int64)
        return obs

    def _write_video(self) -> None:
        if ROBOT_CAMERA_NAMES["R1Pro"]["head"] + "::rgb" not in self.obs:
            return
        left_wrist_rgb = cv2.resize(
            self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["left_wrist"] + "::rgb"].numpy(),
            (224, 224),
        )
        right_wrist_rgb = cv2.resize(
            self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["right_wrist"] + "::rgb"].numpy(),
            (224, 224),
        )
        head_rgb = cv2.resize(
            self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["head"] + "::rgb"].numpy(),
            (448, 448),
        )
        write_video(
            np.expand_dims(np.hstack([np.vstack([left_wrist_rgb, right_wrist_rgb]), head_rgb]), 0),
            video_writer=self.video_writer,
            batch_size=1,
            mode="rgb",
        )

    def reset(self) -> None:
        self.obs = self._preprocess_obs(self.env.reset()[0])
        for metric in self.metrics:
            metric.reset(self.env)
        self.policy.reset()
        self.n_success_trials, self.n_trials = 0, 0

    def __enter__(self):
        signal(SIGINT, self._sigint_handler)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"Total success trials: {self.n_success_trials}")
        logger.info(f"Total trials: {self.n_trials}")
        if self.n_trials > 0:
            logger.info(f"Success rate: {self.n_success_trials / self.n_trials}")
        logger.info("=" * 50)
        logger.info("")
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, exc_tb)
        self.video_writer = None
        self.env.close()
        og.shutdown()

    def _sigint_handler(self, signal_received, frame):
        logger.warning("SIGINT or CTRL-C detected.\n")
        self.__exit__(None, None, None)
        sys.exit(0)


class BehaviorEvalEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        task_name: str = "turning_on_radio",
        instance_indices: list[int] | None = None,
        log_path: str = "/tmp/og-envhub",
        max_steps: int | None = 1,
        partial_scene_load: bool = True,
        headless: bool = True,
        write_video: bool = False,
    ) -> None:
        super().__init__()
        self.task_name = task_name
        self.instance_ids = self._resolve_instance_ids(task_name, instance_indices or [0])
        self._next_instance_idx = 0
        self._loaded_instance_id = None
        self._closed = False

        gm.HEADLESS = headless
        self.evaluator = Evaluator(
            OmegaConf.create(
                {
                    "env_wrapper": {"_target_": "omnigibson.eval.wrappers.RGBLowResWrapper"},
                    "policy_name": "local",
                    "model": {"_target_": "omnigibson.eval.policies.LocalPolicy", "action_dim": None},
                    "headless": headless,
                    "eval_on_train_instances": False,
                    "eval_instance_ids": instance_indices or [0],
                    "write_video": write_video,
                    "partial_scene_load": partial_scene_load,
                    "max_steps": max_steps,
                    "log_path": log_path,
                    "test_hidden": False,
                    "task": {"name": task_name},
                    "robot": {"type": "R1Pro", "controllers": None},
                }
            )
        )
        self.action_space = self.evaluator.robot.action_space

        obs, _ = self.reset(options={"instance_id": self.instance_ids[0]})
        self.observation_space = gym.spaces.Dict({key: self._space_from_value(value) for key, value in obs.items()})

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        instance_id = options.get("instance_id")
        if instance_id is None:
            instance_id = self.instance_ids[self._next_instance_idx % len(self.instance_ids)]
            self._next_instance_idx += 1

        if instance_id != self._loaded_instance_id:
            self.evaluator.load_task_instance(int(instance_id), test_hidden=False)
            self._loaded_instance_id = int(instance_id)

        self.evaluator.reset()
        obs = self._to_numpy(self.evaluator.obs)
        return obs, {"task_name": self.task_name, "instance_id": int(instance_id)}

    def step(self, action):
        action = th.as_tensor(action, dtype=th.float32)
        obs, reward, terminated, truncated, info = self.evaluator.env.step(action, n_render_iterations=1)
        self.evaluator.obs = self.evaluator._preprocess_obs(obs)

        if terminated or truncated:
            self.evaluator.n_trials += 1
            if info["done"]["success"]:
                self.evaluator.n_success_trials += 1

        for metric in self.evaluator.metrics:
            metric.step(self.evaluator.env, action, obs, reward, terminated, truncated, info)

        if terminated or truncated:
            metrics = {}
            for metric in self.evaluator.metrics:
                metrics.update(metric.aggregate(self.evaluator.env))
            info["metrics"] = metrics

        return self._to_numpy(self.evaluator.obs), float(reward), bool(terminated), bool(truncated), info

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.evaluator.video_writer = None
        self.evaluator.env.close()
        og.shutdown()

    @staticmethod
    def _resolve_instance_ids(task_name: str, instance_indices: list[int]) -> list[int]:
        task_instance_csv_path = os.path.join(
            gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "test_instances.csv"
        )
        with open(task_instance_csv_path, newline="", encoding="utf-8") as f:
            row = next(row for row in csv.DictReader(f) if row["Task"] == task_name)
        test_instances = [int(instance_id.strip()) for instance_id in row["Public Test Instance IDs"].split(",")]
        return [int(test_instances[i]) for i in instance_indices]

    @staticmethod
    def _space_from_value(value):
        value = np.asarray(value)
        if np.issubdtype(value.dtype, np.integer):
            if value.dtype == np.uint8:
                return gym.spaces.Box(low=0, high=255, shape=value.shape, dtype=value.dtype)
            dtype_info = np.iinfo(value.dtype)
            return gym.spaces.Box(low=dtype_info.min, high=dtype_info.max, shape=value.shape, dtype=value.dtype)
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=value.shape, dtype=np.float32)

    @staticmethod
    def _to_numpy(obs: dict[str, Any]) -> dict[str, np.ndarray]:
        converted = {}
        for key, value in obs.items():
            if isinstance(value, th.Tensor):
                value = value.detach().cpu().numpy()
            else:
                value = np.asarray(value)
            if np.issubdtype(value.dtype, np.floating):
                value = value.astype(np.float32, copy=False)
            converted[key] = value
        return converted


def _get_cfg_value(cfg, name, default):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def make_env(n_envs: int = 1, use_async_envs: bool = False, cfg=None):
    if n_envs != 1:
        raise ValueError(
            "BEHAVIOR-1K EnvHub currently supports n_envs=1 because OmniGibson uses a singleton simulator."
        )
    if use_async_envs:
        raise ValueError("BEHAVIOR-1K EnvHub currently supports only synchronous execution.")

    def _make_env():
        return BehaviorEvalEnv(
            task_name=_get_cfg_value(cfg, "task_name", "turning_on_radio"),
            instance_indices=_get_cfg_value(cfg, "instance_indices", [0]),
            log_path=_get_cfg_value(cfg, "log_path", "/tmp/og-envhub"),
            max_steps=_get_cfg_value(cfg, "max_steps", 1),
            partial_scene_load=_get_cfg_value(cfg, "partial_scene_load", True),
            headless=_get_cfg_value(cfg, "headless", True),
            write_video=_get_cfg_value(cfg, "write_video", False),
        )

    return {"behavior_1k": {0: gym.vector.SyncVectorEnv([_make_env])}}


__all__ = ["BehaviorEvalEnv", "Evaluator", "make_env"]
