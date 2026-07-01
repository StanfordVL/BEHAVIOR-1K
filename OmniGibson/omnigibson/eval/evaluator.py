import csv
import cv2
import json
import logging
import os
import sys
import traceback
from signal import SIGINT, signal
from typing import Any, List, Tuple

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
    get_task_relevant_room_types,
    load_available_tasks,
)
from omnigibson.envs.env_wrapper import EnvironmentWrapper
from omnigibson.eval.utils.eval_utils import (
    TASK_NAMES_TO_INDICES,
    flatten_obs_dict,
    generate_basic_environment_config,
    get_robot_camera_names,
)
from omnigibson.eval.utils.obs_utils import create_video_writer, write_video
from omnigibson.eval.utils.score_utils import load_human_stats
from omnigibson.macros import gm
from omnigibson.metrics import AgentMetric, MetricBase, TaskMetric
from omnigibson.robots import Robot
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.bddl_utils import is_system_bddl_inst
from omnigibson.eval.utils.light_utils import LightToggleSynchronizer, set_light_control_toggles
from omnigibson.utils.python_utils import recursively_convert_to_torch
from omnigibson.utils.ui_utils import create_module_logger


LIGHT_EVAL_TASKS = {"turning_out_all_lights_before_sleep"}
EVAL_BASE_LINK_MASS = 250.0
EVAL_HEAD_HORIZONTAL_APERTURE = 40.0
NUM_TEST_INSTANCES = 40
DEFAULT_ROBOT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "r1pro.yaml")

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True

logger = create_module_logger(module_name=__name__)
logger.setLevel(logging.INFO)


def _to_plain_dict(cfg: Any) -> dict | None:
    if cfg is None:
        return None
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    return dict(cfg)


def resolve_instance_ids(task_name: str, instance_indices: list[int]) -> list[int]:
    task_instance_csv_path = os.path.join(
        gm.DATA_PATH, "2026-challenge-task-instances", "metadata", "test_instances.csv"
    )
    with open(task_instance_csv_path, newline="", encoding="utf-8") as f:
        row = next(row for row in csv.DictReader(f) if row["Task"] == task_name)
    test_instances = [int(instance_id.strip()) for instance_id in row["Test Instance IDs"].split(",")]
    assert (
        len(test_instances) == NUM_TEST_INSTANCES
    ), f"Expected {NUM_TEST_INSTANCES} test instances for {task_name}, got {len(test_instances)}"
    assert set(instance_indices).issubset(
        set(range(NUM_TEST_INSTANCES))
    ), f"Instance indices must be in range({NUM_TEST_INSTANCES})"
    return [int(test_instances[i]) for i in instance_indices]


class Evaluator:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        self.n_trials = 0
        self.n_success_trials = 0
        self.total_time = 0
        self.robot_action = dict()
        self.robot_name = None
        self.robot_eval_config = {}
        self.robot_camera_names = {}

        self.env = self.load_env(env_wrapper=self.cfg.env_wrapper)
        self.robot = self.load_robot()
        self.robot_camera_names = get_robot_camera_names(self.robot.name, self.robot_eval_config)
        self._validate_robot_eval_config()
        self._apply_robot_eval_settings()
        self.policy = self.load_policy()
        self.metrics = self.load_metrics()
        self.obs = None
        self.light_synchronizer = None

        # Initialize the physics views so the first reset() (which eval.py issues before the first
        # load_task_instance) can read/restore robot joint state.
        og.sim.update_handles()

        self.env._current_episode = 0
        self._video_writer = None
        self._video_path = None
        self._video_rate = 30

    @property
    def should_sync_lights(self) -> bool:
        return self.env.task.activity_name in LIGHT_EVAL_TASKS

    def _reset_light_synchronizer(self) -> None:
        if self.should_sync_lights:
            self.light_synchronizer = LightToggleSynchronizer(self.env.scene)
            self.light_synchronizer.reset_from_current_state()
        else:
            self.light_synchronizer = None

    def _sync_lights_and_get_obs(self, obs: dict | None = None) -> dict:
        if not self.should_sync_lights:
            return obs

        if self.light_synchronizer is None:
            self._reset_light_synchronizer()
        else:
            self.light_synchronizer.sync_from_current_state()
        for _ in range(3):
            og.sim.render()
        obs, _ = self.env.get_obs()
        return obs

    def load_env(self, env_wrapper: DictConfig) -> EnvironmentWrapper:
        for rule in DISABLED_TRANSITION_RULES:
            rule.ENABLED = False

        available_tasks = load_available_tasks()
        task_name = self.cfg.task.name
        assert task_name in available_tasks, f"Got invalid task name: {task_name}"

        self.human_stats = load_human_stats(task_name)

        task_cfg = available_tasks[task_name][0]
        cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg)
        if self.cfg.partial_scene_load:
            relevant_rooms = get_task_relevant_room_types(activity_name=task_name)
            relevant_rooms = augment_rooms(relevant_rooms, task_cfg["scene_model"], task_name)
            cfg["scene"]["load_room_types"] = relevant_rooms

        robot_cfg = self._build_robot_config(task_name=task_name, task_cfg=task_cfg)
        self.robot_name = robot_cfg["name"]
        cfg["robots"] = [robot_cfg]

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
        env._eval_robot_config = self.robot_eval_config
        return instantiate(env_wrapper, env=env)

    def load_robot(self) -> Robot:
        if self.robot_name is not None:
            return self.env.scene.object_registry("name", self.robot_name)
        return self.env.robots[0]

    def _validate_robot_eval_config(self) -> None:
        missing_camera_sensors = []
        for camera_id, camera_name in self.robot_camera_names.items():
            sensor_name = camera_name.split("::", 1)[1]
            if sensor_name not in self.robot.sensors:
                missing_camera_sensors.append(f"{camera_id}: {sensor_name}")
        if missing_camera_sensors:
            raise ValueError(
                "Configured eval.camera_sensor_names entries were not found in robot.sensors: "
                f"{missing_camera_sensors}"
            )

        if self.cfg.get("write_video", False):
            required_camera_ids = {"left_wrist", "right_wrist", "head"}
            missing_camera_ids = sorted(required_camera_ids - set(self.robot_camera_names))
            if missing_camera_ids:
                raise ValueError(
                    "--write-video requires eval.camera_sensor_names roles "
                    f"{sorted(required_camera_ids)}; missing {missing_camera_ids}"
                )

    def _build_robot_config(self, task_name: str, task_cfg: dict) -> dict:
        robot_cfg = _to_plain_dict(OmegaConf.select(self.cfg, "robot"))
        if robot_cfg is None:
            robot_cfg = _to_plain_dict(OmegaConf.load(DEFAULT_ROBOT_CONFIG_PATH))
        else:
            robot_cfg = dict(robot_cfg)
        assert "model" in robot_cfg, "Robot config must include canonical 'model'"
        assert "type" not in robot_cfg, "Robot config must use canonical 'model', not 'type'"
        robot_cfg["model"] = robot_cfg["model"].lower()
        assert "name" in robot_cfg, "Robot config must include 'name'"
        self.robot_eval_config = _to_plain_dict(robot_cfg.pop("eval", None)) or {}
        robot_cfg["position"] = task_cfg["robot_start_position"]
        robot_cfg["orientation"] = task_cfg["robot_start_orientation"]

        return robot_cfg

    def _apply_robot_eval_settings(self) -> None:
        if self.robot.model in ("r1", "r1pro"):
            og.sim.stop()
            self.robot.base_footprint_link.mass = EVAL_BASE_LINK_MASS
            og.sim.play()

        head_camera_name = self.robot_camera_names.get("head")
        if head_camera_name is not None:
            head_sensor_name = head_camera_name.split("::")[1]
            if head_sensor_name in self.robot.sensors:
                self.robot.sensors[head_sensor_name].horizontal_aperture = EVAL_HEAD_HORIZONTAL_APERTURE

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
        obs = self._sync_lights_and_get_obs(obs)
        self.obs = self._preprocess_obs(obs)

        if self._video_path is not None:
            self._write_video()

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

    def load_task_instance(self, instance_id: int) -> None:
        scene_model = self.env.task.scene_name
        tro_filename = self.env.task.get_cached_activity_scene_filename(
            scene_model=scene_model,
            activity_name=self.env.task.activity_name,
            activity_definition_id=self.env.task.activity_definition_id,
            activity_instance_id=instance_id,
        )
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
                    logger.info("No generic presampled robot pose found, using robot-specific pose.")
                    available_poses = presampled_robot_poses[self.robot.model]
                else:
                    raise KeyError(f"No generic or model-specific presampled robot pose found for {self.robot.model}!")
                self.robot.set_position_orientation(available_poses[0]["position"], available_poses[0]["orientation"])
                self.env.scene.write_task_metadata(key=tro_key, data=tro_state)
            else:
                self.env.task.object_scope[tro_key].load_state(tro_state, serialized=False)

        if self.should_sync_lights:
            set_light_control_toggles(self.env.task.object_scope.values(), True)

        # Keep all task-relevant entities (including the robot/agent) still while the scene settles,
        # so the snapshotted per-instance initial state is stable. The robot must already be in a clean
        # configuration when this is called -- eval.py resets before load_task_instance for this reason.
        og.sim.update_handles()
        for _ in range(25):
            og.sim.step_physics()
            for inst, entity in self.env.task.object_scope.items():
                if not is_system_bddl_inst(inst) and entity is not None:
                    entity.keep_still()

        self.env.scene.update_initial_file()
        self.env.scene.reset()
        self._reset_light_synchronizer()

    def _preprocess_obs(self, obs: dict) -> dict:
        obs = flatten_obs_dict(obs)
        base_pose = self.robot.get_position_orientation()
        cam_rel_poses = []
        for camera_name in self.robot_camera_names.values():
            sensor_name = camera_name.split("::")[1]
            if sensor_name not in self.robot.sensors:
                continue
            camera = self.robot.sensors[sensor_name]
            direct_cam_pose = camera.camera_parameters["cameraViewTransform"]
            if np.allclose(direct_cam_pose, np.zeros(16)):
                cam_rel_poses.append(
                    th.cat(T.relative_pose_transform(*(camera.get_position_orientation()), *base_pose))
                )
            else:
                cam_pose = T.mat2pose(th.tensor(np.linalg.inv(np.reshape(direct_cam_pose, [4, 4]).T), dtype=th.float32))
                cam_rel_poses.append(th.cat(T.relative_pose_transform(*cam_pose, *base_pose)))
        if cam_rel_poses:
            obs[f"{self.robot.name}::cam_rel_poses"] = th.cat(cam_rel_poses, axis=-1)
        obs["task_id"] = th.tensor([TASK_NAMES_TO_INDICES[self.cfg.task.name]], dtype=th.int64)
        return obs

    def _write_video(self) -> None:
        required_camera_ids = ("left_wrist", "right_wrist", "head")
        if not all(camera_id in self.robot_camera_names for camera_id in required_camera_ids):
            return
        if self.robot_camera_names["head"] + "::rgb" not in self.obs:
            return
        left_wrist_rgb = cv2.resize(
            self.obs[self.robot_camera_names["left_wrist"] + "::rgb"].numpy(),
            (224, 224),
        )
        right_wrist_rgb = cv2.resize(
            self.obs[self.robot_camera_names["right_wrist"] + "::rgb"].numpy(),
            (224, 224),
        )
        head_rgb = cv2.resize(
            self.obs[self.robot_camera_names["head"] + "::rgb"].numpy(),
            (448, 448),
        )
        frame = np.expand_dims(np.hstack([np.vstack([left_wrist_rgb, right_wrist_rgb]), head_rgb]), 0)
        if self._video_writer is None:
            # Writer is created lazily so its resolution matches the composite (H, W) frame above.
            self.video_writer = create_video_writer(
                self._video_path, resolution=frame.shape[1:3], rate=self._video_rate
            )
        write_video(frame, video_writer=self.video_writer, batch_size=1, mode="rgb")

    def start_recording(self, fpath: str, rate: int = 30) -> None:
        # Finalize any in-progress recording, then arm a new one (writer is created on the first frame).
        self.video_writer = None
        self._video_path = fpath
        self._video_rate = rate

    def stop_recording(self) -> None:
        self.video_writer = None
        self._video_path = None

    def reset(self) -> None:
        obs = self.env.reset()[0]
        self._reset_light_synchronizer()
        obs = self._sync_lights_and_get_obs(obs)
        self.obs = self._preprocess_obs(obs)
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


__all__ = ["Evaluator", "resolve_instance_ids"]
