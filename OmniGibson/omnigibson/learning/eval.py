import csv
import cv2
import hydra
import json
import logging
import numpy as np
import omnigibson as og
import omnigibson.utils.transform_utils as T
import os
import sys
import torch as th
import traceback
from av.container import Container
from av.stream import Stream
from gello.robots.sim_robot.og_teleop_cfg import SUPPORTED_ROBOTS
from gello.robots.sim_robot.og_teleop_utils import (
    augment_rooms,
    load_available_tasks,
    get_task_relevant_room_types,
    generate_robot_config,
)
from hydra.utils import instantiate
from inspect import getsourcefile
from omegaconf import DictConfig, OmegaConf
from omnigibson.learning.utils.config_utils import register_omegaconf_resolvers
from omnigibson.learning.utils.eval_utils import (
    HEAD_RESOLUTION,
    WRIST_RESOLUTION,
    ROBOT_CAMERA_NAMES,
    PROPRIOCEPTION_INDICES,
    generate_basic_environment_config,
    flatten_obs_dict,
    TASK_NAMES_TO_INDICES,
)
from omnigibson.learning.utils.obs_utils import (
    create_video_writer,
    write_video,
)
from omnigibson.macros import gm
from omnigibson.metrics import MetricBase, RobotMetric, TaskMetric
from omnigibson.robots import BaseRobot
from omnigibson.utils.python_utils import recursively_convert_to_torch
from pathlib import Path
from signal import signal, SIGINT
from typing import Any, Tuple, List


# set global variables to boost performance
gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True

# create module logger
logger = logging.getLogger("evaluator")
logger.setLevel(20)  # info


class Evaluator:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        # record total number and success number of trials and trial time
        self.n_trials = 0
        self.n_success_trials = 0
        self.total_time = 0
        self.robot_action = dict()
        # fetch env type, currently only supports "omnigibson"
        self.env_type = cfg.env_type

        self.env = self.load_env()
        self.policy = self.load_policy()
        self.robot = self.load_robot()
        self.metrics = self.load_metrics()

        self.reset()
        # manually reset environment episode number
        self.env._current_episode = 0
        self._video_writer = None

    def load_env(self) -> og.Environment:
        """
        Read the environment config file and create the environment.
        The config file is located in the configs/envs directory.
        """
        # Load config file
        if self.env_type == "sim":
            available_tasks = load_available_tasks()
            task_name = self.cfg.task
            assert task_name in available_tasks, f"Got invalid OmniGibson task name: {task_name}"
            # Load the seed instance by default
            task_cfg = available_tasks[task_name][0]
            robot_type = self.cfg.robot.type
            assert robot_type in SUPPORTED_ROBOTS, f"Got invalid OmniGibson robot type: {robot_type}"
            cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg)
            cfg["robots"] = [
                generate_robot_config(
                    task_name=task_name,
                    task_cfg=task_cfg,
                )
            ]
            # Update observation modalities
            cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb", "depth_linear"]
            cfg["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())
            if self.cfg.robot.controllers is not None:
                cfg["robots"][0]["controller_config"].update(self.cfg.robot.controllers)
            cfg["task"]["termination_config"]["max_steps"] = self.cfg.max_steps
            cfg["task"]["include_obs"] = self.cfg.use_task_info
            relevant_rooms = get_task_relevant_room_types(activity_name=task_name)
            relevant_rooms = augment_rooms(relevant_rooms, task_cfg["scene_model"], task_name)
            cfg["scene"]["load_room_types"] = relevant_rooms
            env = og.Environment(configs=cfg)
        else:
            raise ValueError(f"Invalid environment type {self.env_type}")
        return env

    def load_robot(self) -> BaseRobot:
        if self.env_type == "sim":
            robot = self.env.scene.object_registry("name", "robot_r1")
            og.sim.step()
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
            self.env.load_observation_space()
        else:
            raise ValueError(f"Invalid environment type {self.env_type}")
        return robot

    def load_policy(self) -> Any:
        policy = instantiate(self.cfg.model)
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"Loaded policy: {self.cfg.policy_name}")
        logger.info("=" * 50)
        logger.info("")
        return policy

    def load_metrics(self) -> List[MetricBase]:
        return [RobotMetric(), TaskMetric()]

    def step(self) -> Tuple[bool, bool]:
        """
        Single step of the task
        """
        self.robot_action = self.policy.forward(obs=self.obs)

        self.obs, _, terminated, truncated, info = self.env.step(self.robot_action, n_render_iterations=3)
        # process obs
        if terminated or truncated:
            self.n_trials += 1
            if info["done"]["success"]:
                self.n_success_trials += 1

        self.obs = self._preprocess_obs(self.obs)
        for metric in self.metrics:
            metric.step_callback(self.env)
        return terminated, truncated

    @property
    def video_writer(self) -> Tuple[Container, Stream]:
        return self._video_writer

    @video_writer.setter
    def video_writer(self, video_writer: Tuple[Container, Stream]) -> None:
        if self._video_writer is not None:
            (container, stream) = self._video_writer
            # Flush any remaining packets
            for packet in stream.encode():
                container.mux(packet)
            # Close the container
            container.close()
        self._video_writer = video_writer

    def load_task_instance(self, instance_id: int) -> None:
        assert self.env_type == "sim", "load_task_instance is only supported in sim environments"
        scene_model = self.env.task.scene_name
        tro_filename = self.env.task.get_cached_activity_scene_filename(
            scene_model=scene_model,
            activity_name=self.env.task.activity_name,
            activity_definition_id=self.env.task.activity_definition_id,
            activity_instance_id=instance_id,
        )
        tro_file_path = f"{gm.DATASET_PATH}/scenes/{scene_model}/json/{scene_model}_task_{self.env.task.activity_name}_instances/{tro_filename}-tro_state.json"
        assert os.path.exists(
            tro_file_path
        ), f"Could not find TRO file at {tro_file_path}, did you run ./populate_behavior_tasks.sh?"
        with open(tro_file_path, "r") as f:
            tro_state = recursively_convert_to_torch(json.load(f))
        self.env.scene.reset()
        for bddl_name, obj_state in tro_state.items():
            if bddl_name == "robot_poses":
                presampled_robot_poses = obj_state
                # Only set pose (we assume this is a holonomic robot, so ignore Rx / Ry and only take Rz component
                # for orientation
                robot_pos = presampled_robot_poses[self.robot.model_name][0]["position"]
                robot_quat = presampled_robot_poses[self.robot.model_name][0]["orientation"]
                self.robot.set_position_orientation(robot_pos, robot_quat)
            else:
                self.env.task.object_scope[bddl_name].load_state(obj_state, serialized=False)

        # Try to ensure that all task-relevant objects are stable
        # They should already be stable from the sampled instance, but there is some issue where loading the state
        # causes some jitter (maybe for small mass / thin objects?)
        for _ in range(25):
            og.sim.step_physics()
            for entity in self.env.task.object_scope.values():
                if not entity.is_system and entity.exists:
                    entity.keep_still()
        self.env.scene.update_initial_file()

    def _preprocess_obs(self, obs: dict) -> dict:
        """
        Preprocess the observation dictionary before passing it to the policy.
        """
        obs = flatten_obs_dict(obs)
        base_pose = self.robot.get_position_orientation()
        cam_rel_poses = []
        for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
            cam_pose = self.robot.sensors[camera_name.split("::")[1]].get_position_orientation()
            cam_rel_poses.append(th.cat(T.relative_pose_transform(*cam_pose, *base_pose)))
        obs["robot_r1::cam_rel_poses"] = th.cat(cam_rel_poses, axis=-1)
        return obs

    def _write_video(self) -> None:
        # concatenate obs
        left_wrist_rgb = cv2.resize(
            self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["left_wrist"] + "::rgb"].numpy(),
            (360, 360),
        )
        right_wrist_rgb = cv2.resize(
            self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["right_wrist"] + "::rgb"].numpy(),
            (360, 360),
        )
        head_rgb = self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["head"] + "::rgb"].numpy()
        write_video(
            np.expand_dims(np.hstack([np.vstack([left_wrist_rgb, right_wrist_rgb]), head_rgb]), 0),
            video_writer=self.video_writer,
            batch_size=1,
            mode="rgb",
        )

    def reset(self) -> None:
        self.obs = self.env.reset()[0]
        self.obs = self._preprocess_obs(self.obs)
        # run metric start callbacks
        for metric in self.metrics:
            metric.start_callback(self.env)
        self.policy.reset()
        self.n_success_trials, self.n_trials = 0, 0

    def __enter__(self):
        signal(SIGINT, self._sigint_handler)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        # print stats
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

    def _sigint_handler(self, signal_received, frame):
        logger.warning("SIGINT or CTRL-C detected.\n")
        self.__exit__(None, None, None)
        og.shutdown()
        sys.exit(0)


if __name__ == "__main__":
    register_omegaconf_resolvers()
    # open yaml from task path
    with hydra.initialize_config_dir(f"{Path(getsourcefile(lambda:0)).parents[0]}/configs", version_base="1.1"):
        config = hydra.compose("base_config.yaml", overrides=sys.argv[1:])
    OmegaConf.resolve(config)
    # set headless mode
    gm.HEADLESS = config.headless
    # set video path
    if config.write_video:
        video_path = Path(config.log_path).expanduser() / "videos"
        video_path.mkdir(parents=True, exist_ok=True)
    # get run instances
    instances_to_run = config.eval_instance_ids if config.eval_instance_ids is not None else set(range(10))
    assert set(instances_to_run).issubset(set(range(10))), "eval instance ids must be in range(10)"
    # load csv file
    task_instance_csv_path = f"{gm.DATASET_PATH}/metadata/test_instances.csv"
    with open(task_instance_csv_path, "r") as f:
        lines = list(csv.reader(f))[1:]
    assert (
        lines[TASK_NAMES_TO_INDICES[config.task]][1] == config.task
    ), f"Task name from config {config.task} does not match task name from csv {lines[TASK_NAMES_TO_INDICES[config.task]][1]}"
    test_instances = lines[TASK_NAMES_TO_INDICES[config.task]][2].strip().split(",")
    instances_to_run = [int(test_instances[i]) for i in instances_to_run]
    # establish metrics
    metrics = {}
    metrics_path = Path(config.log_path).expanduser() / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)

    with Evaluator(config) as evaluator:
        logger.info("Starting evaluation...")

        for idx in instances_to_run:
            evaluator.load_task_instance(idx)
            logger.info(f"Starting task instance {idx} for evaluation...")
            for epi in range(config.episodes_per_instance):
                for _ in range(10):
                    og.sim.render()
                evaluator.reset()
                done = False
                if config.write_video:
                    video_name = str(video_path) + f"/video_{idx}_{epi}.mp4"
                    evaluator.video_writer = create_video_writer(
                        fpath=video_name,
                        resolution=(720, 1080),
                    )
                # run metric start callbacks
                for metric in evaluator.metrics:
                    metric.start_callback(evaluator.env)
                while not done:
                    terminated, truncated = evaluator.step()
                    if terminated or truncated:
                        done = True
                    if config.write_video:
                        evaluator._write_video()
                    if evaluator.env._current_step % 1000 == 0:
                        logger.info(f"Current step: {evaluator.env._current_step}")
                # run metric end callbacks
                for metric in evaluator.metrics:
                    metric.end_callback(evaluator.env)
                logger.info(f"Evaluation finished at step {evaluator.env._current_step}.")
                logger.info(f"Evaluation exit state: {terminated}, {truncated}")
                logger.info(f"Total trials: {evaluator.n_trials}")
                logger.info(f"Total success trials: {evaluator.n_success_trials}")
                # gather metric results and write to file
                for metric in evaluator.metrics:
                    metrics[metric.name] = metric.gather_results()
                with open(metrics_path / f"{config.task}::{idx}::{epi}.json", "w") as f:
                    json.dump(metrics, f)
                # reset video writer
                if config.write_video:
                    evaluator.video_writer = None
                    logger.info(f"Saved video to {video_name}")
                else:
                    logger.warning("No observations were recorded.")
