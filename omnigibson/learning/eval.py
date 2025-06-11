import hydra
import logging
import omnigibson as og
import sys
import traceback
from gello.robots.sim_robot.og_teleop_utils import load_available_tasks
from hydra.utils import call
from inspect import getsourcefile
from omegaconf import DictConfig, OmegaConf
from omnigibson.learning.utils.config_utils import register_omegaconf_resolvers
from omnigibson.learning.utils.eval_utils import (
    SUPPORTED_ROBOTS,
    generate_basic_environment_config,
    generate_robot_config,
    flatten_obs_dict,
)
from omnigibson.macros import gm
from omnigibson.robots import BaseRobot
from pathlib import Path
from signal import signal, SIGINT
from typing import Any, Tuple
import torch as th
import torch
import mediapy
# set global variables to boost performance
gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True

# create module logger
logger = logging.getLogger("evaluator")
logger.setLevel(20)  # info


def load_openpi_model():

    from omnigibson.learning.policies.eval_b1k_wrapper import OpenPIWrapper

    # need to launch the openpi server first
    openpi_policy = OpenPIWrapper(
        host="10.79.12.231",
        port=8000,
        text_prompt="pick up the trash",
        control_mode="receeding_temporal"
    )

    return openpi_policy

class Evaluator:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        # record total number and success number of trials and trial time
        self.n_trials = 0
        self.n_success_trials = 0
        self.total_time = 0
        self.robot_action = dict()
        # fetch env type, currently only supports "omnigibson"
        self.env_type = cfg.env

        self.policy = self.load_policy()
        self.env = self.load_env()
        self.robot = self.load_robot()

        self.obs = self.env.reset()[0]
        # manually reset environment episode number
        self.env._current_episode = 0
        self.obs_buffer = []

    def load_env(self) -> og.Environment:
        """
        Read the environment config file and create the environment.
        The config file is located in the configs/envs directory.
        """
        # Load config file
        if self.env_type == "omnigibson":
            available_tasks = load_available_tasks()
            task_name = self.cfg.task.name
            assert task_name in available_tasks, f"Got invalid OmniGibson task name: {task_name}"
            task_cfg = available_tasks[task_name][0]
            robot_type = self.cfg.robot.type
            robot_controller_cfg = self.cfg.robot.controllers
            assert robot_type in SUPPORTED_ROBOTS, f"Got invalid OmniGibson robot type: {robot_type}"
            cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg, robot_type=robot_type)
            cfg["robots"] = [
                generate_robot_config(
                    task_name=task_name,
                    task_cfg=task_cfg,
                    robot_type=robot_type,
                    overwrite_controller_cfg=robot_controller_cfg,
                )
            ]
            env = og.Environment(configs=cfg)
        else:
            raise ValueError(f"Invalid environment type {self.env_type}")
        return env

    def load_robot(self) -> BaseRobot:
        if self.env_type == "omnigibson":
            robot = self.env.scene.object_registry("name", "robot_r1")
        else:
            raise ValueError(f"Invalid environment type {self.env_type}")
        return robot

    def load_policy(self) -> Any:
        if self.cfg.policy_name == 'pi0':
            policy = load_openpi_model()
        else:
            policy = call(self.cfg.eval)
            policy.eval()
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"Loaded policy: {self.cfg.policy_name}")
        logger.info("=" * 50)
        logger.info("")
        return policy

    def step(self) -> Tuple[bool, bool]:
        """
        Single step of the task
        """
        obs = self._preprocess_obs(self.obs)
        self.robot_action = self.policy.forward(obs={"obs": obs})
        #concatenate the three camera images into one and save to obs_buffer
        all_obs = torch.cat(
            [
                obs["robot_r1::robot_r1:zed_link:Camera:0::rgb"],
                obs["robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"],
                obs["robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"],
            ],
            dim=0,
        )
        self.obs_buffer.append(all_obs)
        
        self.obs, _, terminated, truncated, info = self.env.step(self.robot_action)
        # process obs
        if terminated:
            self.n_trials += 1
            if info["done"]["success"]:
                self.n_success_trials += 1
        return terminated, truncated

    def _preprocess_obs(self, obs: dict) -> dict:
        """
        Preprocess the observation dictionary before passing it to the policy.
        """
        obs = flatten_obs_dict(obs)
        base_link_pose = th.concatenate(self.robot.get_position_orientation())
        left_cam_pose = th.concatenate(
            self.robot.sensors["robot_r1:left_realsense_link:Camera:0"].get_position_orientation()
        )
        right_cam_pose = th.concatenate(
            self.robot.sensors["robot_r1:right_realsense_link:Camera:0"].get_position_orientation()
        )
        external_cam_pose = th.concatenate(self.env.external_sensors["external_sensor0"].get_position_orientation())
        # store the poses to obs
        obs["robot_r1::robot_base_link_pose"] = base_link_pose
        obs["robot_r1::left_cam_pose"] = left_cam_pose
        obs["robot_r1::right_cam_pose"] = right_cam_pose
        obs["robot_r1::external_cam_pose"] = external_cam_pose
        return obs

    def reset(self) -> None:
        self.obs = self.env.reset()[0]
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
        
    from omnigibson.macros import gm
    gm.HEADLESS = True

    
    video_path = Path(config.log_path)/"eval_video.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
            
    OmegaConf.resolve(config)
    with Evaluator(config) as evaluator:
        logger.info("Starting evaluation...")
        done = False
        while not done:
            terminated, truncated = evaluator.step()
            if terminated:
                evaluator.env.reset()
            if truncated:
                done = True
                
            if evaluator.env._current_step % 100 == 0:
                logger.info(f"Current step: {evaluator.env._current_step}")
                mediapy.write_video(
                    str(video_path),
                    torch.stack(evaluator.obs_buffer).cpu().numpy()[...,:3],
                    fps=30,
                )
                logger.info(f"Saved video to {video_path}")
        
        logger.info(f"Evaluation finished at step {evaluator.env._current_step}.")
        logger.info(f"Evaluation exit state: {terminated}, {truncated}")
        logger.info(f"Total trials: {evaluator.n_trials}")
        logger.info(f"Total success trials: {evaluator.n_success_trials}")
        #save obs_buffer to a video using mediapy
        if len(evaluator.obs_buffer) > 0:
            mediapy.write_video(
                str(video_path),
                torch.stack(evaluator.obs_buffer).cpu().numpy()[...,:3],
                fps=30,
            )
            evaluator.obs_buffer = []
            logger.info(f"Saved video to {video_path}")
        else:
            logger.warning("No observations were recorded.")
        
