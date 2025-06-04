import argparse
import hydra
import logging
import omnigibson as og
import sys
import traceback
import gello.robots.sim_robot.og_teleop_utils as utils
from hydra.utils import instantiate
from inspect import getsourcefile
from omegaconf import DictConfig, OmegaConf
from omnigibson.macros import gm
from omnigibson.robots import BaseRobot
from pathlib import Path
from signal import signal, SIGINT
from typing import Any, Tuple


# set global variables to boost performance
gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True
SUPPORTED_ROBOTS = ["R1Pro"]

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
        self.env_type = cfg.env

        self.policy = self.load_policy()
        self.env = self.load_env()
        self.robot = self.load_robot()

        self.obs = self.env.reset()[0]
        # manually reset environment episode number
        self.env._current_episode = 0

    def load_env(self) -> og.Environment:
        """
        Read the environment config file and create the environment.
        The config file is located in the configs/envs directory.
        """
        # Load config file
        if self.env_type == "omnigibson":
            available_tasks = utils.load_available_tasks()
            task_name = self.cfg.task.name
            assert task_name in available_tasks, f"Got invalid OmniGibson task name: {task_name}"
            task_cfg = available_tasks[task_name][0]
            robot_cls = self.cfg.robot.type
            assert robot_cls in SUPPORTED_ROBOTS, f"Got invalid OmniGibson robot type: {robot_cls}"
            cfg = utils.generate_basic_environment_config(task_name, task_cfg)
            cfg["robots"] = [utils.generate_robot_config(task_name, task_cfg)]
            env = og.Environment(configs=cfg)
        else:
            raise ValueError(f"Invalid environment type {self.env_type}")
        return env

    def load_robot(self) -> BaseRobot:
        if self.env_type == "omnigibson":
            robot = self.env.scene.object_registry("name", "robot")
        else:
            raise ValueError(f"Invalid environment type {self.env_type}")
        return robot

    def load_policy(self) -> Any:
        if "policy" in self.cfg:
            return instantiate(self.cfg.policy)
        return None

    def step(self) -> Tuple[bool, bool]:
        """
        Single step of the task
        """
        self.robot_action = self.policy.forward(obs=self.obs)
        self.obs, _, terminated, truncated, info = self.env.step(self.robot_action)
        if terminated:
            self.n_trials += 1
            if info["done"]["success"]:
                self.n_success_trials += 1
        return terminated, truncated

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
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Config name")
    args, _ = parser.parse_known_args()
    # open yaml from task path
    with hydra.initialize_config_dir(f"{Path(getsourcefile(lambda:0)).parents[0]}/configs", version_base="1.1"):
        config = hydra.compose(args.config)

    OmegaConf.resolve(config)
    with Evaluator(config) as evaluator:
        done = False
        while not done:
            terminated, truncated = evaluator.step()
            if terminated:
                evaluator.env.reset()
            if truncated:
                done = True
