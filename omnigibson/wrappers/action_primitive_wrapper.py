import logging
import time

import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import queue
import cv2
import os
import omnigibson as og
from collections import OrderedDict

from omnigibson.action_primitives.action_primitive_set_base import (
    REGISTERED_PRIMITIVE_SETS,
    ActionPrimitiveError,
    BaseActionPrimitiveSet,
)
from omnigibson.wrappers.wrapper_base import BaseWrapper

from omnigibson.object_states.pose import Pose
from omnigibson.sensors.vision_sensor import VisionSensor
logger = logging.getLogger(__name__)


class ActionPrimitiveWrapper(BaseWrapper):
    def __init__(
        self,
        env,
        action_generator="BehaviorActionPrimitives",
        reward_accumulation="sum",
        accumulate_obs=False,
        num_attempts=3,
        flatten_nested_obs=True,
        mode=''
    ):
        """
        Environment wrapper class for mapping action primitives to low-level environment actions

        Args:
            env (OmniGibsonEnv): The environment to wrap.
            @param action_generator (str): The BaseActionPrimitives subclass name to use for generating actions.
            @param reward_accumulation (str): Whether rewards across lower-level env timesteps should be summed or maxed.
                Options: {"sum"}
            @param accumulate_obs (bool): Whether all observations should be returned instead of just the last lower-level step.
            @param num_attempts (int): How many times a primitive will be re-attempted if previous tries fail.
        """
        super().__init__(env=env)
        self.mode = mode
        self.seed(0)
        self.action_generator: BaseActionPrimitiveSet = REGISTERED_PRIMITIVE_SETS[action_generator](
            self, self.task, self.scene, self.robots[0], mode=self.mode
        )
        self.action_space = self.action_generator.get_action_space()
        self.reward_accumulation = reward_accumulation
        self.accumulate_obs = accumulate_obs
        self.num_attempts = num_attempts
        self.accum_reward = np.asarray([0])
        self.arm = 'left'
        self.step_index = 0
        self.done = False
        self.fallback_state = None
        self.max_step = 30  # env.config['max_step']
        self.is_success_list = []

        observation_space = OrderedDict()
        observation_space["rgb"] = gym.spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(observation_space)
        self.reset()

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        """
        By default, run the normal environment reset() function
        Returns:
            OrderedDict: Environment observation space after reset occurs
        """
        self.action_generator.robot.release_grasp_immediately()
        self.step_index = 0
        self.done = False
        self.accum_reward = 0
        return_obs = self.env.reset()
        return_obs = {
            'rgb': return_obs['robot0']['robot0:eyes_Camera_sensor_rgb'][:, :, :3] / 255.
        }
        
        return_obs, accumulated_reward, done, info = self.step(0)
        self.fallback_state = og.sim.dump_state(serialized=False)
        print('Success Rate: {}\n'.format(np.mean(self.is_success_list)))
        return return_obs

    def step(self, action: int):
        # Run the goal generator and feed the goals into the env.
        accumulated_reward = 0
        accumulated_obs = []

        start_time = time.time()

        print('++++++++++++++++ take action:{} ++++++++++++++++'.format(action))

        pre_action = 10
        for lower_level_action in self.action_generator.apply(pre_action):
            obs, reward, done, info = super().step(lower_level_action)
            if self.accumulate_obs:
                accumulated_obs.append(obs)
            else:
                accumulated_obs = [obs]

        for _ in range(self.num_attempts):
            obs, done, info = None, None, {}
            try:
                for lower_level_action in self.action_generator.apply(action):
                    obs, reward, done, info = super().step(lower_level_action)

                    if self.reward_accumulation == "sum":
                        accumulated_reward += reward
                    elif self.reward_accumulation == "max":
                        accumulated_reward = max(reward, accumulated_reward)
                    else:
                        raise ValueError("Reward accumulation should be one of 'sum' and 'max'.")

                    if self.accumulate_obs:
                        accumulated_obs.append(obs)
                    else:
                        accumulated_obs = [obs]

                    # Record additional info.
                    info["primitive_success"] = True
                    info["primitive_error_reason"] = None
                    info["primitive_error_metadata"] = None
                    info["primitive_error_message"] = None

                self.fallback_state = og.sim.dump_state(serialized=False)
                break
            except ActionPrimitiveError as e:
                print("--- Primitive Error! Execute dummy action (don't move) to get an observation!")
                from copy import deepcopy
                og.sim.load_state(deepcopy(self.fallback_state), serialized=False)

                dummy_action_id = 10
                for lower_level_action in self.action_generator.apply(dummy_action_id):
                    obs, reward, done, info = super().step(lower_level_action)
                    if self.accumulate_obs:
                        accumulated_obs.append(obs)
                    else:
                        accumulated_obs = [obs]

                info["primitive_success"] = False
                info["primitive_error_reason"] = e.reason
                info["primitive_error_metadata"] = e.metadata
                info["primitive_error_message"] = str(e)

        return_obs = None
        if accumulated_obs:
            if self.accumulate_obs:
                return_obs = accumulated_obs
            else:
                return_obs = accumulated_obs[-1]
        self.accum_reward = self.accum_reward + accumulated_reward
        print('reward: ', accumulated_reward, 'accum reward: ', self.accum_reward)
        self.step_index = self.step_index + 1
        if self.accum_reward >= 1.0:
            self.done = True
            info["is_success"] = True
            self.is_success_list.append(True)
        elif self.step_index >= self.max_step:
            self.done = True
            info["is_success"] = False
            self.is_success_list.append(False)
        else:
            self.done = False
        return_obs = {
            'rgb': return_obs['robot0']['robot0:eyes_Camera_sensor_rgb'][:, :, :3] / 255.
        }
        print('done: {}'.format(self.done))
        return return_obs, accumulated_reward, self.done, info
