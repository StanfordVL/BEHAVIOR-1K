import logging
import time

import gym

from igibson.action_primitives.action_primitive_set_base import (
    REGISTERED_PRIMITIVE_SETS,
    ActionPrimitiveError,
    BaseActionPrimitiveSet,
)
from igibson.wrappers.wrapper_base import BaseWrapper

logger = logging.getLogger(__name__)


class ActionPrimitiveWrapper(BaseWrapper):
    def __init__(
        self,
        env,
        action_generator,
        reward_accumulation="sum",
        accumulate_obs=False,
        num_attempts=10,
    ):
        """
        Environment wrapper class for mapping action primitives to low-level environment actions

        Args:
            env (iGibsonEnv): The environment to wrap.
            @param action_generator (str): The BaseActionPrimitives subclass name to use for generating actions.
            @param reward_accumulation (str): Whether rewards across lower-level env timesteps should be summed or maxed.
                Options: {"sum"}
            @param accumulate_obs (bool): Whether all observations should be returned instead of just the last lower-level step.
            @param num_attempts (int): How many times a primitive will be re-attempted if previous tries fail.
        """
        super().__init__(env=env)

        self.action_generator: BaseActionPrimitiveSet = REGISTERED_PRIMITIVE_SETS[action_generator](
            self, self.task, self.scene, self.robots[0]
        )

        self.action_space = self.action_generator.get_action_space()
        self.reward_accumulation = reward_accumulation
        self.accumulate_obs = accumulate_obs
        self.num_attempts = num_attempts

    def step(self, action: int):
        # Run the goal generator and feed the goals into the env.
        accumulated_reward = 0
        accumulated_obs = []

        start_time = time.time()

        for _ in range(self.num_attempts):
            obs, done, info = None, None, {}
            try:
                for lower_level_action in self.action_generator.apply(action):
                    # Run super step
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
                        accumulated_obs = [obs]  # Do this to save some memory.

                    # Record additional info.
                    info["primitive_success"] = True
                    info["primitive_error_reason"] = None
                    info["primitive_error_metadata"] = None
                    info["primitive_error_message"] = None

                break
            except ActionPrimitiveError as e:
                end_time = time.time()
                logger.error("AP time: {}".format(end_time - start_time))
                logger.warning("Action primitive failed! Exception {}".format(e))
                # Record the error info.
                info["primitive_success"] = False
                info["primitive_error_reason"] = e.reason
                info["primitive_error_metadata"] = e.metadata
                info["primitive_error_message"] = str(e)

        # TODO: Think more about what to do when no observations etc. can be obtained.
        return_obs = None
        if accumulated_obs:
            if self.accumulate_obs:
                return_obs = accumulated_obs
            else:
                return_obs = accumulated_obs[-1]
        end_time = time.time()
        logger.error("AP time: {}".format(end_time - start_time))
        return return_obs, accumulated_reward, done, info
