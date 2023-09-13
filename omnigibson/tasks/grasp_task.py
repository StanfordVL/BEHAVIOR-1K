import numpy as np
from omnigibson.reward_functions.grasp_reward import GraspReward

from omnigibson.tasks.task_base import BaseTask
from omnigibson.scenes.scene_base import Scene
from omnigibson.termination_conditions.grasp_goal import GraspGoal
from omnigibson.termination_conditions.timeout import Timeout
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.sim_utils import land_object

DIST_COEFF = 0.1
GRASP_REWARD = 1.0

class GraspTask(BaseTask):
    """
    Grasp task
    """

    def __init__(
        self,
        obj_name,
        termination_config=None,
        reward_config=None,
    ):
        self.obj_name = obj_name
    
        super().__init__(termination_config=termination_config, reward_config=reward_config)
        

    def _load(self, env):
        # Do nothing here
        pass  

    def _create_termination_conditions(self):
        # Run super first
        terminations = super()._create_termination_conditions()

        terminations["graspgoal"] = GraspGoal(
            self.obj_name
        )
        terminations["timeout"] = Timeout(max_steps=self._termination_config["max_steps"])

        return terminations


    def _create_reward_functions(self):
        rewards = dict()
        rewards["grasp"] = GraspReward(
            self.obj_name,
            dist_coeff=self._reward_config["r_dist_coeff"],
            grasp_reward=self._reward_config["r_grasp"]
        )
        return rewards

    def _get_obs(self, env):
        # No task-specific obs of any kind
        return dict(), dict()

    def _load_non_low_dim_observation_space(self):
        # No non-low dim observations so we return an empty dict
        return dict()

    @classproperty
    def valid_scene_types(cls):
        # Any scene works
        return {Scene}

    @classproperty
    def default_termination_config(cls):
        # Empty dict
        return {}

    @classproperty
    def default_reward_config(cls):
        return {
            "r_dist_coeff": DIST_COEFF,
            "r_grasp": GRASP_REWARD,
        }

