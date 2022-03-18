import logging
from collections import OrderedDict

import numpy as np


from igibson.reward_functions.potential_reward import PotentialReward
from igibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from igibson.tasks.task_base import BaseTask
from igibson.termination_conditions.max_collision import MaxCollision
from igibson.termination_conditions.falling import Falling
from igibson.termination_conditions.timeout import Timeout
from igibson.utils.python_utils import classproperty


FURNITURE_CATEGORIES = {
    "bottom_cabinet",
    "bottom_cabinet_no_top",
    "top_cabinet",
    "dishwasher",
    "fridge",
    "microwave",
    "oven",
    "washer",
    "dryer",
}


class FurnitureClosingTask(BaseTask):
    """
    Furniture Closing Task
    The goal is to close as many furniture (e.g. cabinets and fridges) as possible
    """

    def __init__(
            self,
            robot_idn=0,
            floor=0,
            categories="all",
            p_open=0.5,
            termination_config=None,
            reward_config=None,

    ):
        # Store inputs
        self._robot_idn = robot_idn
        self._floor = floor
        self._categories = FURNITURE_CATEGORIES if categories == "all" else \
            set([categories]) if isinstance(categories, str) else set(categories)
        self._p_open = p_open

        # Initialize other values that will be loaded at runtime
        self._r_prismatic = None
        self._r_revolute = None
        self._opened_objects = None

        # Run super init
        super().__init__(termination_config=termination_config, reward_config=reward_config)

    def _load(self, env):
        # Nothing to do here
        pass

    def _create_termination_conditions(self):
        # Initialize termination conditions dict and fill in with MaxCollision, Timeout, and Falling
        terminations = OrderedDict()

        terminations["max_collision"] = MaxCollision(max_collisions=self._termination_config["max_collisions"])
        terminations["timeout"] = Timeout(max_steps=self._termination_config["max_steps"])
        terminations["falling"] = Falling(robot_idn=self._robot_idn, fall_height=self._termination_config["fall_height"])

        return terminations

    def _create_reward_functions(self):
        # Initialize reward functions dict and fill in with Potential reward
        rewards = OrderedDict()

        rewards["potential"] = PotentialReward(
            potential_fcn=self.get_potential,
            r_potential=self._reward_config["r_potential"],
        )

        # Also save other rewards not associated with a reward function internall
        self._r_prismatic = self._reward_config["r_prismatic"]
        self._r_revolute = self._reward_config["r_revolute"]

        return rewards

    def get_potential(self, env):
        """
        Compute task-specific potential: furniture joint positions

        Args:
            env (BaseEnv): Environment instance
        """
        task_potential = 0.0
        for obj in self._opened_objects:
            for joint in obj.joints.values():
                # Make sure we're only dealing with prismatic / revolute joints
                assert joint.n_dof == 1, "Can only get task potential of prismatic / revolute joints!"
                # Potential is scaled value of the joint's position
                scale = self._r_prismatic if joint.joint_type == "PrismaticJoint" else self._r_revolute
                task_potential += scale * joint.get_state(normalized=True)[0][0]

        return task_potential

    def _reset_scene(self, env):
        # Reset the scene normally
        env.scene.reset_scene_objects()
        # Make sure all objects are awake
        env.scene.wake_scene_objects()
        # Sample opening objects and grab their references
        opened_objects = []
        for category in self._categories:
            opened_objects += env.scene.open_all_objs_by_category(category=category, mode="random", p=self._p_open)
        self._opened_objects = opened_objects

    def _sample_initial_pose(self, env):

        _, initial_pos = env.scene.get_random_point(floor=self._floor)
        initial_ori = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
        return initial_pos, initial_ori

    def _reset_agent(self, env):
        # We attempt to sample valid initial poses and goal positions
        success, max_trials = False, 100

        # Store the state of the environment now, so that we can restore it after each setting attempt
        state = env.dump_state(serialized=True)

        success, initial_pos, initial_ori = False, None, None
        for i in range(max_trials):
            initial_pos, initial_ori = self._sample_initial_pose(env)
            # Make sure the sampled robot start pose and goal position are both collision-free
            success = env.test_valid_position(env.robots[self._robot_idn], initial_pos, initial_ori)

            # Load the original state
            env.load_state(state=state, serialized=True)

            # Don't need to continue iterating if we succeeded
            if success:
                break

        # Notify user if we failed to reset a collision-free sampled pose
        if not success:
            logging.warning("WARNING: Failed to reset robot without collision")

        # Land the robot
        env.land(env.robots[self._robot_idn], initial_pos, initial_ori)

    def _get_obs(self, env):
        # No task-specific obs of any kind
        return OrderedDict(), OrderedDict()

    def _load_non_low_dim_observation_space(self):
        # No non-low dim observations so we return an empty dict
        return OrderedDict()

    @classproperty
    def valid_scene_types(cls):
        # Must be an interactive traversable scene
        return {InteractiveTraversableScene}

    @classproperty
    def default_termination_config(cls):
        return {
            "max_collisions": 500,
            "max_steps": 500,
            "fall_height": 0.03,
        }

    @classproperty
    def default_reward_config(cls):
        return {
            "r_prismatic": 1.0,
            "r_revolute": 1.0,
            "r_potential": 1.0,
        }

