import logging

import numpy as np


from igibson.reward_functions.potential_reward import PotentialReward
from igibson.scenes.interactive_traversable_scene import InteractiveIndoorScene
from igibson.tasks.task_base import BaseTask
from igibson.termination_conditions.max_collision import MaxCollision
from igibson.termination_conditions.out_of_bound import OutOfBound
from igibson.termination_conditions.timeout import Timeout
from igibson.utils.utils import restoreState


class RoomRearrangementTask(BaseTask):
    """
    Room Rearrangement Task
    The goal is to close as many furniture (e.g. cabinets and fridges) as possible
    """

    def __init__(self, env):
        super(RoomRearrangementTask, self).__init__(env)
        assert isinstance(
            env.scene, InteractiveIndoorScene
        ), "room rearrangement can only be done in InteractiveIndoorScene"
        self.prismatic_joint_reward_scale = self.config.get("prismatic_joint_reward_scale", 1.0)
        self.revolute_joint_reward_scale = self.config.get("revolute_joint_reward_scale", 1.0)
        self.termination_conditions = [
            MaxCollision(self.config),
            Timeout(self.config),
            OutOfBound(self.config),
        ]
        self.reward_functions = [
            PotentialReward(self.config),
        ]
        self.floor_num = 0

    def get_potential(self, env):
        """
        Compute task-specific potential: furniture joint positions

        :param env: environment instance
        :param: task potential
        """
        task_potential = 0.0
        for (body_id, joint_id) in self.body_joint_pairs:
            j_type = p.getJointInfo(body_id, joint_id)[2]
            j_pos = p.getJointState(body_id, joint_id)[0]
            scale = (
                self.prismatic_joint_reward_scale if j_type == p.JOINT_PRISMATIC else self.revolute_joint_reward_scale
            )
            task_potential += scale * j_pos
        return task_potential

    def reset_scene(self, env):
        """
        Reset all scene objects and then open certain object categories of interest.

        :param env: environment instance
        """
        env.scene.reset_scene_objects()
        env.scene.force_wakeup_scene_objects()
        self.body_joint_pairs = env.scene.open_all_objs_by_categories(
            [
                "bottom_cabinet",
                "bottom_cabinet_no_top",
                "top_cabinet",
                "dishwasher",
                "fridge",
                "microwave",
                "oven",
                "washer" "dryer",
            ],
            mode="random",
            prob=0.5,
        )

    def sample_initial_pose(self, env):
        """
        Sample robot initial pose

        :param env: environment instance
        :return: initial pose
        """
        _, initial_pos = env.scene.get_random_point(floor=self.floor_num)
        initial_orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
        return initial_pos, initial_orn

    def reset_agent(self, env):
        """
        Reset robot initial pose.
        Sample initial pose, check validity, and land it.

        :param env: environment instance
        """
        reset_success = False
        max_trials = 100

        # cache pybullet state
        # TODO: p.saveState takes a few seconds, need to speed up
        state_id = p.saveState()
        for _ in range(max_trials):
            initial_pos, initial_orn = self.sample_initial_pose(env)
            reset_success = env.test_valid_position(env.robots[0], initial_pos, initial_orn)
            restoreState(state_id)
            if reset_success:
                break

        if not reset_success:
            logging.warning("WARNING: Failed to reset robot without collision")

        env.land(env.robots[0], initial_pos, initial_orn)
        p.removeState(state_id)

    def get_task_obs(self, env):
        """
        No task-specific observation
        """
        return
