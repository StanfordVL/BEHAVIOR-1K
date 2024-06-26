import torch as th

import omnigibson.utils.transform_utils as T
from omnigibson.tasks.point_navigation_task import PointNavigationTask
from omnigibson.termination_conditions.point_goal import PointGoal

# Valid point navigation reward types
POINT_NAVIGATION_REWARD_TYPES = {"l2", "geodesic"}


class PointReachingTask(PointNavigationTask):
    """
    Point Reaching Task
    The goal is to reach a random goal position with the robot's end effector

    Args:
        robot_idn (int): Which robot that this task corresponds to
        floor (int): Which floor to navigate on
        initial_pos (None or 3-array): If specified, should be (x,y,z) global initial position to place the robot
            at the start of each task episode. If None, a collision-free value will be randomly sampled
        initial_quat (None or 3-array): If specified, should be (r,p,y) global euler orientation to place the robot
            at the start of each task episode. If None, a value will be randomly sampled about the z-axis
        goal_pos (None or 3-array): If specified, should be (x,y,z) global goal position to reach for the given task
            episode. If None, a collision-free value will be randomly sampled
        goal_tolerance (float): Distance between goal position and current position below which is considered a task
            success
        goal_in_polar (bool): Whether to represent the goal in polar coordinates or not when capturing task observations
        path_range (None or 2-array): If specified, should be (min, max) values representing the range of valid
            total path lengths that are valid when sampling initial / goal positions
        height_range (None or 2-array): If specified, should be (min, max) values representing the range of valid
            total heights that are valid when sampling goal positions
        visualize_goal (bool): Whether to visualize the initial / goal locations
        visualize_path (bool): Whether to visualize the path from initial to goal location, as represented by
            discrete waypoints
        goal_height (float): If visualizing, specifies the height of the visual goals (m)
        waypoint_height (float): If visualizing, specifies the height of the visual waypoints (m)
        waypoint_width (float): If visualizing, specifies the width of the visual waypoints (m)
        n_vis_waypoints (int): If visualizing, specifies the number of waypoints to generate
        termination_config (None or dict): Keyword-mapped configuration to use to generate termination conditions. This
            should be specific to the task class. Default is None, which corresponds to a default config being usd.
            Note that any keyword required by a specific task class but not specified in the config will automatically
            be filled in with the default config. See cls.default_termination_config for default values used
        reward_config (None or dict): Keyword-mapped configuration to use to generate reward functions. This should be
            specific to the task class. Default is None, which corresponds to a default config being usd. Note that
            any keyword required by a specific task class but not specified in the config will automatically be filled
            in with the default config. See cls.default_reward_config for default values used
    """

    def __init__(
        self,
        robot_idn=0,
        floor=0,
        initial_pos=None,
        initial_quat=None,
        goal_pos=None,
        goal_tolerance=0.1,
        goal_in_polar=False,
        path_range=None,
        height_range=None,
        visualize_goal=False,
        visualize_path=False,
        goal_height=0.06,
        waypoint_height=0.05,
        waypoint_width=0.1,
        n_vis_waypoints=10,
        reward_config=None,
        termination_config=None,
    ):
        # Store inputs
        self._height_range = height_range

        # Run super
        super().__init__(
            robot_idn=robot_idn,
            floor=floor,
            initial_pos=initial_pos,
            initial_quat=initial_quat,
            goal_pos=goal_pos,
            goal_tolerance=goal_tolerance,
            goal_in_polar=goal_in_polar,
            path_range=path_range,
            visualize_goal=visualize_goal,
            visualize_path=visualize_path,
            goal_height=goal_height,
            waypoint_height=waypoint_height,
            waypoint_width=waypoint_width,
            n_vis_waypoints=n_vis_waypoints,
            reward_type="l2",  # Must use l2 for reaching task
            reward_config=reward_config,
            termination_config=termination_config,
        )

    def _create_termination_conditions(self):
        # Run super first
        terminations = super()._create_termination_conditions()

        # We replace the pointgoal condition with a new one, specifying xyz instead of only xy as the axes to measure
        # distance to the goal
        terminations["pointgoal"] = PointGoal(
            robot_idn=self._robot_idn,
            distance_tol=self._goal_tolerance,
            distance_axes="xyz",
        )

        return terminations

    def _sample_initial_pose_and_goal_pos(self, env, max_trials=100):
        # Run super first
        initial_pos, initial_ori, goal_pos = super()._sample_initial_pose_and_goal_pos(env=env, max_trials=max_trials)

        # Sample goal position to be within requested height range if specified
        if self._height_range is not None:
            goal_pos[2] += th.rand(1) * (self._height_range[1] - self._height_range[0]) + self._height_range[0]

        return initial_pos, initial_ori, goal_pos

    def _get_l2_potential(self, env):
        # Distance calculated from robot EEF, not base!
        return T.l2_distance(env.robots[self._robot_idn].get_eef_position(), self._goal_pos)

    def _get_obs(self, env):
        # Get obs from super
        low_dim_obs, obs = super()._get_obs(env=env)

        # Remove xy-pos and replace with full xyz relative distance between current and goal pos
        low_dim_obs.pop("xy_pos_to_goal")
        low_dim_obs["eef_to_goal"] = self._global_pos_to_robot_frame(env=env, pos=self._goal_pos)

        # Add local eef position as well
        low_dim_obs["eef_local_pos"] = self._global_pos_to_robot_frame(
            env=env, pos=env.robots[self._robot_idn].get_eef_position()
        )

        return low_dim_obs, obs

    def get_current_pos(self, env):
        # Current position is the robot's EEF, not base!
        return env.robots[self._robot_idn].get_eef_position()
