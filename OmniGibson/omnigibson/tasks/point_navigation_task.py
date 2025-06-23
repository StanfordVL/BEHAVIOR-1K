import math

import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.object_states import Pose
from omnigibson.objects.primitive_object import PrimitiveObject
from omnigibson.reward_functions.collision_reward import CollisionReward
from omnigibson.reward_functions.point_goal_reward import PointGoalReward
from omnigibson.reward_functions.potential_reward import PotentialReward
from omnigibson.scenes.traversable_scene import TraversableScene
from omnigibson.tasks.task_base import BaseTask
from omnigibson.termination_conditions.falling import Falling
from omnigibson.termination_conditions.max_collision import MaxCollision
from omnigibson.termination_conditions.point_goal import PointGoal
from omnigibson.termination_conditions.timeout import Timeout
from omnigibson.utils.python_utils import assert_valid_key, classproperty
from omnigibson.utils.sim_utils import land_object, test_valid_pose
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)


# Valid point navigation reward types
POINT_NAVIGATION_REWARD_TYPES = {"l2", "geodesic"}


class PointNavigationTask(BaseTask):
    """
    Point Navigation Task
    The task is to navigate to a goal position

    Args:
        robot_idn (int): Which robot that this task corresponds to
        floor (int): Which floor to navigate on
        initial_pos (None or 3-array): If specified, should be (x,y,z) global initial position to place the robot
            at the start of each task episode. If None, a collision-free value will be randomly sampled
        initial_quat (None or 4-array): If specified, should be (x,y,z,w) global quaternion orientation to place the
            robot at the start of each task episode. If None, a value will be randomly sampled about the z-axis
        goal_pos (None or 3-array): If specified, should be (x,y,z) global goal position to reach for the given task
            episode. If None, a collision-free value will be randomly sampled
        goal_tolerance (float): Distance between goal position and current position below which is considered a task
            success
        goal_in_polar (bool): Whether to represent the goal in polar coordinates or not when capturing task observations
        path_range (None or 2-array): If specified, should be (min, max) values representing the range of valid
            total path lengths that are valid when sampling initial / goal positions
        visualize_goal (bool): Whether to visualize the initial / goal locations
        visualize_path (bool): Whether to visualize the path from initial to goal location, as represented by
            discrete waypoints
        goal_height (float): If visualizing, specifies the height of the visual goals (m)
        waypoint_height (float): If visualizing, specifies the height of the visual waypoints (m)
        waypoint_width (float): If visualizing, specifies the width of the visual waypoints (m)
        n_vis_waypoints (int): If visualizing, specifies the number of waypoints to generate
        reward_type (str): Type of reward to use. Valid options are: {"l2", "geodesic"}
        termination_config (None or dict): Keyword-mapped configuration to use to generate termination conditions. This
            should be specific to the task class. Default is None, which corresponds to a default config being usd.
            Note that any keyword required by a specific task class but not specified in the config will automatically
            be filled in with the default config. See cls.default_termination_config for default values used
        reward_config (None or dict): Keyword-mapped configuration to use to generate reward functions. This should be
            specific to the task class. Default is None, which corresponds to a default config being usd. Note that
            any keyword required by a specific task class but not specified in the config will automatically be filled
            in with the default config. See cls.default_reward_config for default values used
        include_obs (bool): Whether to include observations or not for this task
    """

    def __init__(
        self,
        robot_idn=0,
        floor=0,
        initial_pos=None,
        initial_quat=None,
        goal_pos=None,
        goal_tolerance=0.5,
        goal_in_polar=False,
        path_range=None,
        visualize_goal=False,
        visualize_path=False,
        goal_height=0.06,
        waypoint_height=0.05,
        waypoint_width=0.1,
        n_vis_waypoints=10,
        reward_type="l2",
        termination_config=None,
        reward_config=None,
        include_obs=True,
    ):
        # Store inputs
        self._robot_idn = robot_idn
        self._floor = floor
        self._initial_pos = initial_pos if initial_pos is None else th.tensor(initial_pos)
        self._initial_quat = initial_quat if initial_quat is None else th.tensor(initial_quat)
        self._goal_pos = goal_pos if goal_pos is None else th.tensor(goal_pos)
        self._goal_tolerance = goal_tolerance
        self._goal_in_polar = goal_in_polar
        self._path_range = path_range
        self._randomize_initial_pos = initial_pos is None
        self._randomize_initial_quat = initial_quat is None
        self._randomize_goal_pos = goal_pos is None
        self._visualize_goal = visualize_goal
        self._visualize_path = visualize_path
        self._goal_height = goal_height
        self._waypoint_height = waypoint_height
        self._waypoint_width = waypoint_width
        self._n_vis_waypoints = n_vis_waypoints
        assert_valid_key(key=reward_type, valid_keys=POINT_NAVIGATION_REWARD_TYPES, name="reward type")
        self._reward_type = reward_type

        # Create other attributes that will be filled in at runtime
        self._initial_pos_marker = None
        self._goal_pos_marker = None
        self._waypoint_markers = None
        self._path_length = None
        self._current_robot_pos = None
        self._geodesic_dist = None

        # Run super
        super().__init__(termination_config=termination_config, reward_config=reward_config, include_obs=include_obs)

    def _create_termination_conditions(self):
        # Initialize termination conditions dict and fill in with MaxCollision, Timeout, Falling, and PointGoal
        terminations = dict()
        terminations["max_collision"] = MaxCollision(max_collisions=self._termination_config["max_collisions"])
        terminations["timeout"] = Timeout(max_steps=self._termination_config["max_steps"])
        terminations["falling"] = Falling(
            robot_idn=self._robot_idn, fall_height=self._termination_config["fall_height"]
        )
        terminations["pointgoal"] = PointGoal(
            robot_idn=self._robot_idn,
            distance_tol=self._goal_tolerance,
            distance_axes="xy",
        )

        return terminations

    def _create_reward_functions(self):
        # Initialize reward functions dict and fill in with Potential, Collision, and PointGoal rewards
        rewards = dict()

        rewards["potential"] = PotentialReward(
            potential_fcn=self.get_potential,
            r_potential=self._reward_config["r_potential"],
        )
        rewards["collision"] = CollisionReward(r_collision=self._reward_config["r_collision"])
        rewards["pointgoal"] = PointGoalReward(
            pointgoal=self._termination_conditions["pointgoal"],
            r_pointgoal=self._reward_config["r_pointgoal"],
        )

        return rewards

    def _load(self, env):
        # Load visualization
        self._load_visualization_markers(env=env)

        # Auto-initialize all markers
        og.sim.play()
        self._reset_agent(env=env)
        env.scene.update_initial_file()
        og.sim.stop()

    def _load_visualization_markers(self, env):
        """
        Load visualization, such as initial and target position, shortest path, etc

        Args:
            env (Environment): Active environment instance
        """
        if self._visualize_goal:
            self._initial_pos_marker = PrimitiveObject(
                relative_prim_path="/task_initial_pos_marker",
                primitive_type="Cylinder",
                name="task_initial_pos_marker",
                radius=self._goal_tolerance,
                height=self._goal_height,
                visual_only=True,
                rgba=th.tensor([1, 0, 0, 0.3]),
            )
            self._goal_pos_marker = PrimitiveObject(
                relative_prim_path="/task_goal_pos_marker",
                primitive_type="Cylinder",
                name="task_goal_pos_marker",
                radius=self._goal_tolerance,
                height=self._goal_height,
                visual_only=True,
                rgba=th.tensor([0, 0, 1, 0.3]),
            )

            # Load the objects into the simulator
            env.scene.add_object(self._initial_pos_marker)
            env.scene.add_object(self._goal_pos_marker)

        # Additionally generate waypoints along the path if we're building the map in the environment
        if self._visualize_path:
            waypoints = []
            for i in range(self._n_vis_waypoints):
                waypoint = PrimitiveObject(
                    relative_prim_path=f"/task_waypoint_marker{i}",
                    primitive_type="Cylinder",
                    name=f"task_waypoint_marker{i}",
                    radius=self._waypoint_width,
                    height=self._waypoint_height,
                    visual_only=True,
                    rgba=th.tensor([0, 1, 0, 0.3]),
                )
                env.scene.add_object(waypoint)
                waypoints.append(waypoint)

            # Store waypoints
            self._waypoint_markers = waypoints

    def _sample_initial_pose_and_goal_pos(self, env, max_trials=100):
        """
        Potentially sample the robot initial pos / ori and target pos, based on whether we're using randomized
        initial and goal states. If not randomzied, then this value will return the corresponding values inputted
        during this task initialization.

        Args:
            env (Environment): Environment instance
            max_trials (int): Number of trials to attempt to sample valid poses and positions

        Returns:
            3-tuple:
                - 3-array: (x,y,z) global sampled initial position
                - 4-array: (x,y,z,w) global sampled initial orientation in quaternion form
                - 3-array: (x,y,z) global sampled goal position
        """
        # Possibly sample initial pos
        if self._randomize_initial_pos:
            _, initial_pos = env.scene.get_random_point(floor=self._floor, robot=env.robots[self._robot_idn])
        else:
            initial_pos = self._initial_pos

        # Possibly sample initial ori
        quat_lo, quat_hi = 0, math.pi * 2
        initial_quat = (
            T.euler2quat(th.tensor([0, 0, (th.rand(1) * (quat_hi - quat_lo) + quat_lo).item()]))
            if self._randomize_initial_quat
            else self._initial_quat
        )

        # Possibly sample goal pos
        if self._randomize_goal_pos:
            dist, in_range_dist = 0.0, False
            for _ in range(max_trials):
                _, goal_pos = env.scene.get_random_point(
                    floor=self._floor, reference_point=initial_pos, robot=env.robots[self._robot_idn]
                )
                _, dist = env.scene.get_shortest_path(
                    self._floor, initial_pos[:2], goal_pos[:2], entire_path=False, robot=env.robots[self._robot_idn]
                )
                # If a path range is specified, make sure distance is valid
                if dist is not None and (self._path_range is None or self._path_range[0] < dist < self._path_range[1]):
                    in_range_dist = True
                    break
            # Notify if we weren't able to get a valid start / end point sampled in the requested range
            if not in_range_dist:
                log.warning("Failed to sample initial and target positions within requested path range")
        else:
            goal_pos = self._goal_pos

        # Add additional logging info
        log.info("Sampled initial pose: {}, {}".format(initial_pos, initial_quat))
        log.info("Sampled goal position: {}".format(goal_pos))
        return initial_pos, initial_quat, goal_pos

    def _get_geodesic_potential(self, env):
        """
        Get potential based on geodesic distance

        Args:
            env: environment instance

        Returns:
            float: geodesic distance to the target position
        """
        _, geodesic_dist = self.get_shortest_path_to_goal(env=env)
        return geodesic_dist

    def _get_l2_potential(self, env):
        """
        Get potential based on L2 distance

        Args:
            env: environment instance

        Returns:
            float: L2 distance to the target position
        """
        return T.l2_distance(env.robots[self._robot_idn].states[Pose].get_value()[0][:2], self._goal_pos[:2])

    def get_potential(self, env):
        """
        Compute task-specific potential: distance to the goal

        Args:
            env (Environment): Environment instance

        Returns:
            float: Computed potential
        """
        if self._reward_type == "l2":
            potential = self._get_l2_potential(env)
        elif self._reward_type == "geodesic":
            potential = self._get_geodesic_potential(env)
            # If no path is found, fall back to L2 potential
            if potential is None:
                potential = self._get_l2_potential(env)
        else:
            raise ValueError(f"Invalid reward type! {self._reward_type}")

        return potential

    def _reset_agent(self, env):
        # Reset agent
        env.robots[self._robot_idn].reset()

        # We attempt to sample valid initial poses and goal positions
        success, max_trials = False, 100

        initial_pos, initial_quat, goal_pos = None, None, None
        for i in range(max_trials):
            initial_pos, initial_quat, goal_pos = self._sample_initial_pose_and_goal_pos(env)
            # Make sure the sampled robot start pose and goal position are both collision-free
            success = test_valid_pose(
                env.robots[self._robot_idn], initial_pos, initial_quat, env.initial_pos_z_offset
            ) and test_valid_pose(env.robots[self._robot_idn], goal_pos, None, env.initial_pos_z_offset)

            # Don't need to continue iterating if we succeeded
            if success:
                break

        # Notify user if we failed to reset a collision-free sampled pose
        if not success:
            log.warning("Failed to reset robot without collision")

        # Land the robot
        land_object(env.robots[self._robot_idn], initial_pos, initial_quat, env.initial_pos_z_offset)

        # Store the sampled values internally
        self._initial_pos = initial_pos
        self._initial_quat = initial_quat
        self._goal_pos = goal_pos

        # Update visuals if requested
        if self._visualize_goal:
            self._initial_pos_marker.set_position_orientation(position=self._initial_pos)
            self._goal_pos_marker.set_position_orientation(position=self._goal_pos)

    def _reset_variables(self, env):
        # Run super first
        super()._reset_variables(env=env)

        # Reset internal variables
        self._path_length = 0.0
        self._current_robot_pos = self._initial_pos
        self._geodesic_dist = self._get_geodesic_potential(env)

    def _step_termination(self, env, action, info=None):
        # Run super first
        done, info = super()._step_termination(env=env, action=action, info=info)

        # Add additional info
        info["path_length"] = self._path_length
        info["spl"] = (
            float(info["success"]) * min(1.0, self._geodesic_dist / self._path_length)
            if done and self._path_length != 0.0
            else 0.0
        )

        return done, info

    def _global_pos_to_robot_frame(self, env, pos):
        """
        Convert a 3D point in global frame to agent's local frame

        Args:
            env (TraversableEnv): Environment instance
            pos (th.Tensor): global (x,y,z) position

        Returns:
            th.Tensor: (x,y,z) position in self._robot_idn agent's local frame
        """
        delta_pos_global = pos - env.robots[self._robot_idn].states[Pose].get_value()[0]
        return T.quat2mat(env.robots[self._robot_idn].states[Pose].get_value()[1]).T @ delta_pos_global

    def _get_obs(self, env):
        # Get relative position of goal with respect to the current agent position
        xy_pos_to_goal = self._global_pos_to_robot_frame(env, self._goal_pos)[:2]
        if self._goal_in_polar:
            xy_pos_to_goal = th.tensor(T.cartesian_to_polar(*xy_pos_to_goal))

        # linear velocity and angular velocity
        ori_t = T.quat2mat(env.robots[self._robot_idn].states[Pose].get_value()[1]).T
        lin_vel = ori_t @ env.robots[self._robot_idn].get_linear_velocity()
        ang_vel = ori_t @ env.robots[self._robot_idn].get_angular_velocity()

        # Compose observation dict
        low_dim_obs = dict(
            xy_pos_to_goal=xy_pos_to_goal,
            robot_lin_vel=lin_vel,
            robot_ang_vel=ang_vel,
        )

        # We have no non-low-dim obs, so return empty dict for those
        return low_dim_obs, dict()

    def _load_non_low_dim_observation_space(self):
        # No non-low dim observations so we return an empty dict
        return dict()

    def get_goal_pos(self):
        """
        Returns:
            3-array: (x,y,z) global current goal position
        """
        return self._goal_pos

    def get_current_pos(self, env):
        """
        Returns:
            3-array: (x,y,z) global current position representing the robot
        """
        return env.robots[self._robot_idn].states[Pose].get_value()[0]

    def get_shortest_path_to_goal(self, env, start_xy_pos=None, entire_path=False):
        """
        Get the shortest path and geodesic distance from @start_pos to the target position

        Args:
            env (TraversableEnv): Environment instance
            start_xy_pos (None or 2-array): If specified, should be the global (x,y) start position from which
                to calculate the shortest path to the goal position. If None (default), the robot's current xy position
                will be used
            entire_path (bool): Whether to return the entire shortest path

        Returns:
            2-tuple:
                - list of 2-array: List of (x,y) waypoints representing the path # TODO: is this true?
                - float: geodesic distance of the path to the goal position
        """
        start_xy_pos = (
            env.robots[self._robot_idn].states[Pose].get_value()[0][:2] if start_xy_pos is None else start_xy_pos
        )
        return env.scene.get_shortest_path(
            self._floor, start_xy_pos, self._goal_pos[:2], entire_path=entire_path, robot=env.robots[self._robot_idn]
        )

    def _step_visualization(self, env):
        """
        Step visualization

        Args:
            env (Environment): Environment instance
        """
        if self._visualize_path:
            shortest_path, _ = self.get_shortest_path_to_goal(env=env, entire_path=True)
            floor_height = env.scene.get_floor_height(self._floor)
            num_nodes = min(self._n_vis_waypoints, shortest_path.shape[0])
            for i in range(num_nodes):
                self._waypoint_markers[i].set_position_orientation(
                    position=th.tensor([shortest_path[i][0], shortest_path[i][1], floor_height])
                )
            for i in range(num_nodes, self._n_vis_waypoints):
                self._waypoint_markers[i].set_position_orientation(position=th.tensor([0.0, 0.0, 100.0]))

    def step(self, env, action):
        # Run super method first
        reward, done, info = super().step(env=env, action=action)

        # Step visualization
        self._step_visualization(env=env)

        # Update other internal variables
        new_robot_pos = env.robots[self._robot_idn].states[Pose].get_value()[0]
        self._path_length += T.l2_distance(self._current_robot_pos[:2], new_robot_pos[:2])
        self._current_robot_pos = new_robot_pos

        return reward, done, info

    @classproperty
    def valid_scene_types(cls):
        # Must be a traversable scene
        return {TraversableScene}

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
            "r_potential": 1.0,
            "r_collision": 0.1,
            "r_pointgoal": 10.0,
        }
