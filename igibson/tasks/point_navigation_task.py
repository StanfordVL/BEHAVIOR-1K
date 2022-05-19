import numpy as np
from collections import OrderedDict
import logging

from igibson.objects.primitive_object import PrimitiveObject
from igibson.reward_functions.collision_reward import CollisionReward
from igibson.reward_functions.point_goal_reward import PointGoalReward
from igibson.reward_functions.potential_reward import PotentialReward
from igibson.scenes.traversable_scene import TraversableScene
from igibson.tasks.task_base import BaseTask
from igibson.termination_conditions.max_collision import MaxCollision
from igibson.termination_conditions.falling import Falling
from igibson.termination_conditions.point_goal import PointGoal
from igibson.termination_conditions.timeout import Timeout
from igibson.utils.utils import cartesian_to_polar, l2_distance, rotate_vector_3d
from igibson.utils.python_utils import classproperty, assert_valid_key


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
        initial_ori (None or 3-array): If specified, should be (r,p,y) global euler orientation to place the robot
            at the start of each task episode. If None, a value will be randomly sampled about the z-axis
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
        marker_height (float): If visualizing, specifies the height of the visual markers (m)
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
    """

    def __init__(
            self,
            robot_idn=0,
            floor=0,
            initial_pos=None,
            initial_ori=None,
            goal_pos=None,
            goal_tolerance=0.5,
            goal_in_polar=False,
            path_range=None,
            visualize_goal=False,
            visualize_path=False,
            marker_height=0.2,
            waypoint_width=0.1,
            n_vis_waypoints=250,
            reward_type="l2",
            termination_config=None,
            reward_config=None,
    ):
        # Store inputs
        self._robot_idn = robot_idn
        self._floor = floor
        self._initial_pos = initial_pos if initial_pos is None else np.array(initial_pos)
        self._initial_ori = initial_ori if initial_ori is None else np.array(initial_ori)
        self._goal_pos = goal_pos if goal_pos is None else np.array(goal_pos)
        self._goal_tolerance = goal_tolerance
        self._goal_in_polar = goal_in_polar
        self._path_range = path_range
        self._randomize_initial_pos = initial_pos is None
        self._randomize_initial_ori = initial_ori is None
        self._randomize_goal_pos = goal_pos is None
        self._visualize_goal = visualize_goal
        self._visualize_path = visualize_path
        self._marker_height = marker_height
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
        super().__init__(termination_config=termination_config, reward_config=reward_config)

    def _create_termination_conditions(self):
        # Initialize termination conditions dict and fill in with MaxCollision, Timeout, Falling, and PointGoal
        terminations = OrderedDict()
        terminations["max_collision"] = MaxCollision(max_collisions=self._termination_config["max_collisions"])
        terminations["timeout"] = Timeout(max_steps=self._termination_config["max_steps"])
        terminations["falling"] = Falling(robot_idn=self._robot_idn, fall_height=self._termination_config["fall_height"])
        terminations["pointgoal"] = PointGoal(
            robot_idn=self._robot_idn,
            distance_tol=self._goal_tolerance,
            distance_axes="xy",
        )

        return terminations

    def _create_reward_functions(self):
        # Initialize reward functions dict and fill in with Potential, Collision, and PointGoal rewards
        rewards = OrderedDict()

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

    def _load_visualization_markers(self, env):
        """
        Load visualization, such as initial and target position, shortest path, etc

        Args:
            env (BaseEnv): Active environment instance
        """
        cyl_size = np.array([self._goal_tolerance, self._goal_tolerance, self._marker_height])
        self._initial_pos_marker = PrimitiveObject(
            prim_path="/World/task_initial_pos_marker",
            primitive_type="Cylinder",
            name="task_initial_pos_marker",
            scale=cyl_size,
            visible=self._visualize_goal,
            visual_only=True,
            rgba=np.array([1, 0, 0, 0.3]),
        )
        self._goal_pos_marker = PrimitiveObject(
            prim_path="/World/task_goal_pos_marker",
            primitive_type="Cylinder",
            name="task_goal_pos_marker",
            scale=cyl_size,
            visible=self._visualize_goal,
            visual_only=True,
            rgba=np.array([0, 0, 1, 0.3]),
        )

        # Load the objects into the simulator
        env.simulator.import_object(self._initial_pos_marker)
        env.simulator.import_object(self._goal_pos_marker)

        # Additionally generate waypoints along the path if we're building the map in the environment
        if env.scene.trav_map.build_graph:
            waypoints = []
            waypoint_size = np.array([self._waypoint_width, self._waypoint_width, self._marker_height])
            for i in range(self._n_vis_waypoints):
                waypoint = PrimitiveObject(
                    prim_path=f"/World/task_waypoint_marker{i}",
                    primitive_type="Cylinder",
                    name=f"task_waypoint_marker{i}",
                    scale=waypoint_size,
                    visible=self._visualize_path,
                    visual_only=True,
                    rgba=np.array([0, 1, 0, 0.3]),
                )
                env.simulator.import_object(waypoint)
                waypoints.append(waypoint)

            # Store waypoints
            self._waypoint_markers = waypoints

    def _sample_initial_pose_and_goal_pos(self, env, max_trials=100):
        """
        Potentially sample the robot initial pos / ori and target pos, based on whether we're using randomized
        initial and goal states. If not randomzied, then this value will return the corresponding values inputted
        during this task initialization.

        Args:
            env (BaseEnv): Environment instance
            max_trials (int): Number of trials to attempt to sample valid poses and positions

        Returns:
            3-tuple:
                - 3-array: (x,y,z) global sampled initial position
                - 3-array: (r,p,y) global sampled initial orientation in euler form
                - 3-array: (x,y,z) global sampled goal position
        """
        # Possibly sample initial pos
        if self._randomize_initial_pos:
            _, initial_pos = env.scene.get_random_point(floor=self._floor)
        else:
            initial_pos = self._initial_pos

        # Possibly sample initial ori
        initial_ori = np.array([0, 0, np.random.uniform(0, np.pi * 2)]) if self._randomize_initial_ori else \
            self._initial_ori

        # Possibly sample goal pos
        if self._randomize_goal_pos:
            dist, in_range_dist = 0.0, False
            for _ in range(max_trials):
                _, goal_pos = env.scene.get_random_point(floor=self._floor)
                if env.scene.trav_map.build_graph:
                    _, dist = env.scene.get_shortest_path(self._floor, initial_pos[:2], goal_pos[:2], entire_path=False)
                else:
                    dist = l2_distance(initial_pos, goal_pos)
                # If a path range is specified, make sure distance is valid
                if self._path_range is None or self._path_range[0] < dist < self._path_range[1]:
                    in_range_dist = True
                    break
            # Notify if we weren't able to get a valid start / end point sampled in the requested range
            if not in_range_dist:
                logging.warning("Failed to sample initial and target positions within requested path range")
        else:
            goal_pos = self._goal_pos

        # Add additional logging info
        logging.info("Sampled initial pose: {}, {}".format(initial_pos, initial_ori))
        logging.info("Sampled goal position: {}".format(goal_pos))
        return initial_pos, initial_ori, goal_pos

    def _get_geodesic_potential(self, env):
        """
        Get potential based on geodesic distance

        :param env: environment instance
        :return: geodesic distance to the target position
        """
        _, geodesic_dist = self.get_shortest_path_to_goal(env=env)
        return geodesic_dist

    def _get_l2_potential(self, env):
        """
        Get potential based on L2 distance

        :param env: environment instance
        :return: L2 distance to the target position
        """
        return l2_distance(env.robots[self._robot_idn].get_position()[:2], self._goal_pos[:2])

    def get_potential(self, env):
        """
        Compute task-specific potential: distance to the goal

        Args:
            env (BaseEnv): Environment instance

        Returns:
            float: Computed potential
        """
        if self._reward_type == "l2":
            reward = self._get_l2_potential(env)
        elif self._reward_type == "geodesic":
            reward = self._get_geodesic_potential(env)
        else:
            raise ValueError(f"Invalid reward type! {self._reward_type}")

        return reward

    def _reset_agent(self, env):
        # Reset agent
        env.robots[self._robot_idn].reset()

        # We attempt to sample valid initial poses and goal positions
        success, max_trials = False, 100

        # Store the state of the environment now, so that we can restore it after each setting attempt
        state = env.dump_state(serialized=True)

        initial_pos, initial_ori, goal_pos = None, None, None
        for i in range(max_trials):
            initial_pos, initial_ori, goal_pos = self._sample_initial_pose_and_goal_pos(env)
            # Make sure the sampled robot start pose and goal position are both collision-free
            success = env.test_valid_position(
                env.robots[self._robot_idn], initial_pos, initial_ori
            ) and env.test_valid_position(env.robots[self._robot_idn], goal_pos)

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

        # Store the sampled values internally
        self._initial_pos = initial_pos
        self._initial_ori = initial_ori
        self._goal_pos = goal_pos

        # Update visuals if requested
        if self._visualize_goal:
            self._initial_pos_marker.set_position(self._initial_pos)
            self._goal_pos_marker.set_position(self._goal_pos)

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
        info["spl"] = float(info["success"]) * min(1.0, self._geodesic_dist / self._path_length) if done else 0.0

        return done, info

    def _global_pos_to_robot_frame(self, env, pos):
        """
        Convert a 3D point in global frame to agent's local frame

        Args:
            env (TraversableEnv): Environment instance
            pos (3-array): global (x,y,z) position

        Returns:
            3-array: (x,y,z) position in self._robot_idn agent's local frame

        :param env: environment instance
        :param pos: a 3D point in global frame
        :return: the same 3D point in agent's local frame
        """
        return rotate_vector_3d(np.array(pos) - np.array(env.robots[self._robot_idn].get_position()),
                                *env.robots[self._robot_idn].get_rpy())

    def _get_obs(self, env):
        # Get relative position of goal with respect to the current agent position
        xy_pos_to_goal = self._global_pos_to_robot_frame(env, self._goal_pos)[:2]
        if self._goal_in_polar:
            xy_pos_to_goal = np.array(cartesian_to_polar(*xy_pos_to_goal))

        # linear velocity along the x-axis
        linear_velocity = rotate_vector_3d(env.robots[self._robot_idn].get_linear_velocity(),
                                           *env.robots[self._robot_idn].get_rpy())[0]
        # angular velocity along the z-axis
        angular_velocity = rotate_vector_3d(env.robots[self._robot_idn].get_angular_velocity(),
                                            *env.robots[self._robot_idn].get_rpy())[2]

        # Compose observation dict
        low_dim_obs = OrderedDict(
            xy_pos_to_goal=xy_pos_to_goal,
            robot_lin_vel=np.array([linear_velocity]),
            robot_ang_vel=np.array([angular_velocity]),
        )

        # We have no non-low-dim obs, so return empty dict for those
        return low_dim_obs, OrderedDict()

    def _load_non_low_dim_observation_space(self):
        # No non-low dim observations so we return an empty dict
        return OrderedDict()

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
        return env.robots[self._robot_idn].get_position()

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
        start_xy_pos = env.robots[self._robot_idn].get_position()[:2] if start_xy_pos is None else start_xy_pos
        return env.scene.get_shortest_path(self._floor, start_xy_pos, self._goal_pos[:2], entire_path=entire_path)

    def _step_visualization(self, env):
        """
        Step visualization

        Args:
            env (BaseEnv): Environment instance
        """
        if env.scene.trav_map.build_graph and self._visualize_path:
            shortest_path, _ = self.get_shortest_path_to_goal(env=env, entire_path=True)
            floor_height = env.scene.get_floor_height(self._floor)
            num_nodes = min(self._n_vis_waypoints, shortest_path.shape[0])
            for i in range(num_nodes):
                self._waypoint_markers[i].set_position(
                    position=np.array([shortest_path[i][0], shortest_path[i][1], floor_height])
                )
            for i in range(num_nodes, self._n_vis_waypoints):
                self._waypoint_markers[i].set_position(position=np.array([0.0, 0.0, 100.0]))

    def step(self, env, action):
        # Run super method first
        reward, done, info = super().step(env=env, action=action)

        # Step visualization
        self._step_visualization(env=env)

        # Update other internal variables
        new_robot_pos = env.robots[self._robot_idn].get_position()
        self._path_length += l2_distance(self._current_robot_pos[:2], new_robot_pos[:2])
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
