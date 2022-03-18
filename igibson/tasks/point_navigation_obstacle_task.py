import numpy as np

from igibson import ig_dataset_path
from igibson.objects.dataset_object import DatasetObject
from igibson.robots.turtlebot import Turtlebot
from igibson.tasks.point_navigation_task import PointNavigationTask
from igibson.utils.python_utils import classproperty


class PointNavigationObstacleTask(PointNavigationTask):
    """
    Interactive Navigation Random Task
    The goal is to navigate to a random goal position, in the presence of interactive objects. This is an abstract class

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
        n_obstacles (int): Number of obstacles to generate
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
            goal_tolerance=0.1,
            goal_in_polar=False,
            path_range=None,
            visualize_goal=False,
            visualize_path=False,
            marker_height=0.2,
            waypoint_width=0.1,
            n_vis_waypoints=250,
            reward_type="l2",
            n_obstacles=5,
            reward_config=None,
            termination_config=None,
    ):
        # Store inputs
        self._n_obstacles = n_obstacles

        # Initialize other variables that will be filled in at runtime
        self._obstacles = None

        # Run super init
        super().__init__(
            robot_idn=robot_idn,
            floor=floor,
            initial_pos=initial_pos,
            initial_ori=initial_ori,
            goal_pos=goal_pos,
            goal_tolerance=goal_tolerance,
            goal_in_polar=goal_in_polar,
            path_range=path_range,
            visualize_goal=visualize_goal,
            visualize_path=visualize_path,
            marker_height=marker_height,
            waypoint_width=waypoint_width,
            n_vis_waypoints=n_vis_waypoints,
            reward_type=reward_type,
            reward_config=reward_config,
            termination_config=termination_config,
        )

    def _load(self, env):
        # Load the interactive objects
        self._obstacles = self._load_obstacles(env=env)

    def _load_obstacles(self, env):
        """
        Load obstacles. Must be implemented by subclass.

        Args:
            env (iGibsonEnv): Environment instance

        Returns:
            list of BaseObject: Obstacle(s) generated for this task
        """
        raise NotImplementedError()

    def _reset_obstacles(self, env):
        """
        Reset the poses of obstacles to have no collisions with the scene or the robot

        Args:
            env (iGibsonEnv): Environment instance
        """
        success, max_trials, pos, ori = False, 100, None, None

        for obj in self._obstacles:
            # Save the state of this environment so we can restore it immediately after
            state = env.dump_state(serialized=True)
            for _ in range(max_trials):
                _, pos = env.scene.get_random_point(floor=self._floor)
                ori = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
                success = env.test_valid_position(obj, pos, ori)
                env.load_state(state=state, serialized=True)
                if success:
                    break

            if not success:
                print("WARNING: Failed to reset interactive obj without collision")

            env.land(obj, pos, ori)

    def _reset_scene(self, env):
        # Run super first
        super()._reset_scene(env=env)

        # Reset the obstacles
        self._reset_obstacles(env=env)

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("PointNavigationObstacleTask")
        return classes


class PointNavigationStaticObstacleTask(PointNavigationObstacleTask):
    # TODO: Update YCB object locations in iGibson
    # Define the obstacles for this class -- tuples are (category, model)
    STATIC_OBSTACLES = {
        ("canned_food", "002_master_chef_can"),
        ("cracker_box", "003_cracker_box"),
        ("sugar_box", "004_sugar_box"),
        ("canned_food", "005_tomato_soup_can"),
        ("bottled_food", "006_mustard_bottle"),
    }

    def _load_obstacles(self, env):
        # TODO: Probably need to make tasks Serializable so that we can save which objects were sampled
        # Sample requested number of objects
        obstacle_choices = list(self.STATIC_OBSTACLES)
        sampled_obstacle_ids = np.random.randint(0, len(obstacle_choices), self._n_obstacles)
        obstacles = []
        for i, obstacle_id in enumerate(sampled_obstacle_ids):
            # Create object
            o_category, o_model = obstacle_choices[obstacle_id]
            obstacle = DatasetObject(
                prim_path=f"/World/task_obstacle{i}",
                usd_path=f"{ig_dataset_path}/objects/{o_category}/{o_model}/{o_model}.usd",
                name=f"task_obstacle{i}",
            )
            # Import into the simulator, add to the ignore collisions, and store internally
            env.simulator.import_object(obj=obstacle)
            env.add_ignore_robot_object_collision(robot_idn=self._robot_idn, obj=obstacle)
            obstacles.append(obstacle)

        return obstacles


class PointNavigationDynamicObstacleTask(PointNavigationObstacleTask):
    """
    Dynamic Navigation Random Task
    The goal is to navigate to a random goal position, in the presence of dynamic objects (moving turtlebots)

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
        n_obstacles (int): Number of dynamic obstacles to generate
        n_obstacle_action_repeat (int): How many timesteps a dynamic obstacle should repeat its action before switching
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
            goal_tolerance=0.1,
            goal_in_polar=False,
            path_range=None,
            visualize_goal=False,
            visualize_path=False,
            marker_height=0.2,
            waypoint_width=0.1,
            n_vis_waypoints=250,
            reward_type="l2",
            n_obstacles=1,
            n_obstacle_action_repeat=10,
            reward_config=None,
            termination_config=None,
    ):
        # Store inputs
        self._n_obstacle_action_repeat = n_obstacle_action_repeat

        # Initialize variables that will be filled in at runtime
        self._current_obstacle_actions = None

        # Run super init
        super().__init__(
            robot_idn=robot_idn,
            floor=floor,
            initial_pos=initial_pos,
            initial_ori=initial_ori,
            goal_pos=goal_pos,
            goal_tolerance=goal_tolerance,
            goal_in_polar=goal_in_polar,
            path_range=path_range,
            visualize_goal=visualize_goal,
            visualize_path=visualize_path,
            marker_height=marker_height,
            waypoint_width=waypoint_width,
            n_vis_waypoints=n_vis_waypoints,
            reward_type=reward_type,
            n_obstacles=n_obstacles,
            reward_config=reward_config,
            termination_config=termination_config,
        )

    def _load_obstacles(self, env):
        # Load turtlebots
        obstacles = []
        for i in range(self._n_obstacles):
            obstacle = Turtlebot(
                prim_path=f"/World/task_obstacle{i}",
                name=f"task_obscale{i}",
            )
            env.simulator.import_object(obstacle)
            obstacles.append(obstacle)

        return obstacles

    def step(self, env, action):
        # Run super method first
        reward, done, info = super().step(env=env, action=action)
        # Apply actions for each dynamic obstacle
        if env.current_step % self._n_obstacle_action_repeat == 0:
            self._current_obstacle_actions = [robot.action_space.sample() for robot in self._obstacles]
        for robot, action in zip(self._obstacles, self._current_obstacle_actions):
            robot.apply_action(action)

        return reward, done, info
