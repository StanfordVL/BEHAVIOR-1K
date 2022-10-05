import gym
from collections import OrderedDict

import igibson as ig
from igibson.macros import gm, create_module_macros
from igibson.robots import REGISTERED_ROBOTS
from igibson.tasks import REGISTERED_TASKS
from igibson.scenes import REGISTERED_SCENES
from igibson.utils.gym_utils import GymObservable
from igibson.utils.sim_utils import get_collisions
from igibson.utils.config_utils import parse_config
from igibson.utils.python_utils import assert_valid_key, merge_nested_dicts, create_class_from_registry_and_config,\
    Serializable, Recreatable


# Create settings for this module
m = create_module_macros(module_path=__file__)

# How many predefined randomized scene object configurations we have per scene
m.N_PREDEFINED_OBJ_RANDOMIZATIONS = 10


class Environment(gym.Env, GymObservable, Recreatable):
    """
    Core environment class that handles loading scene, robot(s), and task, following OpenAI Gym interface.
    """
    def __init__(
        self,
        configs,
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
        device=None,
        automatic_reset=False,
    ):
        """
        :param configs (str or dict or list of str or dict): config_file path(s) or raw config dictionaries.
            If multiple configs are specified, they will be merged sequentially in the order specified.
            This allows procedural generation of a "full" config from small sub-configs.
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param device: None or str, specifies the device to be used if running on the gpu with torch backend
        :param automatic_reset: whether to automatic reset after an episode finishes
        """
        # Call super first
        super().__init__()

        # Store settings and other initialized values
        self._automatic_reset = automatic_reset
        self._predefined_object_randomization_idx = 0
        self._n_predefined_object_randomizations = m.N_PREDEFINED_OBJ_RANDOMIZATIONS
        self.action_timestep = action_timestep

        # Initialize other placeholders that will be filled in later
        self._ignore_robot_object_collisions = None         # This will be a list of len(scene.robots) containing sets of objects to ignore collisions for
        self._ignore_robot_self_collisions = None           # This will be a list of len(scene.robots) containing sets of RigidPrims (robot's links) to ignore collisions for
        self._initial_pos_z_offset = None                   # how high to offset object placement to account for one action step of dropping
        self._texture_randomization_freq = None
        self._object_randomization_freq = None
        self._task = None
        self._scene = None
        self._loaded = None
        self._current_episode = 0

        # Variables reset at the beginning of each episode
        self._current_collisions = None
        self._current_step = 0
        self._collision_step = 0

        # Convert config file(s) into a single parsed dict
        configs = configs if isinstance(configs, list) or isinstance(configs, tuple) else [configs]

        # Initial default config
        self.config = self.default_config

        # Merge in specified configs
        for config in configs:
            merge_nested_dicts(base_dict=self.config, extra_dict=parse_config(config), inplace=True)

        # Set the simulator settings
        ig.sim.set_simulation_dt(physics_dt=physics_timestep, rendering_dt=action_timestep)
        ig.sim.viewer_width = self.render_config["viewer_width"]
        ig.sim.viewer_height = self.render_config["viewer_height"]
        ig.sim.vertical_fov = self.render_config["vertical_fov"]
        ig.sim.device = device

        # Load this environment
        self.load()

    def add_ignore_robot_object_collision(self, robot_idn, obj):
        """
        Add a new robot-object pair to ignore collisions for

        NOTE: This ignores collisions for the purpose of COUNTING collisions, NOT for the purpose of disabling
            the actual, physical, collision

        Args:
            robot_idn (int): Which robot to ignore a collision for
            obj (BaseObject): Which object to ignore a collision for
        """
        self._ignore_robot_object_collisions[robot_idn].add(obj)

    def add_ignore_robot_self_collision(self, robot_idn, link):
        """
        Add a new robot-link self pair to ignore collisions for

        NOTE: This ignores collisions for the purpose of COUNTING collisions, NOT for the purpose of disabling
            the actual, physical, collision

        Args:
            robot_idn (int): Which robot to ignore a collision for
            link (RigidPrim): Which robot link to ignore a collision for
        """
        self._ignore_robot_self_collisions[robot_idn].add(link)

    def remove_ignore_robot_object_collision(self, robot_idn, obj):
        """
        Remove a robot-object pair to ignore collisions for

        NOTE: This ignores collisions for the purpose of COUNTING collisions, NOT for the purpose of disabling
            the actual, physical, collision

        Args:
            robot_idn (int): Which robot to ignore a collision for
            obj (BaseObject): Which object to no longer ignore a collision for
        """
        self._ignore_robot_object_collisions[robot_idn].remove(obj)

    def remove_ignore_robot_self_collision(self, robot_idn, link):
        """
        Remove a robot-link self pair to ignore collisions for

        NOTE: This ignores collisions for the purpose of COUNTING collisions, NOT for the purpose of disabling
            the actual, physical, collision

        Args:
            robot_idn (int): Which robot to ignore a collision for
            link (RigidPrim): Which robot link to no longer ignore a collision for
        """
        self._ignore_robot_self_collisions[robot_idn].remove(link)

    def reload(self, configs, overwrite_old=True):
        """
        Reload using another set of config file(s).
        This allows one to change the configuration and hot-reload the environment on the fly.

        Args:
            configs (str or list of str): config_file path(s). If multiple configs are specified, they will
                be merged sequentially in the order specified. This allows procedural generation of a "full" config from
                small sub-configs.
            overwrite_old (bool): If True, will overwrite the internal self.config with @configs. Otherwise, will
                merge in the new config(s) into the pre-existing one. Setting this to False allows for minor
                modifications to be made without having to specify entire configs during each reload.
        """
        # Convert config file(s) into a single parsed dict
        configs = [configs] if isinstance(configs, str) else configs

        # Initial default config
        new_config = self.default_config

        # Merge in specified configs
        for config in configs:
            merge_nested_dicts(base_dict=new_config, extra_dict=parse_config(config), inplace=True)

        # Either merge in or overwrite the old config
        if overwrite_old:
            self.config = new_config
        else:
            merge_nested_dicts(base_dict=self.config, extra_dict=new_config, inplace=True)

        # Load this environment again
        self.load()

    def reload_model(self, scene_model):
        """
        Reload another scene model.
        This allows one to change the scene on the fly.

        :param scene_model: new scene model to load (eg.: Rs_int)
        """
        self.scene_config["model"] = scene_model
        self.load()

    def _load_variables(self):
        """
        Load variables from config
        """
        # Store additional variables after config has been loaded fully
        self._initial_pos_z_offset = self.env_config["initial_pos_z_offset"]
        self._texture_randomization_freq = self.env_config["texture_randomization_freq"]
        self._object_randomization_freq = self.env_config["object_randomization_freq"]

        # Set other values
        self._ignore_robot_object_collisions = [set() for _ in self.robots_config]
        self._ignore_robot_self_collisions = [set() for _ in self.robots_config]

        # Reset bookkeeping variables
        self._reset_variables()
        self._current_episode = 0           # Manually set this to 0 since resetting actually increments this

        # - Potentially overwrite the USD entry for the scene if none is specified and we're online sampling -

        # Make sure the requested scene is valid
        scene_type = self.scene_config["type"]
        assert_valid_key(key=scene_type, valid_keys=REGISTERED_SCENES, name="scene type")

        # If we're using a BehaviorTask, we may load a pre-cached scene configuration
        if self.task_config["type"] == "BehaviorTask":
            usd_file = self.scene_config["usd_file"]
            if usd_file is None and not self.task_config["online_object_sampling"]:
                usd_file = "{}_task_{}_{}_{}_fixed_furniture_template".format(
                    self.scene_config["scene_model"],
                    self.task_config["activity_name"],
                    self.task_config["activity_definition_id"],
                    self.task_config["activity_instance_id"],
                )
            # Update the value in the scene config
            self.scene_config["usd_file"] = usd_file

        # - Additionally run some sanity checks on these values -

        # Check to make sure our z offset is valid -- check that the distance travelled over 1 action timestep is
        # less than the offset we set (dist = 0.5 * gravity * (t^2))
        drop_distance = 0.5 * 9.8 * (self.action_timestep ** 2)
        assert drop_distance < self._initial_pos_z_offset, "initial_pos_z_offset is too small for collision checking"

    def _load_task(self):
        """
        Load task
        """
        # Sanity check task to make sure it's valid
        task_type = self.task_config["type"]
        assert_valid_key(key=task_type, valid_keys=REGISTERED_TASKS, name="task type")

        # Grab the kwargs relevant for the specific task and create the task
        self._task = create_class_from_registry_and_config(
            cls_name=self.task_config["type"],
            cls_registry=REGISTERED_TASKS,
            cfg=self.task_config,
            cls_type_descriptor="task",
        )

        # Load task
        # TODO: Does this need to occur somewhere else?
        self._task.load(env=self)

    def _load_scene(self):
        """
        Load the scene and robot specified in the config file.
        """
        # Create the scene from our scene config
        scene = create_class_from_registry_and_config(
            cls_name=self.scene_config["type"],
            cls_registry=REGISTERED_SCENES,
            cfg=self.scene_config,
            cls_type_descriptor="scene",
        )
        ig.sim.import_scene(scene)

        # Save scene internally
        self._scene = scene

    def _load_robots(self):
        """
        Load robots into the scene
        """
        # Only actually load robots if no robot has been imported from the scene loading directly yet
        if len(self._scene.robots) == 0:
            # Iterate over all robots to generate in the robot config
            for i, robot_config in enumerate(self.robots_config):
                # Add a name for the robot if necessary
                if "name" not in robot_config:
                    robot_config["name"] = f"robot{i}"
                # Set prim path
                robot_config["prim_path"] = f"/World/{robot_config['name']}"
                # Make sure robot exists, grab its corresponding kwargs, and create / import the robot
                robot = create_class_from_registry_and_config(
                    cls_name=robot_config["type"],
                    cls_registry=REGISTERED_ROBOTS,
                    cfg=robot_config,
                    cls_type_descriptor="robot",
                )
                # Import the robot into the simulator
                ig.sim.import_object(robot)

    def _load_observation_space(self):
        # Grab robot(s) and task obs spaces
        obs_space = OrderedDict()

        for robot in self.robots:
            # Load the observation space for the robot
            obs_space[robot.name] = robot.load_observation_space()

        # Also load the task obs space
        obs_space["task"] = self._task.load_observation_space()
        return obs_space

    def _load_action_space(self):
        """
        Load action space for each robot
        """
        self.action_space = gym.spaces.Dict({robot.name: robot.action_space for robot in self.robots})

    def load(self):
        """
        Load the scene and robot specified in the config file.
        """
        # This environment is not loaded
        self._loaded = False

        # Load config variables
        self._load_variables()

        # Load the scene, robots, and task
        self._load_scene()
        self._load_robots()
        self._load_task()

        # Start the simulation, then reset the environment
        ig.sim.play()
        self.reset()

        # Update the initial scene state
        self.scene.update_initial_state()

        # Load the obs / action spaces
        self.load_observation_space()
        self._load_action_space()

        # Denote that the scene is loaded
        self._loaded = True

    def reload_model_object_randomization(self, predefined_object_randomization_idx=None):
        """
        Reload the same model, with either @object_randomization_idx seed or the next object randomization random seed.

        Args:
            predefined_object_randomization_idx (None or int): If set, specifies the specific pre-defined object
                randomization instance to use. Otherwise, the current seed will be incremented by 1 and used.
        """
        assert self._object_randomization_freq is not None, \
            "object randomization must be active to reload environment with object randomization!"
        self._predefined_object_randomization_idx = predefined_object_randomization_idx if \
            predefined_object_randomization_idx is not None else \
            (self._predefined_object_randomization_idx + 1) % self._n_predefined_object_randomizations
        self.load()

    def close(self):
        """
        Clean up the environment and shut down the simulation.
        """
        ig.sim.close()

    def get_obs(self):
        """
        Get the current environment observation.

        Returns:
            OrderedDict: Keyword-mapped observations, which are possibly nested
        """
        obs = OrderedDict()

        # Grab all observations from each robot
        for robot in self.robots:
            obs[robot.name] = robot.get_obs()

        # Add task observations
        obs["task"] = self._task.get_obs(env=self)

        return obs

    def _filter_collisions(self, collisions):
        """
        Filter out collisions that should be ignored, based on self._ignore_robot_[object/self]_collisions

        Args:
            collisions (set of 2-tuple): Unique collision pairs occurring in the simulation at the current timestep,
                represented by their prim_paths

        Returns:
            set of 2-tuple: Filtered collision pairs occurring in the simulation at the current timestep,
                represented by their prim_paths
        """
        # Iterate over all robots
        new_collisions = set()
        for robot, filtered_links, filtered_objs in zip(
                self.robots, self._ignore_robot_self_collisions, self._ignore_robot_object_collisions
        ):
            # Grab all link prim paths owned by the robot
            robot_prims = {link.prim_path for link in robot.links.values()}
            # Loop over all filtered links and compose them
            filtered_link_prims = {link.prim_path for link in filtered_links}
            # Loop over all filtered objects and compose them
            filtered_obj_prims = {link.prim_path for obj in filtered_objs for link in obj.links.values()}
            # Iterate over all collision pairs
            for col_pair in collisions:
                # First check for self_collision -- we check by first subtracting all filtered links from the col_pair
                # set and making sure the length is < 2, and then subtracting all robot_prims and making sure that the
                # length is 0. If both conditions are met, then we know this is a filtered collision!
                col_pair_set = set(col_pair)
                if len(col_pair_set - filtered_link_prims) < 2 and len(col_pair_set - robot_prims) == 0:
                    # Filter this collision
                    continue
                # Check for object filtering -- we check by first subtracting all robot links from the col_pair
                # set and making sure the length is < 2, and then subtracting all filtered_obj_prims and making sure
                # that the length is < 2. If both conditions are met, this means that this was a collision between
                # the robot and a filtered object, so we know this is a filtered collision!
                elif len(col_pair_set - robot_prims) < 2 and len(col_pair_set - filtered_obj_prims) < 2:
                    # Filter this collision
                    continue
                else:
                    # Add this collision
                    new_collisions.add(col_pair)

        return new_collisions

    def _populate_info(self, info):
        """
        Populate info dictionary with any useful information.

        Args:
            info (dict): Information dictionary to populate

        Returns:
            dict: Information dictionary with added info
        """
        info["episode_length"] = self._current_step
        info["collision_step"] = self._collision_step

    def step(self, action):
        """
        Apply robot's action and return the next state, reward, done and info,
        following OpenAI Gym's convention

        :param action: gym.spaces.Dict, dict, np.array, robot actions
        :return: state: next observation
        :return: reward: reward of this time step
        :return: done: whether the episode is terminated
        :return: info: info dictionary with any useful information
        """
        # If the action is not a dictionary, convert into a dictionary
        if not isinstance(action, dict) and not isinstance(action, gym.spaces.Dict):
            action_dict = OrderedDict()
            idx = 0
            for robot in self.robots:
                action_dim = robot.action_dim
                action_dict[robot.name] = action[idx: idx + action_dim]
                idx += action_dim
        else:
            # Our inputted action is the action dictionary
            action_dict = action

        # Iterate over all robots and apply actions
        for robot in self.robots:
            robot.apply_action(action_dict[robot.name])

        # Run simulation step
        ig.sim.step()

        # Grab collisions and store internally
        if gm.ENABLE_GLOBAL_CONTACT_REPORTING:
            collision_objects = self.scene.objects + self.robots
        elif gm.ENABLE_ROBOT_CONTACT_REPORTING:
            collision_objects = self.robots
        else:
            collision_objects = []

        # Update current collisions and corresponding count
        self._current_collisions = self._filter_collisions(collisions=get_collisions(prims=collision_objects))

        # Update the collision count
        self._collision_step += int(len(self._current_collisions) > 0)

        # Grab observations
        obs = self.get_obs()

        # Grab reward, done, and info, and populate with internal info
        reward, done, info = self.task.step(self, action)
        self._populate_info(info)

        if done and self._automatic_reset:
            # Add lost observation to our information dict, and reset
            info["last_observation"] = obs
            obs = self.reset()

        # Increment step
        self._current_step += 1

        return obs, reward, done, info

    def randomize_domain(self):
        """
        Randomize domain.
        Object randomization loads new object models with the same poses.
        Texture randomization loads new materials and textures for the same object models.
        """
        if self._object_randomization_freq is not None:
            if self._current_episode % self._object_randomization_freq == 0:
                self.reload_model_object_randomization()
        if self._texture_randomization_freq is not None:
            if self._current_episode % self._texture_randomization_freq == 0:
                ig.sim.scene.randomize_texture()

    def _reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode.
        """
        self._current_episode += 1
        self._current_collisions = None
        self._current_step = 0
        self._collision_step = 0

    # TODO: Match super class signature?
    def reset(self):
        """
        Reset episode.
        """
        # Stop and restart the simulation
        ig.sim.stop()
        ig.sim.play()

        # Do any domain randomization
        self.randomize_domain()

        # # Move all robots away from the scene since the task will place the robots anyways
        # for robot in self.robots:
        #     robot.set_position(np.array([100.0, 100.0, 100.0]))

        # Reset the task
        self.task.reset(self)

        # Reset internal variables
        self._reset_variables()

        # Run a single simulator step to make sure we can grab updated observations
        ig.sim.step()

        # Grab and return observations
        obs = self.get_obs()

        if self.observation_space is not None and not self.observation_space.contains(obs):
            print("Error: Observation space does not match returned observations!")
            # Print out all observations for all robots and task
            for robot in self.robots:
                for key, value in self.observation_space[robot.name].items():
                    print(key, value.dtype, value.shape)
                    print('obs', obs[robot.name][key].dtype, obs[robot.name][key].shape)
            for key, value in self.observation_space['task'].items():
                print(key, value.dtype, value.shape)
                print('obs', obs['task'][key].dtype, obs['task'][key].shape)

        return obs

    @property
    def episode_steps(self):
        """
        Returns:
            int: Current number of steps in episode
        """
        return self._current_step

    @property
    def episode_collisions(self):
        """
        Returns:
            int: Total number of collisions in current episode
        """
        return self._collision_step

    @property
    def current_collisions(self):
        """
        Returns:
            set of 2-tuple: Cached collision pairs from the last time self.update_collisions() was called
        """
        return self._current_collisions

    @property
    def task(self):
        """
        Returns:
            BaseTask: Active task instance
        """
        return self._task

    @property
    def scene(self):
        """
        Returns:
            Scene: Active scene in this environment
        """
        return self._scene

    @property
    def robots(self):
        """
        Returns:
            list of BaseRobot: Robots in the current scene
        """
        return self._scene.robots

    @property
    def env_config(self):
        """
        Returns:
            dict: Environment-specific configuration kwargs
        """
        return self.config["env"]

    @property
    def render_config(self):
        """
        Returns:
            dict: Render-specific configuration kwargs
        """
        return self.config["render"]

    @property
    def scene_config(self):
        """
        Returns:
            dict: Scene-specific configuration kwargs
        """
        print(self.config["scene"])
        return self.config["scene"]

    @property
    def robots_config(self):
        """
        Returns:
            dict: Robot-specific configuration kwargs
        """
        return self.config["robots"]

    @property
    def task_config(self):
        """
        Returns:
            dict: Task-specific configuration kwargs
        """
        return self.config["task"]

    @property
    def default_config(self):
        """
        Returns:
            dict: Default configuration for this environment. May not be fully specified (i.e.: still requires @config
                to be specified during environment creation)
        """
        return {
            # Environment kwargs
            "env": {
                "initial_pos_z_offset": 0.1,
                "texture_randomization_freq": None,
                "object_randomization_freq": None,
                "usd_file": None,
            },

            # Rendering kwargs
            "render": {
                "viewer_width": 1280,
                "viewer_height": 720,
                "vertical_fov": 90,
                # "optimized_renderer": True,
            },

            # Scene kwargs
            "scene": {
                # Traversibility map kwargs
                "waypoint_resolution": 0.2,
                "num_waypoints": 10,
                "build_graph": False,
                "trav_map_resolution": 0.1,
                "trav_map_erosion": 2,
                "trav_map_with_objects": True,
            },

            # Robot kwargs
            "robots": [], # no robots by default

            # Task kwargs
            "task": {
                "type": "DummyTask",

                # If we're using a BehaviorTask
                "activity_definition_id": 0,
                "activity_instance_id": 0,
            }
        }
