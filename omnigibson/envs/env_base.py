import gym
import logging
from collections import OrderedDict

import omnigibson as og
from omnigibson.macros import gm, create_module_macros
from omnigibson.objects import REGISTERED_OBJECTS
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.tasks import REGISTERED_TASKS
from omnigibson.scenes import REGISTERED_SCENES
from omnigibson.utils.gym_utils import GymObservable
from omnigibson.utils.config_utils import parse_config
from omnigibson.utils.python_utils import assert_valid_key, merge_nested_dicts, create_class_from_registry_and_config,\
    Recreatable


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
        action_timestep=1 / 60.0,
        physics_timestep=1 / 60.0,
        device=None,
        automatic_reset=False,
    ):
        """
        Args:
            configs (str or dict or list of str or dict): config_file path(s) or raw config dictionaries.
                If multiple configs are specified, they will be merged sequentially in the order specified.
                This allows procedural generation of a "full" config from small sub-configs.
            action_timestep (float): environment executes action per action_timestep second
            physics_timestep: physics timestep for physx
            device (None or str): specifies the device to be used if running on the gpu with torch backend
            automatic_reset (bool): whether to automatic reset after an episode finishes
        """
        # Call super first
        super().__init__()

        # Store settings and other initialized values
        self._automatic_reset = automatic_reset
        self._predefined_object_randomization_idx = 0
        self._n_predefined_object_randomizations = m.N_PREDEFINED_OBJ_RANDOMIZATIONS
        self.action_timestep = action_timestep

        # Initialize other placeholders that will be filled in later
        self._initial_pos_z_offset = None                   # how high to offset object placement to account for one action step of dropping
        self._texture_randomization_freq = None
        self._object_randomization_freq = None
        self._task = None
        self._loaded = None
        self._current_episode = 0

        # Variables reset at the beginning of each episode
        self._current_step = 0

        # Convert config file(s) into a single parsed dict
        configs = configs if isinstance(configs, list) or isinstance(configs, tuple) else [configs]

        # Initial default config
        self.config = self.default_config

        # Merge in specified configs
        for config in configs:
            merge_nested_dicts(base_dict=self.config, extra_dict=parse_config(config), inplace=True)

        # Set the simulator settings
        og.sim.set_simulation_dt(physics_dt=physics_timestep, rendering_dt=action_timestep)
        og.sim.viewer_width = self.render_config["viewer_width"]
        og.sim.viewer_height = self.render_config["viewer_height"]
        og.sim.vertical_fov = self.render_config["vertical_fov"]
        og.sim.device = device

        # Load this environment
        self.load()

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

        Args:
            scene_model (str): new scene model to load (eg.: Rs_int)
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
            scene_instance, scene_file = self.scene_config["scene_instance"], self.scene_config["scene_file"]
            if scene_file is None and scene_instance is None and not self.task_config["online_object_sampling"]:
                scene_instance = "{}_task_{}_{}_{}_fixed_furniture_template".format(
                    self.scene_config["scene_model"],
                    self.task_config["activity_name"],
                    self.task_config["activity_definition_id"],
                    self.task_config["activity_instance_id"],
                )
            # Update the value in the scene config
            self.scene_config["scene_instance"] = scene_instance

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

        assert og.sim.is_stopped(), "sim should be stopped when load_task starts"
        og.sim.play()

        # Load task. Should load additinal task-relevant objects and configure the scene into its default initial state
        self._task.load(env=self)

        # Update the initial scene state
        self.scene.update_initial_state()

        og.sim.stop()

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
        og.sim.import_scene(scene)

    def _load_robots(self):
        """
        Load robots into the scene
        """
        # Only actually load robots if no robot has been imported from the scene loading directly yet
        if len(self.scene.robots) == 0:
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
                og.sim.import_object(robot)

    def _load_objects(self):
        """
        Load any additional custom objects into the scene
        """
        for i, obj_config in enumerate(self.objects_config):
            # Add a name for the object if necessary
            if "name" not in obj_config:
                obj_config["name"] = f"obj{i}"
            # Set prim path if not specified
            if "prim_path" not in obj_config:
                obj_config["prim_path"] = f"/World/{obj_config['name']}"
            # Pop the desired position and orientation
            position, orientation = obj_config.pop("position", None), obj_config.pop("orientation", None)
            # Make sure robot exists, grab its corresponding kwargs, and create / import the robot
            obj = create_class_from_registry_and_config(
                cls_name=obj_config["type"],
                cls_registry=REGISTERED_OBJECTS,
                cfg=obj_config,
                cls_type_descriptor="object",
            )
            # Import the robot into the simulator and set the pose
            og.sim.import_object(obj)
            obj.set_position_orientation(position=position, orientation=orientation)

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
        self._load_objects()
        self._load_task()

        og.sim.play()
        self.reset()

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
        og.sim.close()

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

    def _populate_info(self, info):
        """
        Populate info dictionary with any useful information.

        Args:
            info (dict): Information dictionary to populate

        Returns:
            dict: Information dictionary with added info
        """
        info["episode_length"] = self._current_step

    def step(self, action):
        """
        Apply robot's action and return the next state, reward, done and info,
        following OpenAI Gym's convention

        Args:
            action (gym.spaces.Dict or dict or np.array): robot actions. If a dict is specified, each entry should
                map robot name to corresponding action. If a np.array, it should be the flattened, concatenated set
                of actions

        Returns:
            4-tuple:
                - OrderedDict: state, i.e. next observation
                - float: reward, i.e. reward at this current timestep
                - bool: done, i.e. whether this episode is terminated
                - OrderedDict: info, i.e. dictionary with any useful information
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
        og.sim.step()

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
                og.sim.scene.randomize_texture()

    def _reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode.
        """
        self._current_episode += 1
        self._current_step = 0

    # TODO: Match super class signature?
    def reset(self):
        """
        Reset episode.
        """
        # Stop and restart the simulation
        og.sim.stop()
        og.sim.play()

        # Do any domain randomization
        self.randomize_domain()

        # Reset the task
        self.task.reset(self)

        # Reset internal variables
        self._reset_variables()

        # Run a single simulator step to make sure we can grab updated observations
        og.sim.step()

        # Grab and return observations
        obs = self.get_obs()

        if self.observation_space is not None and not self.observation_space.contains(obs):
            # Print out all observations for all robots and task
            for robot in self.robots:
                for key, value in self.observation_space[robot.name].items():
                    logging.error(("obs_space", key, value.dtype, value.shape))
                    logging.error(("obs", key, obs[robot.name][key].dtype, obs[robot.name][key].shape))
            for key, value in self.observation_space["task"].items():
                logging.error(("obs_space", key, value.dtype, value.shape))
                logging.error(("obs", key, obs["task"][key].dtype, obs["task"][key].shape))
            raise ValueError("Observation space does not match returned observations!")

        return obs

    @property
    def episode_steps(self):
        """
        Returns:
            int: Current number of steps in episode
        """
        return self._current_step

    @property
    def initial_pos_z_offset(self):
        """
        Returns:
            float: how high to offset object placement to test valid pose & account for one action step of dropping
        """
        return self._initial_pos_z_offset

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
        return og.sim.scene

    @property
    def robots(self):
        """
        Returns:
            list of BaseRobot: Robots in the current scene
        """
        return self.scene.robots

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
        return self.config["scene"]

    @property
    def robots_config(self):
        """
        Returns:
            dict: Robot-specific configuration kwargs
        """
        return self.config["robots"]

    @property
    def objects_config(self):
        """
        Returns:
            dict: Object-specific configuration kwargs
        """
        return self.config["objects"]

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
            },

            # Rendering kwargs
            "render": {
                "viewer_width": 1280,
                "viewer_height": 720,
                "vertical_fov": 90,
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
                "scene_instance": None,
                "scene_file": None,
            },

            # Robot kwargs
            "robots": [],   # no robots by default

            # Object kwargs
            "objects": [],  # no objects by default

            # Task kwargs
            "task": {
                "type": "DummyTask",

                # If we're using a BehaviorTask
                "activity_definition_id": 0,
                "activity_instance_id": 0,
            }
        }
