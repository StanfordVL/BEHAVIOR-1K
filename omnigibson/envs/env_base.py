import gymnasium as gym
import numpy as np
from copy import deepcopy

import omnigibson as og
from omnigibson.objects import REGISTERED_OBJECTS
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.scene_graphs.graph_builder import SceneGraphBuilder
from omnigibson.tasks import REGISTERED_TASKS
from omnigibson.scenes import REGISTERED_SCENES
from omnigibson.sensors import create_sensor, VisionSensor
from omnigibson.utils.gym_utils import GymObservable, recursively_generate_flat_dict, recursively_generate_compatible_dict
from omnigibson.utils.config_utils import parse_config
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.python_utils import assert_valid_key, merge_nested_dicts, create_class_from_registry_and_config,\
    Recreatable


# Create module logger
log = create_module_logger(module_name=__name__)


class Environment(gym.Env, GymObservable, Recreatable):
    """
    Core environment class that handles loading scene, robot(s), and task, following OpenAI Gym interface.
    """
    def __init__(self, configs):
        """
        Args:
            configs (str or dict or list of str or dict): config_file path(s) or raw config dictionaries.
                If multiple configs are specified, they will be merged sequentially in the order specified.
                This allows procedural generation of a "full" config from small sub-configs. For valid keys, please
                see @default_config below
        """
        # Call super first
        super().__init__()

        # Support gymnasium's render mode metadata
        self.render_mode = "rgb_array"
        self.metadata = {"render.modes": ["rgb_array"]}

        # Initialize other placeholders that will be filled in later
        self._task = None
        self._external_sensors = None
        self._external_sensors_include_in_obs = None
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

        # Store settings and other initialized values
        self._automatic_reset = self.env_config["automatic_reset"]
        self._flatten_action_space = self.env_config["flatten_action_space"]
        self._flatten_obs_space = self.env_config["flatten_obs_space"]
        self.physics_timestep = self.env_config["physics_timestep"]
        self.action_timestep = self.env_config["action_timestep"]
        self.device = self.env_config["device"]
        self._initial_pos_z_offset = self.env_config["initial_pos_z_offset"]    # how high to offset object placement to account for one action step of dropping

        # Create the scene graph builder
        self._scene_graph_builder = None
        if "scene_graph" in self.config and self.config["scene_graph"] is not None:
            self._scene_graph_builder = SceneGraphBuilder(**self.config["scene_graph"])
          
        # Load this environment
        self.load()

    def reload(self, configs, overwrite_old=True):
        """
        Reload using another set of config file(s).
        This allows one to change the configuration and hot-reload the environment on the fly.

        Args:
            configs (dict or str or list of dict or list of str): config_file dict(s) or path(s). 
                If multiple configs are specified, they will be merged sequentially in the order specified. 
                This allows procedural generation of a "full" config from small sub-configs.
            overwrite_old (bool): If True, will overwrite the internal self.config with @configs. Otherwise, will
                merge in the new config(s) into the pre-existing one. Setting this to False allows for minor
                modifications to be made without having to specify entire configs during each reload.
        """
        # Convert config file(s) into a single parsed dict
        configs = [configs] if isinstance(configs, dict) or isinstance(configs, str) else configs

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

        # Reset bookkeeping variables
        self._reset_variables()
        self._current_episode = 0           # Manually set this to 0 since resetting actually increments this

        # - Potentially overwrite the USD entry for the scene if none is specified and we're online sampling -

        # Make sure the requested scene is valid
        scene_type = self.scene_config["type"]
        assert_valid_key(key=scene_type, valid_keys=REGISTERED_SCENES, name="scene type")

        # Verify scene and task configs are valid for the given task type
        REGISTERED_TASKS[self.task_config["type"]].verify_scene_and_task_config(
            scene_cfg=self.scene_config,
            task_cfg=self.task_config,
        )

        # - Additionally run some sanity checks on these values -

        # Check to make sure our z offset is valid -- check that the distance travelled over 1 action timestep is
        # less than the offset we set (dist = 0.5 * gravity * (t^2))
        drop_distance = 0.5 * 9.8 * (self.action_timestep ** 2)
        assert drop_distance < self._initial_pos_z_offset, "initial_pos_z_offset is too small for collision checking"

    def _load_task(self, task_config=None):
        """
        Load task

        Args:
            task_confg (None or dict): If specified, custom task configuration to use. Otherwise, will use
                self.task_config. Note that if a custom task configuration is specified, the internal task config
                will be updated as well
        """
        # Update internal config if specified
        if task_config is not None:
            # Copy task config, in case self.task_config and task_config are the same!
            task_config = deepcopy(task_config)
            self.task_config.clear()
            self.task_config.update(task_config)

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
        assert og.sim.is_stopped(), "Simulator must be stopped before loading tasks!"

        # Load task. Should load additional task-relevant objects and configure the scene into its default initial state
        self._task.load(env=self)

        assert og.sim.is_stopped(), "Simulator must be stopped after loading tasks!"

    def _load_scene(self):
        """
        Load the scene and robot specified in the config file.
        """
        assert og.sim.is_stopped(), "Simulator must be stopped before loading scene!"

        # Set the simulator settings
        # NOTE: This must be done BEFORE the scene is loaded, or else all vision sensors can't retrieve observations
        og.sim.set_simulation_dt(physics_dt=self.physics_timestep, rendering_dt=self.action_timestep)

        # Create the scene from our scene config
        scene = create_class_from_registry_and_config(
            cls_name=self.scene_config["type"],
            cls_registry=REGISTERED_SCENES,
            cfg=self.scene_config,
            cls_type_descriptor="scene",
        )
        og.sim.import_scene(scene)

        # Set the rendering settings
        og.sim.viewer_width = self.render_config["viewer_width"]
        og.sim.viewer_height = self.render_config["viewer_height"]
        og.sim.device = self.device

        assert og.sim.is_stopped(), "Simulator must be stopped after loading scene!"

    def _load_robots(self):
        """
        Load robots into the scene
        """
        # Only actually load robots if no robot has been imported from the scene loading directly yet
        if len(self.scene.robots) == 0:
            assert og.sim.is_stopped(), "Simulator must be stopped before loading robots!"

            # Iterate over all robots to generate in the robot config
            for i, robot_config in enumerate(self.robots_config):
                # Add a name for the robot if necessary
                if "name" not in robot_config:
                    robot_config["name"] = f"robot{i}"

                position, orientation = robot_config.pop("position", None), robot_config.pop("orientation", None)
                # Make sure robot exists, grab its corresponding kwargs, and create / import the robot
                robot = create_class_from_registry_and_config(
                    cls_name=robot_config["type"],
                    cls_registry=REGISTERED_ROBOTS,
                    cfg=robot_config,
                    cls_type_descriptor="robot",
                )
                # Import the robot into the simulator
                og.sim.import_object(robot)
                robot.set_position_orientation(position=position, orientation=orientation)

            # Auto-initialize all robots
            og.sim.play()
            self.scene.reset()
            self.scene.update_initial_state()
            og.sim.stop()

        assert og.sim.is_stopped(), "Simulator must be stopped after loading robots!"

    def _load_objects(self):
        """
        Load any additional custom objects into the scene
        """
        assert og.sim.is_stopped(), "Simulator must be stopped before loading objects!"
        for i, obj_config in enumerate(self.objects_config):
            # Add a name for the object if necessary
            if "name" not in obj_config:
                obj_config["name"] = f"obj{i}"
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

        # Auto-initialize all objects
        og.sim.play()
        self.scene.reset()
        self.scene.update_initial_state()
        og.sim.stop()

        assert og.sim.is_stopped(), "Simulator must be stopped after loading objects!"

    def _load_external_sensors(self):
        """
        Load any additional custom external sensors into the scene
        """
        assert og.sim.is_stopped(), "Simulator must be stopped before loading external sensors!"
        sensors_config = self.env_config["external_sensors"]
        if sensors_config is not None:
            self._external_sensors = dict()
            self._external_sensors_include_in_obs = dict()
            for i, sensor_config in enumerate(sensors_config):
                # Add a name for the object if necessary
                if "name" not in sensor_config:
                    sensor_config["name"] = f"external_sensor{i}"
                # Determine prim path if not specified
                if "prim_path" not in sensor_config:
                    sensor_config["prim_path"] = f"/World/{sensor_config['name']}"
                # Pop the desired position and orientation
                local_position, local_orientation = sensor_config.pop("local_position", None), sensor_config.pop("local_orientation", None)
                # Pop whether or not to include this sensor in the observation
                include_in_obs = sensor_config.pop("include_in_obs", True)
                # Make sure sensor exists, grab its corresponding kwargs, and create the sensor
                sensor = create_sensor(**sensor_config)
                # Load an initialize this sensor
                sensor.load()
                sensor.initialize()
                sensor.set_local_pose(local_position, local_orientation)
                self._external_sensors[sensor.name] = sensor
                self._external_sensors_include_in_obs[sensor.name] = include_in_obs

        assert og.sim.is_stopped(), "Simulator must be stopped after loading external sensors!"

    def _load_observation_space(self):
        # Grab robot(s) and task obs spaces
        obs_space = dict()

        for robot in self.robots:
            # Load the observation space for the robot
            robot_obs = robot.load_observation_space()
            if gym.spaces.utils.flatdim(robot_obs) > 0:
                obs_space[robot.name] = robot_obs

        # Also load the task obs space
        task_space = self._task.load_observation_space()
        if gym.spaces.utils.flatdim(task_space) > 0:
            obs_space["task"] = task_space

        # Also load any external sensors
        if self._external_sensors is not None:
            external_obs_space = dict()
            for sensor_name, sensor in self._external_sensors.items():
                if not self._external_sensors_include_in_obs[sensor_name]:
                    continue

                # Load the sensor observation space
                external_obs_space[sensor_name] = sensor.load_observation_space()
            obs_space["external"] = gym.spaces.Dict(external_obs_space)

        return obs_space

    def load_observation_space(self):
        # Call super first
        obs_space = super().load_observation_space()

        # If we want to flatten it, modify the observation space by recursively searching through all
        if self._flatten_obs_space:
            self.observation_space = gym.spaces.Dict(recursively_generate_flat_dict(dic=obs_space))

        return self.observation_space

    def _load_action_space(self):
        """
        Load action space for each robot
        """
        action_space = gym.spaces.Dict({robot.name: robot.action_space for robot in self.robots})

        # Convert into flattened 1D Box space if requested
        if self._flatten_action_space:
            lows = []
            highs = []
            for space in action_space.values():
                assert isinstance(space, gym.spaces.Box), \
                    "Can only flatten action space where all individual spaces are gym.space.Box instances!"
                assert len(space.shape) == 1, \
                    "Can only flatten action space where all individual spaces are 1D instances!"
                lows.append(space.low)
                highs.append(space.high)
            action_space = gym.spaces.Box(np.concatenate(lows), np.concatenate(highs), dtype=np.float32)

        # Store action space
        self.action_space = action_space

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
        self._load_external_sensors()

        og.sim.play()

        # Load the obs / action spaces
        self.load_observation_space()
        self._load_action_space()

        self.reset()

        # Start the scene graph builder
        if self._scene_graph_builder:
            self._scene_graph_builder.start(self.scene)

        # Denote that the scene is loaded
        self._loaded = True

    def update_task(self, task_config):
        """
        Updates the internal task using @task_config. NOTE: This will internally reset the environment as well!

        Args:
            task_config (dict): Task configuration for updating the new task
        """
        # Make sure sim is playing
        assert og.sim.is_playing(), "Update task should occur while sim is playing!"

        # Denote scene as not loaded yet
        self._loaded = False
        og.sim.stop()
        self._load_task(task_config=task_config)
        og.sim.play()
        self.reset()

        # Load obs / action spaces
        self.load_observation_space()
        self._load_action_space()

        # Scene is now loaded again
        self._loaded = True


    def close(self):
        """
        Clean up the environment and shut down the simulation.
        """
        return
    
        # og.shutdown()

    def get_obs(self):
        """
        Get the current environment observation.

        Returns:
            dict: Keyword-mapped observations, which are possibly nested
        """
        obs = dict()

        # Grab all observations from each robot
        for robot in self.robots:
            if gym.spaces.utils.flatdim(robot.observation_space) > 0:
                obs[robot.name] = robot.get_obs()

        # Add task observations
        if gym.spaces.utils.flatdim(self._task.observation_space) > 0:
            obs["task"] = self._task.get_obs(env=self)

        # Add external sensor observations if they exist
        if self._external_sensors is not None:
            external_obs = dict()
            for sensor_name, sensor in self._external_sensors.items():
                if not self._external_sensors_include_in_obs[sensor_name]:
                    continue

                external_obs[sensor_name] = sensor.get_obs()
            obs["external"] = external_obs

        # Possibly flatten obs if requested
        if self._flatten_obs_space:
            obs = recursively_generate_flat_dict(dic=obs)

        return obs
    
    def get_scene_graph(self):
        """
        Get the current scene graph.

        Returns:
            SceneGraph: Current scene graph
        """
        assert self._scene_graph_builder is not None, "Scene graph builder must be specified in config!"
        return self._scene_graph_builder.get_scene_graph()

    def _populate_info(self, info):
        """
        Populate info dictionary with any useful information.

        Args:
            info (dict): Information dictionary to populate

        Returns:
            dict: Information dictionary with added info
        """
        info["episode_length"] = self._current_step

        if self._scene_graph_builder is not None:
            info["scene_graph"] = self.get_scene_graph()

    def step(self, action):
        """
        Apply robot's action and return the next state, reward, done and info,
        following OpenAI Gym's convention

        Args:
            action (gym.spaces.Dict or dict or np.array): robot actions. If a dict is specified, each entry should
                map robot name to corresponding action. If a np.array, it should be the flattened, concatenated set
                of actions

        Returns:
            5-tuple:
                - dict: state, i.e. next observation
                - float: reward, i.e. reward at this current timestep
                - bool: terminated, i.e. whether this episode ended due to a failure or success
                - bool: truncated, i.e. whether this episode ended due to a time limit etc.
                - dict: info, i.e. dictionary with any useful information
        """
        try:
            # If the action is not a dictionary, convert into a dictionary
            if not isinstance(action, dict) and not isinstance(action, gym.spaces.Dict):
                action_dict = dict()
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

            # Decide if rendering is necessary
            visual_modalities = {"rgb", "depth", "normal", "seg_instance"}
            should_render = any(
                not visual_modalities.isdisjoint(robot.obs_modalities)
                for robot in self.robots
            )

            # Run simulation step
            og.sim.step(render=should_render)

            # Grab observations
            obs = self.get_obs()

            # Step the scene graph builder if necessary
            if self._scene_graph_builder is not None:
                self._scene_graph_builder.step(self.scene)

            # Grab reward, done, and info, and populate with internal info
            reward, done, info = self.task.step(self, action)
            self._populate_info(info)

            if done and self._automatic_reset:
                # Add lost observation to our information dict, and reset
                info["last_observation"] = obs
                obs = self.reset()

            # Hacky way to check for time limit info to split terminated and truncated
            terminated = False
            truncated = False
            for tc, tc_data in info["done"]["termination_conditions"].items():
                if tc_data["done"]:
                    if tc == "timeout":
                        truncated = True
                    else:
                        terminated = True
            assert (terminated or truncated) == done, "Terminated and truncated must match done!"

            # Increment step
            self._current_step += 1

            return obs, reward, terminated, truncated, info
        except:
            raise ValueError(f"Failed to execute environment step {self._current_step} in episode {self._current_episode}")

    def render(self):
        # Only works if there is an external sensor
        if not self._external_sensors:
            return None
        
        # Get the RGB sensors
        rgb_sensors = [
            x for x in self._external_sensors.values()
            if isinstance(x, VisionSensor) and (x.modalities == "all" or "rgb" in x.modalities)
        ]
        if not rgb_sensors:
            return None

        # Render the external sensor
        og.sim.render()

        # Grab the rendered image from each of the rgb sensors, concatenate along dim 1
        rgb_images = [sensor.get_obs()["rgb"] for sensor in rgb_sensors]
        return np.concatenate(rgb_images, axis=1)[:, :, :3]


    def _reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode.
        """
        self._current_episode += 1
        self._current_step = 0

    def reset(self, **kwargs):
        """
        Reset episode.
        """
        # Reset the task
        self.task.reset(self)

        # Reset internal variables
        self._reset_variables()

        # Run a single simulator step to make sure we can grab updated observations
        og.sim.step()

        # Grab and return observations
        obs = self.get_obs()

        if self._loaded:
            # Sanity check to make sure received observations match expected observation space
            check_obs = recursively_generate_compatible_dict(dic=obs)
            if not self.observation_space.contains(check_obs):
                exp_obs = dict()
                for key, value in recursively_generate_flat_dict(dic=self.observation_space).items():
                    exp_obs[key] = ("obs_space", key, value.dtype, value.shape)
                real_obs = dict()
                for key, value in recursively_generate_flat_dict(dic=check_obs).items():
                    if isinstance(value, np.ndarray):
                        real_obs[key] = ("obs", key, value.dtype, value.shape)
                    else:
                        real_obs[key] = ("obs", key, type(value), "()")

                exp_keys = set(exp_obs.keys())
                real_keys = set(real_obs.keys())
                shared_keys = exp_keys.intersection(real_keys)
                missing_keys = exp_keys - real_keys
                extra_keys = real_keys - exp_keys

                log.error("MISSING OBSERVATION KEYS:")
                log.error(missing_keys)
                log.error("EXTRA OBSERVATION KEYS:")
                log.error(extra_keys)
                log.error("SHARED OBSERVATION KEY DTYPES AND SHAPES:")
                for k in shared_keys:
                    log.error(exp_obs[k])
                    log.error(real_obs[k])

                raise ValueError("Observation space does not match returned observations!")

        return obs, {}

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
    def external_sensors(self):
        """
        Returns:
            None or dict: If self.env_config["external_sensors"] is specified, returns the dict mapping sensor name to
                instantiated sensor. Otherwise, returns None
        """
        return self._external_sensors

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
    def wrapper_config(self):
        """
        Returns:
            dict: Wrapper-specific configuration kwargs
        """
        return self.config["wrapper"]

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
                "action_timestep": 1 / 60.,
                "physics_timestep": 1 / 60.,
                "device": None,
                "automatic_reset": False,
                "flatten_action_space": False,
                "flatten_obs_space": False,
                "initial_pos_z_offset": 0.1,
                "external_sensors": None,
            },

            # Rendering kwargs
            "render": {
                "viewer_width": 1280,
                "viewer_height": 720,
            },

            # Scene kwargs
            "scene": {
                # Traversibility map kwargs
                "waypoint_resolution": 0.2,
                "num_waypoints": 10,
                "build_graph": True,
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
            },

            # Wrapper kwargs
            "wrapper": {
                "type": None,
            },
        }
