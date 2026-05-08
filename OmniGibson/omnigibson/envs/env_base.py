from collections import OrderedDict
from collections.abc import Iterable
from copy import deepcopy

import gymnasium as gym
import torch as th

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.objects import REGISTERED_OBJECTS
from omnigibson.robots import Robot, REGISTERED_ROBOTS
from omnigibson.scene_graphs.graph_builder import SceneGraphBuilder
from omnigibson.scenes import REGISTERED_SCENES
from omnigibson.sensors import VisionSensor, create_sensor
from omnigibson.tasks import REGISTERED_TASKS
from omnigibson.utils.config_utils import parse_config
from omnigibson.utils.gym_utils import (
    GymObservable,
    maxdim,
    recursively_generate_compatible_dict,
    recursively_generate_flat_dict,
)
from omnigibson.utils.numpy_utils import NumpyTypes, list_to_np_array
from omnigibson.utils.python_utils import (
    Recreatable,
    assert_valid_key,
    create_class_from_registry_and_config,
    merge_nested_dicts,
)
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.vision_utils import change_pcd_frame

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

        # Required render mode metadata for gymnasium
        self.render_mode = "rgb_array"
        self.metadata = {"render.modes": ["rgb_array"]}

        # Convert config file(s) into a single parsed dict
        configs = configs if isinstance(configs, list) or isinstance(configs, tuple) else [configs]

        # Initial default config
        self.config = self.default_config

        # Merge in specified configs
        for config in configs:
            merge_nested_dicts(base_dict=self.config, extra_dict=parse_config(config), inplace=True)

        # Store number of environments
        self.num_envs = self.env_config.get("num_envs", 1)

        # Store settings and other initialized values
        self._automatic_reset = self.env_config["automatic_reset"]
        self._flatten_action_space = self.env_config["flatten_action_space"]
        self._flatten_obs_space = self.env_config["flatten_obs_space"]
        self.device = self.env_config["device"] if self.env_config["device"] else "cpu"

        physics_dt = 1.0 / self.env_config["physics_frequency"]
        rendering_dt = 1.0 / self.env_config["rendering_frequency"]
        sim_step_dt = 1.0 / self.env_config["action_frequency"]
        viewer_width = self.render_config["viewer_width"]
        viewer_height = self.render_config["viewer_height"]

        # If the sim is launched, check that the parameters match
        if og.sim is not None:
            assert (
                og.sim.initial_physics_dt == physics_dt
            ), f"Physics frequency mismatch! Expected {physics_dt}, got {og.sim.initial_physics_dt}"
            assert (
                og.sim.initial_rendering_dt == rendering_dt
            ), f"Rendering frequency mismatch! Expected {rendering_dt}, got {og.sim.initial_rendering_dt}"
            assert og.sim.device == self.device, f"Device mismatch! Expected {self.device}, got {og.sim.device}"
            assert (
                og.sim.viewer_width == viewer_width
            ), f"Viewer width mismatch! Expected {viewer_width}, got {og.sim.viewer_width}"
            assert (
                og.sim.viewer_height == viewer_height
            ), f"Viewer height mismatch! Expected {viewer_height}, got {og.sim.viewer_height}"
        # Otherwise, launch a simulator instance
        else:
            og.launch(
                physics_dt=physics_dt,
                rendering_dt=rendering_dt,
                sim_step_dt=sim_step_dt,
                device=self.device,
                viewer_width=viewer_width,
                viewer_height=viewer_height,
            )

        # Initialize other placeholders that will be filled in later
        self._task = None
        self._external_sensors = None
        self._external_sensors_include_in_obs = None
        self._loaded = None
        self._current_episodes = th.zeros(self.num_envs, dtype=th.int32)

        # Variables reset at the beginning of each episode
        self._current_steps = th.zeros(self.num_envs, dtype=th.int32)

        # Scene list
        self._scenes = []

        # Sensor registry for obs-key-first iteration (built in load)
        self._robot_sensor_map = {}  # (robot_name, sensor_name) -> [sensor_per_env]
        self._robot_has_proprio = {}  # robot_name -> bool
        self._ext_sensor_map = {}  # sensor_name -> [sensor_per_env]

        # Create the scene graph builders (one per scene)
        self._scene_graph_builders = []
        if "scene_graph" in self.config and self.config["scene_graph"] is not None:
            self._scene_graph_builders = [SceneGraphBuilder(**self.config["scene_graph"]) for _ in range(self.num_envs)]

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
        # Reset bookkeeping variables
        self._reset_variables()
        self._current_episodes = th.zeros(self.num_envs, dtype=th.int32)

        # - Potentially overwrite the USD entry for the scene if none is specified and we're online sampling -

        # Make sure the requested scene is valid
        scene_type = self.scene_config["type"]
        assert_valid_key(key=scene_type, valid_keys=REGISTERED_SCENES, name="scene type")

        # Verify scene and task configs are valid for the given task type
        REGISTERED_TASKS[self.task_config["type"]].verify_scene_and_task_config(
            scene_cfg=self.scene_config,
            task_cfg=self.task_config,
        )

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

        # Create the scene(s) from our scene config
        self._scenes = [
            create_class_from_registry_and_config(
                cls_name=self.scene_config["type"],
                cls_registry=REGISTERED_SCENES,
                cfg=deepcopy(self.scene_config),
                cls_type_descriptor="scene",
            )
            for _ in range(self.num_envs)
        ]
        og.sim.import_scene(self._scenes)

        assert og.sim.is_stopped(), "Simulator must be stopped after loading scene!"

    def _load_robots(self):
        """
        Load robots into the scene
        """
        # Only actually load robots if no robot has been imported from the scene loading directly yet
        loaded_from_config = False

        # Pre-assign robot names so all scenes share the same names
        for i, robot_config in enumerate(self.robots_config):
            # Add a name for the robot if necessary. Make sure robots in different scenes share the same relative_prim_path
            if "name" not in robot_config:
                robot_config["name"] = f"robot_{i}"

        for scene in self._scenes:
            if len(scene.robots) == 0:
                loaded_from_config = True
                assert og.sim.is_stopped(), "Simulator must be stopped before loading robots!"

                # Iterate over all robots to generate in the robot config
                for robot_config in self.robots_config:
                    robot_config = deepcopy(robot_config)
                    if "model" in robot_config:
                        assert (
                            "type" not in robot_config
                        ), "CANNOT SPECIFY BOTH TYPE AND MODEL. Robot config key 'type' is deprecated; use 'model' instead."
                    elif "type" in robot_config:
                        log.warning(
                            "Robot config key 'type' is deprecated; use 'model' instead. "
                            "Model IDs are lowercase (e.g. 'model': 'fetch'). "
                        )
                        robot_config["model"] = robot_config["type"].lower()
                        del robot_config["type"]
                    assert (
                        robot_config["model"] in REGISTERED_ROBOTS
                    ), f"{robot_config['model']} is not a registered robot."
                    position, orientation = robot_config.pop("position", None), robot_config.pop("orientation", None)
                    pose_frame = robot_config.pop("pose_frame", "scene")
                    if position is not None:
                        position = (
                            position if isinstance(position, th.Tensor) else th.tensor(position, dtype=th.float32)
                        )
                    if orientation is not None:
                        orientation = (
                            orientation
                            if isinstance(orientation, th.Tensor)
                            else th.tensor(orientation, dtype=th.float32)
                        )

                    robot = Robot(**robot_config)
                    # Import the robot into the simulator
                    scene.add_object(robot)
                    robot.set_position_orientation(position=position, orientation=orientation, frame=pose_frame)

        # Persist scene 0's robot names back to config for downstream use (e.g. data collection).
        # Only needed when robots were created from config (ordering matches). When robots come
        # from a scene file, names are already correct and ordering may not match config.
        if loaded_from_config:
            for robot_idx, robot in enumerate(self._scenes[0].robots):
                if robot_idx < len(self.robots_config):
                    self.robots_config[robot_idx]["name"] = robot.name

        assert og.sim.is_stopped(), "Simulator must be stopped after loading robots!"

    def _load_objects(self):
        """
        Load any additional custom objects into the scene
        """
        assert og.sim.is_stopped(), "Simulator must be stopped before loading objects!"
        for scene in self._scenes:
            for i, obj_config in enumerate(self.objects_config):
                obj_config = deepcopy(obj_config)
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
                scene.add_object(obj)
                obj.set_position_orientation(position=position, orientation=orientation, frame="scene")

        assert og.sim.is_stopped(), "Simulator must be stopped after loading objects!"

    def _load_external_sensors(self):
        """
        Load any additional custom external sensors into the scene
        """
        assert og.sim.is_stopped(), "Simulator must be stopped before loading external sensors!"
        sensors_config = self.env_config["external_sensors"]
        if sensors_config is not None:
            self._external_sensors = []
            self._external_sensors_include_in_obs = dict()
            for i, sensor_config in enumerate(sensors_config):
                # Add a name for the object if necessary
                if "name" not in sensor_config:
                    sensor_config["name"] = f"external_sensor{i}"
                # Determine prim path if not specified
                if "relative_prim_path" not in sensor_config:
                    sensor_config["relative_prim_path"] = f"/{sensor_config['name']}"
            # Create separate sensor instances for each scene
            for scene_idx, scene in enumerate(self._scenes):
                scene_sensors = dict()
                for i, sensor_config in enumerate(sensors_config):
                    sc = deepcopy(sensor_config)
                    # Pop the desired position and orientation
                    position, orientation = sc.pop("position", None), sc.pop("orientation", None)
                    pose_frame = sc.pop("pose_frame", "scene")
                    # Pop whether or not to include this sensor in the observation
                    include_in_obs = sc.pop("include_in_obs", True)
                    # Make sure sensor exists, grab its corresponding kwargs, and create the sensor
                    sensor = create_sensor(**sc)
                    # Load an initialize this sensor
                    sensor.load(scene)
                    sensor.initialize()
                    sensor.set_position_orientation(position=position, orientation=orientation, frame=pose_frame)
                    scene_sensors[sensor.name] = sensor
                    # Populate include_in_obs only once (same for all scenes)
                    if scene_idx == 0:
                        self._external_sensors_include_in_obs[sensor.name] = include_in_obs
                self._external_sensors.append(scene_sensors)

        assert og.sim.is_stopped(), "Simulator must be stopped after loading external sensors!"

    def _load_observation_space(self):
        # Grab robot(s) and task obs spaces
        # Assumes all scenes share the same robot/sensor configuration
        obs_space = dict()

        for robot in self._scenes[0].robots:
            # Load the observation space for the robot
            robot_obs = robot.load_observation_space()
            if maxdim(robot_obs) > 0:
                obs_space[robot.name] = robot_obs

        # Also load the task obs space
        task_space = self._task.load_observation_space()
        if maxdim(task_space) > 0:
            obs_space["task"] = task_space

        # Also load any external sensors
        if self._external_sensors is not None:
            external_obs_space = dict()
            for sensor_name, sensor in self._external_sensors[0].items():
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
        action_space = gym.spaces.Dict({robot.name: robot.action_space for robot in self._scenes[0].robots})

        # Convert into flattened 1D Box space if requested
        if self._flatten_action_space:
            lows = []
            highs = []
            for space in action_space.values():
                assert isinstance(
                    space, gym.spaces.Box
                ), "Can only flatten action space where all individual spaces are gym.space.Box instances!"
                assert (
                    len(space.shape) == 1
                ), "Can only flatten action space where all individual spaces are 1D instances!"
                lows.append(space.low)
                highs.append(space.high)
            action_space = gym.spaces.Box(
                list_to_np_array(lows),
                list_to_np_array(highs),
                dtype=NumpyTypes.FLOAT32,
            )

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
        self._load_objects()
        self._load_robots()
        self._load_task()
        self._load_external_sensors()

        # Play and complete loading
        og.sim.play()
        # Run any additional task post-loading behavior
        self.task.post_play_load(env=self)

        # Save the state for objects from load_robots / load_objects / load_task
        for scene in self._scenes:
            scene.update_initial_file()

        # Load the obs / action spaces
        self.load_observation_space()
        self._load_action_space()

        # Build sensor registry for get_obs()
        self._build_sensor_registry()

        self.reset()

        # Start the scene graph builders
        for builder, scene in zip(self._scene_graph_builders, self._scenes):
            builder.start(scene)

        # Denote that the scene is loaded
        self._loaded = True

    def close(self):
        """No-op to satisfy certain RL frameworks."""
        pass

    def _build_sensor_registry(self):
        """
        Build cross-scene sensor maps for obs-key-first observation gathering.
        All scenes share the same robot/sensor configuration, so scene 0 is used as the template.
        """
        self._robot_sensor_map = {}
        self._robot_has_proprio = {}

        for robot_idx, robot in enumerate(self._scenes[0].robots):
            self._robot_has_proprio[robot.name] = "proprio" in robot._obs_modalities
            for sensor_name in robot.sensors:
                self._robot_sensor_map[(robot.name, sensor_name)] = [
                    self._scenes[env_idx].robots[robot_idx].sensors[sensor_name] for env_idx in range(self.num_envs)
                ]

        self._ext_sensor_map = {}
        if self._external_sensors is not None:
            for sensor_name in self._external_sensors[0]:
                if not self._external_sensors_include_in_obs.get(sensor_name, False):
                    continue
                self._ext_sensor_map[sensor_name] = [
                    self._external_sensors[env_idx][sensor_name] for env_idx in range(self.num_envs)
                ]

    def get_obs(self, env_indices=None):
        """
        Get the current environment observation.

        Args:
            env_indices (None or th.Tensor): Indices of envs to get observations for. If None, gets all.

        Returns:
            2-tuple:
                list[dict]: Keyword-mapped observations per env
                list[dict]: Additional information about the observations per env
        """
        if env_indices is None:
            env_indices = list(range(self.num_envs))
        else:
            env_indices = [int(i) for i in env_indices]

        # Pre-allocate result dicts for each requested env
        all_obs = [{} for _ in env_indices]
        all_info = [{} for _ in env_indices]

        # --- Robot observations ---
        for robot_idx, robot_0 in enumerate(self._scenes[0].robots):
            robot_name = robot_0.name
            if maxdim(robot_0.observation_space) <= 0:
                continue

            # Initialize per-env robot obs/info dicts
            for i in range(len(env_indices)):
                all_obs[i][robot_name] = {}
                all_info[i][robot_name] = {}

            # Sensor observations
            for (rname, sensor_name), sensors in self._robot_sensor_map.items():
                if rname != robot_name:
                    continue
                # TODO: Replace with a single batched tiled rendering call
                for i, env_idx in enumerate(env_indices):
                    s_obs, s_info = sensors[env_idx].get_obs()
                    # Convert pointcloud from world frame to robot base frame
                    for key in s_obs:
                        if "pointcloud" in key:
                            robot = self._scenes[env_idx].robots[robot_idx]
                            s_obs[key] = change_pcd_frame(
                                pcd=s_obs[key],
                                rel_pose=th.cat(robot.get_position_orientation()),
                            )
                    all_obs[i][robot_name][sensor_name] = s_obs
                    all_info[i][robot_name][sensor_name] = s_info

            # Proprioception
            if self._robot_has_proprio.get(robot_name, False):
                for i, env_idx in enumerate(env_indices):
                    robot = self._scenes[env_idx].robots[robot_idx]
                    all_obs[i][robot_name]["proprio"], all_info[i][robot_name]["proprio"] = robot.get_proprioception()

        # --- Task observations ---
        if maxdim(self._task.observation_space) > 0:
            for i, env_idx in enumerate(env_indices):
                all_obs[i]["task"] = self._task.get_obs(env=self, env_idx=env_idx)

        # --- External sensor observations ---
        if self._external_sensors is not None:
            for i in range(len(env_indices)):
                all_obs[i]["external"] = {}
                all_info[i]["external"] = {}

            for sensor_name, sensors in self._ext_sensor_map.items():
                # TODO: Replace with a single batched tiled rendering call
                for i, env_idx in enumerate(env_indices):
                    s_obs, s_info = sensors[env_idx].get_obs()
                    all_obs[i]["external"][sensor_name] = s_obs
                    all_info[i]["external"][sensor_name] = s_info

        # --- Flatten if requested ---
        if self._flatten_obs_space:
            for i in range(len(env_indices)):
                all_obs[i] = recursively_generate_flat_dict(dic=all_obs[i])

        return all_obs, all_info

    def get_scene_graph(self, env_idx=0):
        """
        Get the current scene graph for a given environment.

        Args:
            env_idx (int): Index of the environment to get the scene graph for. Default is 0.

        Returns:
            SceneGraph: Current scene graph
        """
        assert self._scene_graph_builders, "Scene graph builder must be specified in config!"
        return self._scene_graph_builders[env_idx].get_scene_graph()

    def _populate_info(self, infos):
        """
        Populate info dictionary with any useful information.

        Args:
            infos (list[dict]): Information dictionaries to populate, one per env

        Returns:
            list[dict]: Information dictionaries with added info
        """
        for env_idx in range(self.num_envs):
            infos[env_idx]["episode_length"] = self._current_steps[env_idx].item()

        for i, builder in enumerate(self._scene_graph_builders):
            infos[i]["scene_graph"] = builder.get_scene_graph()

    def _convert_action_dict_to_tensor(self, action_dict):
        """Convert a single action dict's values to tensors.

        Args:
            action_dict (dict): Maps robot name to action (array, list, or tensor).

        Returns:
            dict: Same keys, with values converted to flattened float tensors.
        """
        return {
            k: th.as_tensor(v, dtype=th.float).flatten()
            if isinstance(v, Iterable) and not isinstance(v, (dict, OrderedDict, str))
            else v
            for k, v in action_dict.items()
        }

    def _convert_action_to_tensor(self, action):
        """Convert action to torch tensor format.

        Args:
            action: Action in various formats (dict, list of dicts, numpy array, list, torch tensor).
                    For tensors/arrays, should be (num_envs, action_dim) shaped.
                    A flat (action_dim,) tensor is accepted for single-env and reshaped to (1, action_dim).
                    For dicts, should be a list of dicts (one per env), or a single dict for single-env.

        Returns:
            th.Tensor of shape (num_envs, action_dim), or list of dicts (one per env).
        """
        if isinstance(action, dict):
            return [self._convert_action_dict_to_tensor(action)]
        elif isinstance(action, list) and len(action) > 0 and isinstance(action[0], dict):
            return [self._convert_action_dict_to_tensor(a) for a in action]
        elif isinstance(action, Iterable):
            # Convert numpy arrays and lists to tensors
            action = th.as_tensor(action, dtype=th.float)
            if action.dim() == 1:
                action = action.unsqueeze(0)
            return action
        return action

    def step(self, action, n_render_iterations=1):
        """
        Apply robot's action and return the next state, reward, done and info,
        following OpenAI Gym's convention

        Args:
            action (gym.spaces.Dict or dict or list[dict] or th.tensor): robot actions. If a dict is specified,
                each entry should map robot name to corresponding action. If a th.tensor, it should be the flattened,
                concatenated set of actions. For multi-env, should be (num_envs, action_dim) shaped for tensors,
                or a list of dicts (one per env) for dict actions.
            n_render_iterations (int): Number of rendering iterations to use before returning observations

        Returns:
            5-tuple:
                - list[dict]: states, i.e. next observations per env
                - th.Tensor: (num_envs,) rewards, i.e. reward at this current timestep
                - th.Tensor: (num_envs,) terminated bool, i.e. whether this episode ended due to a failure or success
                - th.Tensor: (num_envs,) truncated bool, i.e. whether this episode ended due to a time limit etc.
                - list[dict]: info per env, i.e. dictionary with any useful information
        """
        # Pre-processing before stepping simulation
        action = self._convert_action_to_tensor(action)
        if action is not None:
            for env_idx in range(self.num_envs):
                env_action = action[env_idx]

                scene = self._scenes[env_idx]

                # If the action is not a dictionary, convert into a dictionary
                if not isinstance(env_action, dict) and not isinstance(env_action, gym.spaces.Dict):
                    action_dict = dict()
                    idx = 0
                    for robot in scene.robots:
                        action_dim = robot.action_dim
                        action_dict[robot.name] = env_action[idx : idx + action_dim]
                        idx += action_dim
                else:
                    # Our inputted action is the action dictionary
                    action_dict = env_action

                # Iterate over all robots and apply actions
                for robot in scene.robots:
                    robot.apply_action(action_dict[robot.name])

        # Step simulation
        og.sim.step()

        # Render any additional times requested
        for _ in range(n_render_iterations - 1):
            og.sim.render()

        # Grab observations
        obs_list, obs_info_list = self.get_obs()

        # Step the scene graph builders if necessary
        for builder, scene in zip(self._scene_graph_builders, self._scenes):
            builder.step(scene)

        # Grab reward, done, and info, and populate with internal info
        rewards, dones, infos = self.task.step(self, action)
        self._populate_info(infos)
        for env_idx in range(self.num_envs):
            infos[env_idx]["obs_info"] = obs_info_list[env_idx]

        # Split terminated vs truncated per env
        terminateds = th.zeros(self.num_envs, dtype=th.bool)
        truncateds = th.zeros(self.num_envs, dtype=th.bool)
        for env_idx in range(self.num_envs):
            for tc_name, tc_data in infos[env_idx]["done"]["termination_conditions"].items():
                if tc_data["done"]:
                    if tc_name == "timeout":
                        truncateds[env_idx] = True
                    else:
                        terminateds[env_idx] = True

        assert th.all((terminateds | truncateds) == dones), "Terminated and truncated must match done!"

        # Auto-reset only done envs
        done_mask = dones
        if self._automatic_reset and done_mask.any():
            done_indices = th.where(done_mask)[0]
            for env_idx in done_indices:
                # Add lost observation to our information dict, and reset
                infos[env_idx.item()]["last_observation"] = obs_list[env_idx.item()]
            reset_obs_list, reset_info = self.reset(env_indices=done_indices)
            for i, env_idx in enumerate(done_indices):
                obs_list[env_idx.item()] = reset_obs_list[i]

        # Increment step
        self._current_steps += 1
        return obs_list, rewards, terminateds, truncateds, infos

    def render(self):
        """Render the environment for debug viewing."""
        # Only works if there is an external sensor
        if not self._external_sensors:
            return None

        # Get the RGB sensors
        rgb_sensors = [
            x
            for x in self._external_sensors[0].values()
            if isinstance(x, VisionSensor) and (x.modalities == "all" or "rgb" in x.modalities)
        ]
        if not rgb_sensors:
            return None

        # Render the external sensor
        og.sim.render()

        # Grab the rendered image from each of the rgb sensors, concatenate along dim 1
        rgb_images = [sensor.get_obs()[0]["rgb"] for sensor in rgb_sensors]
        return th.cat(rgb_images, dim=1)[:, :, :3]

    def _reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode.
        """
        self._current_episodes += 1
        self._current_steps[:] = 0

    def reset(self, env_indices=None, get_obs=True, **kwargs):
        """
        Reset episode.

        Args:
            env_indices (None or th.Tensor): Indices of envs to reset. If None, resets all.
            get_obs (bool): Whether to return observations after reset.
        """
        if env_indices is None:
            env_indices = th.arange(self.num_envs)

        # Reset the task
        self.task.reset(self, env_indices=env_indices)

        # Reset internal variables
        self._current_episodes[env_indices] += 1
        self._current_steps[env_indices] = 0

        if get_obs:
            # Run a single simulator step and a replicator step
            og.sim.step()
            # Render 3 times to make sure we can grab updated observations
            for _ in range(3):
                og.sim.render()
            # Grab and return observations
            obs_list, info_list = self.get_obs(env_indices=env_indices)

            if self._loaded:
                # Sanity check to make sure received observations match expected observation space
                check_obs = recursively_generate_compatible_dict(dic=obs_list[0])
                if not self.observation_space.contains(check_obs):
                    exp_obs = dict()
                    for key, value in recursively_generate_flat_dict(dic=self.observation_space).items():
                        exp_obs[key] = ("obs_space", key, value.dtype, value.shape)
                    real_obs = dict()
                    for key, value in recursively_generate_flat_dict(dic=check_obs).items():
                        if isinstance(value, th.Tensor):
                            real_obs[key] = ("obs", key, value.dtype, value.shape)
                        else:
                            real_obs[key] = ("obs", key, type(value), "()")

                    exp_keys = set(exp_obs.keys())
                    real_keys = set(real_obs.keys())
                    shared_keys = exp_keys.intersection(real_keys)
                    missing_keys = exp_keys - real_keys
                    extra_keys = real_keys - exp_keys

                    if missing_keys:
                        log.error("MISSING OBSERVATION KEYS:")
                        log.error(missing_keys)
                    if extra_keys:
                        log.error("EXTRA OBSERVATION KEYS:")
                        log.error(extra_keys)

                    mismatched_keys = []
                    for k in shared_keys:
                        if exp_obs[k][2:] != real_obs[k][2:]:  # Compare dtypes and shapes
                            mismatched_keys.append(k)
                            log.error(f"MISMATCHED OBSERVATION FOR KEY '{k}':")
                            log.error(f"Expected: {exp_obs[k]}")
                            log.error(f"Received: {real_obs[k]}")

                    raise ValueError("Observation space does not match returned observations!")

            return obs_list, {"obs_info": info_list}

    @property
    def episode_steps(self):
        """
        Returns:
            th.Tensor: (num_envs,) current step count per env
        """
        return self._current_steps

    @property
    def task(self):
        """
        Returns:
            BaseTask: Active task instance
        """
        return self._task

    @property
    def scenes(self):
        """
        Returns:
            list[Scene]: All scene instances in this environment
        """
        return self._scenes

    @property
    def scene(self):
        """
        Returns:
            Scene: Active scene in this environment (first scene, for backward compatibility)
        """
        assert (
            len(self._scenes) == 1
        ), "The legacy 'env.scene' property is only supported for single-scene environments!"
        return self._scenes[0]

    @property
    def robots(self):
        """
        Returns:
            list[list[BaseRobot]]: Robots per scene. robots[env_idx] -> list of robots in that scene.
        """
        return [s.robots for s in self._scenes]

    @property
    def external_sensors(self):
        """
        Returns:
            None or list[dict]: If self.env_config["external_sensors"] is specified, returns a list of dicts
                (one per scene), each mapping sensor name to instantiated sensor. Otherwise, returns None
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
                "action_frequency": gm.DEFAULT_SIM_STEP_FREQ,
                "rendering_frequency": gm.DEFAULT_RENDERING_FREQ,
                "physics_frequency": gm.DEFAULT_PHYSICS_FREQ,
                "device": None,
                "automatic_reset": False,
                "flatten_action_space": False,
                "flatten_obs_space": False,
                "external_sensors": None,
                "num_envs": 1,
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
                "trav_map_resolution": 0.1,
                "default_erosion_radius": 0.0,
                "trav_map_with_objects": True,
                "scene_instance": None,
                "scene_file": None,
            },
            # Robot kwargs
            "robots": [],  # no robots by default
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
