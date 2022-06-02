import gym
from collections import OrderedDict

# from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
# from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.robots import REGISTERED_ROBOTS
from igibson.scenes import REGISTERED_SCENES
from igibson.simulator_omni import Simulator
# from igibson.simulator_vr import SimulatorVR
from igibson.utils.gym_utils import GymObservable
from igibson.utils.utils import parse_config
from igibson.utils.python_utils import merge_nested_dicts, create_class_from_registry_and_config, Serializable, Recreatable


class BaseEnv(gym.Env, GymObservable, Serializable, Recreatable):
    """
    Base Env class that handles loading scene and robot, following OpenAI Gym interface.
    Functions like reset and step are not implemented.
    """

    def __init__(
        self,
        configs,
        scene_model=None,
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
        rendering_settings=None,
        vr=False,
        vr_settings=None,
        device_idx=0,
    ):
        """
        :param configs (str or list of str): config_file path(s). If multiple configs are specified, they will
            be merged sequentially in the order specified. This allows procedural generation of a "full" config from
            small sub-configs.
        :param scene_model: override scene_model in config file
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param rendering_settings: rendering_settings to override the default one
        vr (bool): Whether we're using VR or not
        :param vr_settings: vr_settings to override the default one
        :param device_idx: device_idx: which GPU to run the simulation and rendering on
        """
        # Call super first
        super().__init__()

        # Convert config file(s) into a single parsed dict
        configs = [configs] if isinstance(configs, str) else configs

        # Initial default config
        self.config = self.default_config

        # Merge in specified configs
        for config in configs:
            merge_nested_dicts(base_dict=self.config, extra_dict=parse_config(config), inplace=True)

        # Possibly override scene_id
        if scene_model is not None:
            self.scene_config["model"] = scene_model

        # Store other settings
        self.action_timestep = action_timestep

        # TODO! Update
        self.rendering_settings = None #MeshRendererSettings()
        # TODO!
        # # TODO: We currently only support the optimized renderer due to some issues with obj highlighting.
        # self.rendering_settings = rendering_settings if rendering_settings is not None else \
        #      MeshRendererSettings(
        #         enable_shadow=enable_shadow,
        #         enable_pbr=enable_pbr,
        #         msaa=False,
        #         texture_scale=texture_scale,
        #         optimized=self.config["optimized_renderer"],
        #         load_textures=self.config["load_texture"],
        #         hide_robot=self.config["hide_robot"],
        #     )

        # Create other placeholders that will be filled in later
        self._scene = None
        self._loaded = None

        # Create simulator
        if vr:
            raise NotImplementedError("VR must still be implemented")
            if vr_settings is None:
                vr_settings = VrSettings(use_vr=True)
            self._simulator = SimulatorVR(
                physics_timestep=physics_timestep,
                render_timestep=action_timestep,
                image_width=self.config["image_width"],
                image_height=self.config["image_height"],
                vertical_fov=self.config["vertical_fov"],
                device_idx=device_idx,
                rendering_settings=self.rendering_settings,
                vr_settings=vr_settings,
                use_pb_gui=use_pb_gui,
            )
        else:
            self._simulator = Simulator(
                physics_dt=physics_timestep,
                rendering_dt=action_timestep,
                viewer_width=self.render_config["viewer_width"],
                viewer_height=self.render_config["viewer_height"],
                vertical_fov=self.render_config["vertical_fov"],
                device_idx=device_idx,
            )

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

        :param scene_model: new scene model to load (eg.: Rs_int)
        """
        self.scene_config["model"] = scene_model
        self.load()

    def _load_variables(self):
        """
        Load variables from config
        """
        # Default is no-op
        pass

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
        self._simulator.import_scene(scene)

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
                self._simulator.import_object(robot)

    def _load(self):
        """
        Loads this environment. Can be extended by subclass
        """
        # Load config variables
        self._load_variables()

        # Load the scene and robots
        self._load_scene()
        self._load_robots()

    def load(self):
        """
        Load the scene and robot specified in the config file.
        """
        # This environment is not loaded
        self._loaded = False

        # Run internal loading
        self._load()

        # Denote that the scene is loaded
        self._loaded = True

    def clean(self):
        """
        Clean up the environment.
        """
        if self._simulator is not None:
            self._simulator.close()

    def get_obs(self):
        """
        Get the current environment observation.

        Returns:
            OrderedDict: Keyword-mapped observations, which are possibly nested
        """
        # Grab all observations from each robot
        obs = OrderedDict()

        for robot in self.robots:
            obs[robot.name] = robot.get_obs()

        return obs

    def close(self):
        """
        Synonymous function with clean.
        """
        self.clean()

    def _simulator_step(self):
        """
        Step the simulation.
        This is different from environment step that returns the next
        observation, reward, done, info. This should exclusively run internal simulator stepping and related
        functionality
        """
        self._simulator.step()

    def step(self, action):
        """
        Overwritten by subclasses.
        """
        return NotImplementedError()

    def reset(self):
        """
        Overwritten by subclasses.

        Returns:
            OrderedDict: Observations after resetting
        """
        return NotImplementedError()

    @property
    def simulator(self):
        """
        Returns:
            Simulator: Active simulator instance
        """
        return self._simulator

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
    def default_config(self):
        """
        Returns:
            dict: Default configuration for this environment. May not be fully specified (i.e.: still requires @config
                to be specified during environment creation)
        """
        return {
            # Environment kwargs
            "env": {
                # none by default
            },

            # Rendering kwargs
            "render": {
                "viewer_width": 128,
                "viewer_height": 128,
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
        }

    @property
    def state_size(self):
        # Total state size is the state size of our scene
        return self._scene.state_size

    def _dump_state(self):
        # Default state is from the scene
        return self._scene.dump_state(serialized=False)

    def _load_state(self, state):
        # Default state is from the scene
        self._scene.load_state(state=state, serialized=False)

    def load_state(self, state, serialized=False):
        # Run super first
        super().load_state(state=state, serialized=serialized)

        # # We also need to manually update the simulator app
        # self._simulator.app.update()

    def _serialize(self, state):
        # Default state is from the scene
        return self._scene.serialize(state=state)

    def _deserialize(self, state):
        # Default state is from the scene
        end_idx = self._scene.state_size
        return self._scene.deserialize(state=state[:end_idx]), end_idx
