import inspect
import logging
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from copy import deepcopy

import gym
import numpy as np

from future.utils import with_metaclass

from igibson.controllers import ControlType, create_controller
# # from igibson.external.pybullet_tools.utils import get_joint_info
from igibson.sensors import create_sensor, SENSOR_PRIMS_TO_SENSOR_CLS, ALL_SENSOR_MODALITIES
from igibson.objects.usd_object import USDObject
from igibson.objects.controllable_object import ControllableObject
from igibson.utils.gym_utils import GymObservable
from igibson.utils.python_utils import classproperty, save_init_info, Registerable
from igibson.utils.utils import rotate_vector_3d

from pxr import UsdPhysics

# Global dicts that will contain mappings
REGISTERED_ROBOTS = OrderedDict()

# Add proprio sensor modality to ALL_SENSOR_MODALITIES
ALL_SENSOR_MODALITIES.add("proprio")


class BaseRobot(USDObject, ControllableObject, GymObservable, Registerable):
    """
    Base class for USD-based robot agents.

    This class handles object loading, and provides method interfaces that should be
    implemented by subclassed robots.
    """
    @save_init_info
    def __init__(
        self,
        # Shared kwargs in hierarchy
        prim_path,
        name=None,
        category="agent",
        class_id=None,
        scale=None,
        rendering_params=None,
        visible=True,
        fixed_base=False,
        visual_only=False,
        self_collisions=False,
        load_config=None,

        # Unique to USDObject hierarchy
        abilities=None,

        # Unique to ControllableObject hierarchy
        control_freq=None,
        controller_config=None,
        action_type="continuous",
        action_normalize=True,
        reset_joint_pos=None,

        # Unique to this class
        obs_modalities="all",
        proprio_obs="default",

        **kwargs,
    ):
        """
        @param prim_path: str, global path in the stage to this object
        @param name: Name for the object. Names need to be unique per scene. If no name is set, a name will be generated
            at the time the object is added to the scene, using the object's category.
        @param category: Category for the object. Defaults to "object".
        @param class_id: What class ID the object should be assigned in semantic segmentation rendering mode.
        @param scale: float or 3-array, sets the scale for this object. A single number corresponds to uniform scaling
            along the x,y,z axes, whereas a 3-array specifies per-axis scaling.
        @param rendering_params: Any relevant rendering settings for this object.
        @param visible: bool, whether to render this object or not in the stage
        @param fixed_base: bool, whether to fix the base of this object or not
        visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
        self_collisions (bool): Whether to enable self collisions for this object
        load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
            loading this prim at runtime.
        @param abilities: dict in the form of {ability: {param: value}} containing
            robot abilities and parameters.
        :param control_freq: float, control frequency (in Hz) at which to control the robot. If set to be None,
            simulator.import_object will automatically set the control frequency to be 1 / render_timestep by default.
        :param controller_config: None or Dict[str, ...], nested dictionary mapping controller name(s) to specific
            controller configurations for this object. This will override any default values specified by this class.
        :param action_type: str, one of {discrete, continuous} - what type of action space to use
        :param action_normalize: bool, whether to normalize inputted actions. This will override any default values
         specified by this class.
        obs_modalities (str or list of str): Observation modalities to use for this robot. Default is "all", which
            corresponds to all modalities being used.
            Otherwise, valid options should be part of igibson.sensors.ALL_SENSOR_MODALITIES.
        :param proprio_obs: str or tuple of str, proprioception observation key(s) to use for generating proprioceptive
            observations. If str, should be exactly "default" -- this results in the default proprioception observations
            being used, as defined by self.default_proprio_obs. See self._get_proprioception_dict for valid key choices
        :param reset_joint_pos: None or Array[float], if specified, should be the joint positions that the robot should
            be set to during a reset. If None (default), self.default_joint_pos will be used instead.
        kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
            for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
        # Store inputs
        self._obs_modalities = obs_modalities if obs_modalities == "all" else \
            {obs_modalities} if isinstance(obs_modalities, str) else set(obs_modalities)              # this will get updated later when we fill in our sensors
        self._proprio_obs = self.default_proprio_obs if proprio_obs == "default" else list(proprio_obs)

        # Process abilities
        robot_abilities = {"robot": {}}
        abilities = robot_abilities if abilities is None else robot_abilities.update(abilities)

        # Initialize internal attributes that will be loaded later
        self._sensors = None                     # e.g.: scan sensor, vision sensor
        self._simulator = None                   # Required for AG by ManipulationRobot

        # Run super init
        super().__init__(
            prim_path=prim_path,
            usd_path=self.model_file,
            name=name,
            category=category,
            class_id=class_id,
            scale=scale,
            rendering_params=rendering_params,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            self_collisions=self_collisions,
            load_config=load_config,
            abilities=abilities,
            control_freq=control_freq,
            controller_config=controller_config,
            action_type=action_type,
            action_normalize=action_normalize,
            reset_joint_pos=reset_joint_pos,
            **kwargs,
        )

    def _post_load(self, simulator=None):
        # Run super post load first
        super()._post_load(simulator=simulator)

        # Search for any sensors this robot might have attached to any of its links
        self._sensors = OrderedDict()
        obs_modalities = set()
        for link_name, link in self._links.items():
            # Search through all children prims and see if we find any sensor
            for prim in link.prim.GetChildren():
                prim_type = prim.GetPrimTypeInfo().GetTypeName()
                if prim_type in SENSOR_PRIMS_TO_SENSOR_CLS:
                    # Infer what obs modalities to use for this sensor
                    sensor_cls = SENSOR_PRIMS_TO_SENSOR_CLS[prim_type]
                    modalities = sensor_cls.all_modalities if self._obs_modalities == "all" else \
                        sensor_cls.all_modalities.intersection(self._obs_modalities)
                    obs_modalities = obs_modalities.union(modalities)
                    # Create the sensor and store it internally
                    sensor = create_sensor(
                        sensor_type=prim_type,
                        prim_path=str(prim.GetPrimPath()),
                        name=f"{self.name}:{link_name}_{prim_type}_sensor",
                        modalities=modalities,
                    )
                    self._sensors[sensor.name] = sensor

        # Since proprioception isn't an actual sensor, we need to possibly manually add it here as well
        if self._obs_modalities == "all":
            obs_modalities.add("proprio")

        # Update our overall obs modalities
        self._obs_modalities = obs_modalities

        # A persistent reference to simulator is needed for AG in ManipulationRobot
        self._simulator = simulator

    def _initialize(self):
        # Run super first
        super()._initialize()

        # Initialize all sensors
        for sensor in self._sensors.values():
            sensor.initialize()

        # Validate this robot configuration
        self._validate_configuration()

    def _validate_configuration(self):
        """
        Run any needed sanity checks to make sure this robot was created correctly.
        """
        pass

    def get_state(self):
        """
        Calculate proprioceptive states for the robot. By default, this is:
            [pos, rpy, lin_vel, ang_vel, joint_states]

        :return Array[float]: Flat array of proprioceptive states (e.g.: [position, orientation, ...])
        """
        # Grab relevant states
        pos = self.get_position()
        rpy = self.get_rpy()
        joints_state = self.get_joints_state(normalized=False).flatten()

        # rotate linear and angular velocities to local frame
        lin_vel = rotate_vector_3d(self.get_linear_velocity(), *rpy)
        ang_vel = rotate_vector_3d(self.get_angular_velocity(), *rpy)

        # Compile and return state
        state = np.concatenate([pos, rpy, lin_vel, ang_vel, joints_state])
        return state

    def can_toggle(self, toggle_position, toggle_distance_threshold):
        """
        Returns True if the part of the robot that can toggle a toggleable is within the given range of a
        point corresponding to a toggle marker
        by default, we assume robot cannot toggle toggle markers

        :param toggle_position: Array[float], (x,y,z) cartesian position values as a reference point for evaluating
            whether a toggle can occur
        :param toggle_distance_threshold: float, distance value below which a toggle is allowed

        :return bool: True if the part of the robot that can toggle a toggleable is within the given range of a
            point corresponding to a toggle marker. By default, we assume robot cannot toggle toggle markers
        """
        return False

    def get_obs(self):
        """
        Grabs all observations from the robot. This is keyword-mapped based on each observation modality
            (e.g.: proprio, rgb, etc.)

        Returns:
            OrderedDict: Keyword-mapped dictionary mapping observation modality names to
                observations (usually np arrays)
        """
        # Our sensors already know what observation modalities it has, so we simply iterate over all of them
        # and grab their observations, processing them into a flat dict
        obs_dict = OrderedDict()
        for sensor_name, sensor in self._sensors.items():
            sensor_obs = sensor.get_obs()
            for obs_modality, obs in sensor_obs.items():
                obs_dict[f"{sensor_name}_{obs_modality}"] = obs

        # Have to handle proprio separately since it's not an actual sensor
        if "proprio" in self._obs_modalities:
            obs_dict["proprio"] = self.get_proprioception()

        return obs_dict

    def get_proprioception(self):
        """
        :return Array[float]: numpy array of all robot-specific proprioceptive observations.
        """
        proprio_dict = self._get_proprioception_dict()
        return np.concatenate([proprio_dict[obs] for obs in self._proprio_obs])

    def dump_config(self):
        # Grab running config
        cfg = super().dump_config()

        # Add relevant robot params
        cfg["proprio_obs"] = self._proprio_obs

        return cfg

    def _get_proprioception_dict(self):
        """
        :return dict: keyword-mapped proprioception observations available for this robot. Can be extended by subclasses
        """
        joints_state = self.get_joints_state(normalized=False)
        pos, ori = self.get_position_orientation()
        return {
            "joint_qpos": joints_state.positions,
            "joint_qpos_sin": np.sin(joints_state.positions),
            "joint_qpos_cos": np.cos(joints_state.positions),
            "joint_qvel": joints_state.velocities,
            "joint_qeffort": joints_state.efforts,
            "robot_pos": pos,
            "robot_rpy": self.get_rpy(),
            "robot_quat": ori,
            "robot_lin_vel": self.get_linear_velocity(),
            "robot_ang_vel": self.get_angular_velocity(),
        }

    def _load_observation_space(self):
        # We compile observation spaces from our sensors
        obs_space = OrderedDict()

        for sensor_name, sensor in self._sensors.items():
            # Load the sensor observation space
            sensor_obs_space = sensor.load_observation_space()
            for obs_modality, obs_modality_space in sensor_obs_space.items():
                obs_space[f"{sensor_name}_{obs_modality}"] = obs_modality_space

        # Have to handle proprio separately since it's not an actual sensor
        if "proprio" in self._obs_modalities:
            obs_space["proprio"] = self._build_obs_box_space(shape=(self.proprioception_dim,), low=-np.inf, high=np.inf)

        return obs_space

    def add_obs_modality(self, modality):
        """
        Adds observation modality @modality to this robot. Note: Should be one of igibson.sensors.ALL_SENSOR_MODALITIES

        Args:
            modality (str): Observation modality to add to this robot
        """
        # Iterate over all sensors we own, and if the requested modality is a part of its possible valid modalities,
        # then we add it
        for sensor in self._sensors.values():
            if modality in sensor.all_modalities:
                sensor.add_modality(modality=modality)

    def remove_obs_modality(self, modality):
        """
        Remove observation modality @modality from this robot. Note: Should be one of
        igibson.sensors.ALL_SENSOR_MODALITIES

        Args:
            modality (str): Observation modality to remove from this robot
        """
        # Iterate over all sensors we own, and if the requested modality is a part of its possible valid modalities,
        # then we remove it
        for sensor in self._sensors.values():
            if modality in sensor.all_modalities:
                sensor.remove_modality(modality=modality)

    @property
    def root_link_name(self):
        # Most robots have base_link as the root_link defined. If it exists, we use that, otherwise, we use the default
        return "base_link" if "base_link" in self._links else super().root_link_name

    @property
    def sensors(self):
        """
        Returns:
            OrderedDict: Keyword-mapped dictionary mapping sensor names to BaseSensor instances owned by this robot
        """
        return self._sensors

    @property
    def obs_modalities(self):
        """
        Returns:
            set of str: Observation modalities used for this robot (e.g.: proprio, rgb, etc.)
        """
        assert self._loaded, "Cannot check observation modalities until we load this robot!"
        return self._obs_modalities

    @property
    def proprioception_dim(self):
        """
        :return int: Size of self.get_proprioception() vector
        """
        return len(self.get_proprioception())

    # TODO: Bake-in camera into this link
    # # TODO!! Maybe create a camera prim? cf. Alan
    # @property
    # def eyes(self):
    #     """
    #     Returns the RobotLink corresponding to the robot's camera. Assumes that there is a link
    #     with name "eyes" in the underlying robot model. If not, an error will be raised.
    #
    #     :return RobotLink: link containing the robot's camera
    #     """
    #     assert "eyes" in self._links, "Cannot find 'eyes' in links, current link names are: {}".format(
    #         list(self._links.keys())
    #     )
    #     return self._links["eyes"]

    @property
    def default_proprio_obs(self):
        """
        :return Array[str]: Default proprioception observations to use
        """
        return []

    @property
    def model_name(self):
        """
        Returns:
            str: name of this robot model. usually corresponds to the class name of a given robot model
        """
        return self.__class__.__name__

    @property
    @abstractmethod
    def model_file(self):
        """
        :return str: absolute path to robot model's URDF / MJCF file
        """
        raise NotImplementedError

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("BaseRobot")
        return classes

    @classproperty
    def _cls_registry(cls):
        # Global robot registry
        global REGISTERED_ROBOTS
        return REGISTERED_ROBOTS
