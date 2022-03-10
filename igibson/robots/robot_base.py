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
from igibson.objects.usd_object import USDObject
from igibson.objects.controllable_object import ControllableObject
from igibson.utils.python_utils import assert_valid_key, merge_nested_dicts
from igibson.utils.utils import rotate_vector_3d

from pxr import UsdPhysics

# Global dicts that will contain mappings
REGISTERED_ROBOTS = {}
ROBOT_TEMPLATE_CLASSES = {
    "BaseRobot",
    "ActiveCameraRobot",
    "TwoWheelRobot",
    "ManipulationRobot",
    "LocomotionRobot",
}


def register_robot(cls):
    if cls.__name__ not in REGISTERED_ROBOTS and cls.__name__ not in ROBOT_TEMPLATE_CLASSES:
        REGISTERED_ROBOTS[cls.__name__] = cls


class BaseRobot(USDObject, ControllableObject):
    """
    Base class for USD-based robot agents.

    This class handles object loading, and provides method interfaces that should be
    implemented by subclassed robots.
    """

    def __init_subclass__(cls, **kwargs):
        """
        Registers all subclasses as part of this registry. This is useful to decouple internal codebase from external
        user additions. This way, users can add their custom robot by simply extending this Robot class,
        and it will automatically be registered internally. This allows users to then specify their robot
        directly in string-from in e.g., their config files, without having to manually set the str-to-class mapping
        in our code.
        """
        if not inspect.isabstract(cls):
            register_robot(cls)

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
        :param proprio_obs: str or tuple of str, proprioception observation key(s) to use for generating proprioceptive
            observations. If str, should be exactly "default" -- this results in the default proprioception observations
            being used, as defined by self.default_proprio_obs. See self._get_proprioception_dict for valid key choices
        :param reset_joint_pos: None or Array[float], if specified, should be the joint positions that the robot should
            be set to during a reset. If None (default), self.default_joint_pos will be used instead.
        kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
            for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
        # Store inputs
        self._proprio_obs = self.default_proprio_obs if proprio_obs == "default" else list(proprio_obs)

        # Process abilities
        robot_abilities = {"robot": {}}
        abilities = robot_abilities if abilities is None else robot_abilities.update(abilities)

        # Initialize internal attributes that will be loaded later
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

        # A persistent reference to simulator is needed for AG in ManipulationRobot
        self._simulator = simulator

    def _initialize(self):
        # Run super first
        super()._initialize()

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
