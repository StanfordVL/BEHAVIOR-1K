import logging
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from copy import deepcopy

import gym
import numpy as np

from future.utils import with_metaclass

from igibson.controllers import ControlType, create_controller
from igibson.external.pybullet_tools.utils import get_joint_info
from igibson.object_states.factory import get_state_name, prepare_object_states
from igibson.object_states.object_state_base import AbsoluteObjectState
from igibson.object_states.utils import clear_cached_states
from igibson.utils.constants import SemanticClass
from igibson.utils.python_utils import assert_valid_key, merge_nested_dicts
from igibson.utils.utils import rotate_vector_3d

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


class BaseRobot(with_metaclass(ABCMeta, object)):
    """
    Base class for mujoco xml/ROS urdf based robot agents.

    This class handles object loading, and provides method interfaces that should be
    implemented by subclassed robots.
    """

    DEFAULT_RENDERING_PARAMS = {
        "use_pbr": True,
        "use_pbr_mapping": True,
        "shadow_caster": True,
    }

    def __init_subclass__(cls, **kwargs):
        """
        Registers all subclasses as part of this registry. This is useful to decouple internal codebase from external
        user additions. This way, users can add their custom robot by simply extending this Robot class,
        and it will automatically be registered internally. This allows users to then specify their robot
        directly in string-from in e.g., their config files, without having to manually set the str-to-class mapping
        in our code.
        """
        register_robot(cls)

    def __init__(
        self,
        name=None,
        control_freq=None,
        action_type="continuous",
        action_normalize=True,
        proprio_obs="default",
        reset_joint_pos=None,
        controller_config=None,
        base_name=None,
        scale=1.0,
        self_collision=False,
        class_id=SemanticClass.ROBOTS,
        rendering_params=None,
    ):
        """
        :param name: None or str, name of the robot object
        :param control_freq: float, control frequency (in Hz) at which to control the robot. If set to be None,
            simulator.import_robot will automatically set the control frequency to be 1 / render_timestep by default.
        :param action_type: str, one of {discrete, continuous} - what type of action space to use
        :param action_normalize: bool, whether to normalize inputted actions. This will override any default values
         specified by this class.
        :param proprio_obs: str or tuple of str, proprioception observation key(s) to use for generating proprioceptive
            observations. If str, should be exactly "default" -- this results in the default proprioception observations
            being used, as defined by self.default_proprio_obs. See self._get_proprioception_dict for valid key choices
        :param reset_joint_pos: None or Array[float], if specified, should be the joint positions that the robot should
            be set to during a reset. If None (default), self.default_joint_pos will be used instead.
        :param controller_config: None or Dict[str, ...], nested dictionary mapping controller name(s) to specific controller
            configurations for this robot. This will override any default values specified by this class.
        :param base_name: None or str, robot link name that will represent the entire robot's frame of reference. If not None,
            this should correspond to one of the link names found in this robot's corresponding URDF / MJCF file.
            None defaults to the base link name used in @model_file
        :param scale: int, scaling factor for model (default is 1)
        :param self_collision: bool, whether to enable self collision
        :param class_id: SemanticClass, semantic class this robot belongs to. Default is SemanticClass.ROBOTS.
        :param rendering_params: None or Dict[str, Any], If not None, should be keyword-mapped rendering options to set.
            See DEFAULT_RENDERING_PARAMS for the values passed by default.
        """
        # Store arguments
        if type(name) == dict:
            raise ValueError(
                "Robot name is a dict. You are probably using the deprecated constructor API which takes in robot_config (a dict) as input. Check the new API in BaseRobot."
            )
        if name is None:
            address = "%08X" % id(self)
            name = "{}_{}".format(self.category, address)

        self.name = name
        self.base_name = base_name
        self.control_freq = control_freq
        self.scale = scale
        self.self_collision = self_collision
        assert_valid_key(key=action_type, valid_keys={"discrete", "continuous"}, name="action type")
        self.action_type = action_type
        self.action_normalize = action_normalize
        self.proprio_obs = self.default_proprio_obs if proprio_obs == "default" else list(proprio_obs)
        self.reset_joint_pos = reset_joint_pos if reset_joint_pos is None else np.array(reset_joint_pos)
        self.controller_config = {} if controller_config is None else controller_config

        # Initialize internal attributes that will be loaded later
        # These will have public interfaces
        self.simulator = None
        self.model_type = None
        self.action_list = None  # Array of discrete actions to deploy
        self._links = None
        self._joints = None
        self._controllers = None
        self._mass = None
        self._joint_state = {  # This is filled in periodically every time self.update_state() is called
            "unnormalized": {
                "position": None,
                "velocity": None,
                "torque": None,
            },
            "normalized": {
                "position": None,
                "velocity": None,
                "torque": None,
            },
            "at_limits": None,
        }

        # TODO: Replace this with a reasonable StatefulObject inheritance.
        prepare_object_states(self, abilities={"robot": {}})

        # This logic is repeated because Robot does not inherit from Object.
        # TODO: Remove this logic once the object refactoring is complete.
        self.class_id = class_id
        self._rendering_params = dict(self.DEFAULT_RENDERING_PARAMS)
        if rendering_params is not None:
            self._rendering_params.update(rendering_params)

        self._loaded = False
        self._body_ids = None

    def load(self, simulator):
        """
        Load object into pybullet and return list of loaded body ids.

        :param simulator: Simulator, iGibson simulator reference

        :return Array[int]: List of unique pybullet IDs corresponding to this model. This will usually
            only be a single value
        """
        if self._loaded:
            raise ValueError("Cannot load a single model multiple times.")
        self._loaded = True

        # Store body ids and return them
        _body_ids = self._load(simulator)

        # Reset the robot and keep all joints still after loading
        self.reset()
        self.keep_still()

        # A persistent reference to simulator is needed for AG in ManipulationRobot
        self.simulator = simulator
        return _body_ids

    @property
    def loaded(self):
        return self._loaded

    def _load(self, simulator):
        """
        Loads this pybullet model into the simulation. Should return a list of unique body IDs corresponding
        to this model.

        :param simulator: Simulator, iGibson simulator reference

        :return Array[int]: List of unique pybullet IDs corresponding to this model. This will usually
            only be a single value
        """
        logging.info("Loading robot model file: {}".format(self.model_file))

        # Set flags for loading model
        flags = p.URDF_USE_MATERIAL_COLORS_FROM_MTL
        if self.self_collision:
            flags = flags | p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT

        # Run some sanity checks and load the model
        model_file_type = self.model_file.split(".")[-1]
        if model_file_type == "urdf":
            self.model_type = "URDF"
            body_ids = (p.loadURDF(self.model_file, globalScaling=self.scale, flags=flags),)
        else:
            self.model_type = "MJCF"
            assert self.scale == 1.0, (
                "robot scale must be 1.0 because pybullet does not support scaling " "for MJCF model (p.loadMJCF)"
            )
            body_ids = p.loadMJCF(self.model_file, flags=flags)

        # Store body ids
        self._body_ids = body_ids

        # Load into simulator and initialize states
        for body_id in self._body_ids:
            simulator.load_object_in_renderer(self, body_id, self.class_id, **self._rendering_params)

        for state in self.states.values():
            state.initialize(simulator)

        # Grab relevant references from the body IDs
        self._setup_references()

        # Disable collisions
        for names in self.disabled_collision_pairs:
            link_a = self._links[names[0]]
            link_b = self._links[names[1]]
            p.setCollisionFilterPair(link_a.body_id, link_b.body_id, link_a.link_id, link_b.link_id, 0)

        # Load controllers
        self._load_controllers()

        # Setup action space
        self._action_space = (
            self._create_discrete_action_space()
            if self.action_type == "discrete"
            else self._create_continuous_action_space()
        )

        # Validate this robot configuration
        self._validate_configuration()

        # Return the body IDs
        return body_ids

    def _setup_references(self):
        """
        Parse the set of robot @body_ids to get properties including joint information and mass
        """
        # Initialize link and joint dictionaries for this robot
        self._links, self._joints, self._mass = OrderedDict(), OrderedDict(), 0.0

        # Grab model base info
        body_ids = self.get_body_ids()
        assert (
            self.base_name is not None or len(body_ids) == 1
        ), "Base name can be inferred only for single-body robots."

        for body_id in body_ids:
            base_name = p.getBodyInfo(body_id)[0].decode("utf8")
            self._links[base_name] = RobotLink(base_name, -1, body_id)
            # if base_name is unspecified, use this link as robot_body (base_link).
            if self.base_name is None:
                self.base_name = base_name

            # Loop through all robot links and infer relevant link / joint / mass references
            for j in range(p.getNumJoints(body_id)):
                self._mass += p.getDynamicsInfo(body_id, j)[0]
                p.setJointMotorControl2(body_id, j, p.POSITION_CONTROL, positionGain=0.1, velocityGain=0.1, force=0)
                _, joint_name, joint_type, _, _, _, _, _, _, _, _, _, link_name, _, _, _, _ = p.getJointInfo(body_id, j)
                logging.debug("Robot joint: {}".format(p.getJointInfo(body_id, j)))
                joint_name = joint_name.decode("utf8")
                link_name = link_name.decode("utf8")
                self._links[link_name] = RobotLink(link_name, j, body_id)

                # We additionally create joint references if they are (not) of certain types
                if joint_name[:6] == "ignore":
                    # We don't save a reference to this joint, but we disable its motor
                    RobotJoint(joint_name, j, body_id).disable_motor()
                elif joint_name[:8] == "jointfix" or joint_type == p.JOINT_FIXED:
                    # Fixed joint, so we don't save a reference to this joint
                    pass
                else:
                    # Default case, we store a reference
                    self._joints[joint_name] = RobotJoint(joint_name, j, body_id)

        # Assert that the base link is link -1 of one of the robot's bodies.
        assert self._links[self.base_name].link_id == -1, "Robot base link should be link -1 of some body."

        # Populate the joint states
        self.update_state()

        # Update the configs
        for group in self.controller_order:
            group_controller_name = (
                self.controller_config[group]["name"]
                if group in self.controller_config and "name" in self.controller_config[group]
                else self._default_controllers[group]
            )
            self.controller_config[group] = merge_nested_dicts(
                base_dict=self._default_controller_config[group][group_controller_name],
                extra_dict=self.controller_config.get(group, {}),
            )

        # Update the reset joint pos
        if self.reset_joint_pos is None:
            self.reset_joint_pos = self.default_joint_pos

    def _validate_configuration(self):
        """
        Run any needed sanity checks to make sure this robot was created correctly.
        """
        pass

    def update_state(self):
        """
        Updates the internal proprioceptive state of this robot, and returns the raw values

        :return Tuple[Array[float], Array[float]]: The raw joint states, normalized joint states
            for this robot
        """
        # Grab raw values
        joint_states = np.array([j.get_state() for j in self._joints.values()]).astype(np.float32).flatten()
        joint_states_normalized = (
            np.array([j.get_relative_state() for j in self._joints.values()]).astype(np.float32).flatten()
        )

        # Get raw joint values and normalized versions
        self._joint_state["unnormalized"]["position"] = joint_states[0::3]
        self._joint_state["unnormalized"]["velocity"] = joint_states[1::3]
        self._joint_state["unnormalized"]["torque"] = joint_states[2::3]
        self._joint_state["normalized"]["position"] = joint_states_normalized[0::3]
        self._joint_state["normalized"]["velocity"] = joint_states_normalized[1::3]
        self._joint_state["normalized"]["torque"] = joint_states_normalized[2::3]

        # Infer whether joints are at their limits
        self._joint_state["at_limits"] = 1.0 * (np.abs(self.joint_positions_normalized) > 0.99)

        # Return the raw joint states
        return joint_states, joint_states_normalized

    def calc_state(self):
        """
        Calculate proprioceptive states for the robot. By default, this is:
            [pos, rpy, lin_vel, ang_vel, joint_states]

        :return Array[float]: Flat array of proprioceptive states (e.g.: [position, orientation, ...])
        """
        # Update states
        joint_states, _ = self.update_state()
        pos = self.get_position()
        rpy = self.get_rpy()

        # rotate linear and angular velocities to local frame
        lin_vel = rotate_vector_3d(self.base_link.get_linear_velocity(), *rpy)
        ang_vel = rotate_vector_3d(self.base_link.get_angular_velocity(), *rpy)

        state = np.concatenate([pos, rpy, lin_vel, ang_vel, joint_states])
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

    def get_body_ids(self):
        """
        Gets the body ID for this robot.

        If the object somehow has multiple bodies, this will be the default body that the default manipulation functions
        will manipulate.

        Should be implemented by all subclasses.

        :return None or int: Body ID representing this model in simulation if it exists, else None
        """
        return self._body_ids

    def reset(self):
        """
        Reset function for each specific robot. Can be overwritten by subclass

        By default, sets all joint states (pos, vel) to 0, and resets all controllers.
        """
        for joint, joint_pos in zip(self._joints.values(), self.reset_joint_pos):
            joint.reset_state(joint_pos, 0.0)

        for controller in self._controllers.values():
            controller.reset()

    def _load_controllers(self):
        """
        Loads controller(s) to map inputted actions into executable (pos, vel, and / or torque) signals on this robot.
        Stores created controllers as dictionary mapping controller names to specific controller
        instances used by this robot.
        """
        # Initialize controllers to create
        self._controllers = OrderedDict()
        # Loop over all controllers, in the order corresponding to @action dim
        for name in self.controller_order:
            assert_valid_key(key=name, valid_keys=self.controller_config, name="controller name")
            cfg = self.controller_config[name]
            # If we're using normalized action space, override the inputs for all controllers
            if self.action_normalize:
                cfg["command_input_limits"] = "default"  # default is normalized (-1, 1)
            # Create the controller
            self._controllers[name] = create_controller(**cfg)

    @abstractmethod
    def _create_discrete_action_space(self):
        """
        Create a discrete action space for this robot. Should be implemented by the subclass (if a subclass does not
        support this type of action space, it should raise an error).

        :return gym.space: Robot-specific discrete action space
        """
        raise NotImplementedError

    def _create_continuous_action_space(self):
        """
        Create a continuous action space for this robot. By default, this loops over all controllers and
        appends their respective input command limits to set the action space.
        Any custom behavior should be implemented by the subclass (e.g.: if a subclass does not
        support this type of action space, it should raise an error).

        :return gym.space.Box: Robot-specific continuous action space
        """
        # Action space is ordered according to the order in _default_controller_config control
        low, high = [], []
        for controller in self._controllers.values():
            limits = controller.command_input_limits
            low.append(np.array([-np.inf]) if limits is None else limits[0])
            high.append(np.array([np.inf]) if limits is None else limits[1])

        return gym.spaces.Box(
            shape=(self.action_dim,), low=np.concatenate(low), high=np.concatenate(high), dtype=np.float32
        )

    def apply_action(self, action):
        """
        Converts inputted actions into low-level control signals and deploys them on the robot

        :param action: Array[float], n-DOF length array of actions to convert and deploy on the robot
        """
        # Update state
        self.update_state()

        # If we're using discrete action space, we grab the specific action and use that to convert to control
        if self.action_type == "discrete":
            action = np.array(self.action_list[action])

        # Run convert actions to controls
        control, control_type = self._actions_to_control(action=action)

        # Deploy control signals
        self._deploy_control(control=control, control_type=control_type)

    def _actions_to_control(self, action):
        """
        Converts inputted @action into low level control signals to deploy directly on the robot.
        This returns two arrays: the converted low level control signals and an array corresponding
        to the specific ControlType for each signal.

        :param action: Array[float], n-DOF length array of actions to convert and deploy on the robot
        :return Tuple[Array[float], Array[ControlType]]: The (1) raw control signals to send to the robot's joints
            and (2) control types for each joint
        """
        # First, loop over all controllers, and calculate the computed control
        control = OrderedDict()
        idx = 0
        for name, controller in self._controllers.items():
            # Compose control_dict
            control_dict = self.get_control_dict()
            # Set command, then take a controller step
            controller.update_command(command=action[idx : idx + controller.command_dim])
            control[name] = {
                "value": controller.step(control_dict=control_dict),
                "type": controller.control_type,
            }
            # Update idx
            idx += controller.command_dim

        # Compose controls
        u_vec = np.zeros(self.n_joints)
        u_type_vec = np.array([ControlType.POSITION] * self.n_joints)
        for group, ctrl in control.items():
            idx = self._controllers[group].joint_idx
            u_vec[idx] = ctrl["value"]
            u_type_vec[idx] = ctrl["type"]

        # Return control
        return u_vec, u_type_vec

    def _deploy_control(self, control, control_type):
        """
        Deploys control signals @control with corresponding @control_type on this robot

        :param control: Array[float], raw control signals to send to the robot's joints
        :param control_type: Array[ControlType], control types for each joint
        """
        # Run sanity check
        joints = self._joints.values()
        assert len(control) == len(control_type) == len(joints), (
            "Control signals, control types, and number of joints should all be the same!"
            "Got {}, {}, and {} respectively.".format(len(control), len(control_type), len(joints))
        )

        # Loop through all control / types, and deploy the signal
        for joint, ctrl, ctrl_type in zip(joints, control, control_type):
            if ctrl_type == ControlType.TORQUE:
                joint.set_torque(ctrl)
            elif ctrl_type == ControlType.VELOCITY:
                joint.set_vel(ctrl)
            elif ctrl_type == ControlType.POSITION:
                joint.set_pos(ctrl)
            else:
                raise ValueError("Invalid control type specified: {}".format(ctrl_type))

    def get_proprioception(self):
        """
        :return Array[float]: numpy array of all robot-specific proprioceptive observations.
        """
        proprio_dict = self._get_proprioception_dict()
        return np.concatenate([proprio_dict[obs] for obs in self.proprio_obs])

    def get_position(self):
        """
        :return Array[float]: (x,y,z) global cartesian coordinates of this model's body (as taken at its body_id)
        """
        return self.get_position_orientation()[0]

    def get_orientation(self):
        """
        :return Array[float]: (x,y,z,w) global orientation in quaternion form of this model's body
            (as taken at its body_id)
        """
        return self.get_position_orientation()[1]

    def get_position_orientation(self):
        """
        :return Tuple[Array[float], Array[float]]: pos (x,y,z) global cartesian coordinates, quat (x,y,z,w) global
            orientation in quaternion form of this model's body (as taken at its body_id)
        """
        pos, orn = p.getBasePositionAndOrientation(self.base_link.body_id)
        return np.array(pos), np.array(orn)

    def get_rpy(self):
        """
        Return robot orientation in roll, pitch, yaw
        :return: roll, pitch, yaw
        """
        return self.base_link.get_rpy()

    def get_velocities(self):
        """Get object bodies' velocity in the format of List[Tuple[Array[vx, vy, vz], Array[wx, wy, wz]]]"""
        velocities = []
        for body_id in self.get_body_ids():
            lin, ang = p.getBaseVelocity(body_id)
            velocities.append((np.array(lin), np.array(ang)))

        return velocities

    def set_velocities(self, velocities):
        """Set object base velocity in the format of List[Tuple[Array[vx, vy, vz], Array[wx, wy, wz]]]"""
        assert len(velocities) == len(self.get_body_ids()), "Number of velocities should match number of bodies."

        for bid, (linear_velocity, angular_velocity) in zip(self.get_body_ids(), velocities):
            p.resetBaseVelocity(bid, linear_velocity, angular_velocity)

    def set_joint_positions(self, joint_positions):
        """Set this robot's joint positions, where @joint_positions is an array"""
        for joint, joint_pos in zip(self._joints.values(), joint_positions):
            joint.reset_state(pos=joint_pos, vel=0.0)

    def set_joint_states(self, joint_states):
        """Set object joint states in the format of Dict[String: (q, q_dot)]]"""
        for joint_name, joint in self._joints.items():
            joint.reset_state(*joint_states[joint_name])

    def get_joint_states(self):
        """Get object joint states in the format of Dict[String: (q, q_dot)]]"""
        joint_states = {}
        for joint_name, joint in self._joints.items():
            joint_states[joint_name] = joint.get_state()[:2]
        return joint_states

    def get_linear_velocity(self):
        """
        Get linear velocity of this robot (velocity associated with base link)

        :return Array[float]: linear (x,y,z) velocity of this robot
        """
        return self.base_link.get_linear_velocity()

    def get_angular_velocity(self):
        """
        Get angular velocity of this robot (velocity associated with base link)

        :return Array[float]: angular (ax,ay,az) velocity of this robot
        """
        return self.base_link.get_angular_velocity()

    def set_position(self, pos):
        """
        Sets the model's global position

        :param pos: Array[float], corresponding to (x,y,z) global cartesian coordinates to set
        """
        old_quat = self.get_orientation()
        self.set_position_orientation(pos, old_quat)

    def set_orientation(self, quat):
        """
        Set the model's global orientation

        :param quat: Array[float], corresponding to (x,y,z,w) global quaternion orientation to set
        """
        old_pos = self.get_position()
        self.set_position_orientation(old_pos, quat)

    def set_position_orientation(self, pos, quat):
        """
        Set model's global position and orientation

        :param pos: Array[float], corresponding to (x,y,z) global cartesian coordinates to set
        :param quat: Array[float], corresponding to (x,y,z,w) global quaternion orientation to set
        """
        p.resetBasePositionAndOrientation(self.base_link.body_id, pos, quat)
        clear_cached_states(self)

    def get_control_dict(self):
        """
        Grabs all relevant information that should be passed to each controller during each controller step.

        :return Dict[str, Array[float]]: Keyword-mapped control values for this robot.
            By default, returns the following:

            - joint_position: (n_dof,) joint positions
            - joint_velocity: (n_dof,) joint velocities
            - joint_torque: (n_dof,) joint torques
            - base_pos: (3,) (x,y,z) global cartesian position of the robot's base link
            - base_quat: (4,) (x,y,z,w) global cartesian orientation of ths robot's base link
        """
        return {
            "joint_position": self.joint_positions,
            "joint_velocity": self.joint_velocities,
            "joint_torque": self.joint_torques,
            "base_pos": self.get_position(),
            "base_quat": self.get_orientation(),
        }

    def dump_config(self):
        """Dump robot config"""
        return {
            "control_freq": self.control_freq,
            "action_type": self.action_type,
            "action_normalize": self.action_normalize,
            "proprio_obs": self.proprio_obs,
            "controller_config": self.controller_config,
            "base_name": self.base_name,
            "scale": self.scale,
            "self_collision": self.self_collision,
            "class_id": self.class_id,
            "rendering_params": self._rendering_params,
        }

    # TODO: Replace this with a reasonable StatefulObject inheritance.
    def dump_state(self):
        """Dump the state of the object other than what's not included in pybullet state."""
        return {
            "states": {
                get_state_name(state_type): state_instance.dump()
                for state_type, state_instance in self.states.items()
                if issubclass(state_type, AbsoluteObjectState)
            },
            "controllers": {
                controller_name: controller.dump_state() for controller_name, controller in self._controllers.items()
            },
        }

    def load_state(self, dump):
        """Dump the state of the object other than what's not included in pybullet state."""
        state_dump = dump["states"]
        for state_type, state_instance in self.states.items():
            if issubclass(state_type, AbsoluteObjectState):
                state_instance.load(state_dump[get_state_name(state_type)])

        controller_dump = dump["controllers"]
        for controller_name, controller in self._controllers.items():
            controller.load_state(controller_dump[controller_name])

    def _get_proprioception_dict(self):
        """
        :return dict: keyword-mapped proprioception observations available for this robot. Can be extended by subclasses
        """
        return {
            "joint_qpos": self.joint_positions,
            "joint_qpos_sin": np.sin(self.joint_positions),
            "joint_qpos_cos": np.cos(self.joint_positions),
            "joint_qvel": self.joint_velocities,
            "joint_qtor": self.joint_torques,
            "robot_pos": self.get_position(),
            "robot_rpy": self.get_rpy(),
            "robot_quat": self.get_orientation(),
            "robot_lin_vel": self.get_linear_velocity(),
            "robot_ang_vel": self.get_angular_velocity(),
        }

    @property
    def proprioception_dim(self):
        """
        :return int: Size of self.get_proprioception() vector
        """
        return len(self.get_proprioception())

    @property
    def links(self):
        """
        Links belonging to this robot.

        :return OrderedDict[str, RobotLink]: Ordered Dictionary mapping robot link names to corresponding
            RobotLink objects owned by this robot
        """
        return self._links

    @property
    def joints(self):
        """
        Joints belonging to this robot.

        :return OrderedDict[str, RobotJoint]: Ordered Dictionary mapping robot joint names to corresponding
            RobotJoint objects owned by this robot
        """
        return self._joints

    @property
    def n_links(self):
        """
        :return int: Number of links for this robot
        """
        return len(list(self._links.keys()))

    @property
    def n_joints(self):
        """
        :return int: Number of joints for this robot
        """
        return len(list(self._joints.keys()))

    @property
    def base_link(self):
        """
        Returns the RobotLink body corresponding to the link as defined by self.base_name.

        Note that if base_name was not specified during this robot's initialization, this will default to be the
        first link in the underlying robot model file.

        :return RobotLink: robot's base link corresponding to self.base_name.
        """
        assert self.base_name in self._links, "Cannot find base link '{}' in links! Valid options are: {}".format(
            self.base_name, list(self._links.keys())
        )
        return self._links[self.base_name]

    @property
    def eyes(self):
        """
        Returns the RobotLink corresponding to the robot's camera. Assumes that there is a link
        with name "eyes" in the underlying robot model. If not, an error will be raised.

        :return RobotLink: link containing the robot's camera
        """
        assert "eyes" in self._links, "Cannot find 'eyes' in links, current link names are: {}".format(
            list(self._links.keys())
        )
        return self._links["eyes"]

    @property
    def mass(self):
        """
        Returns the mass of this robot. Default is 0.0 kg

        :return float: Mass of this robot, in kg
        """
        return self._mass

    @property
    def joint_position_limits(self):
        """
        :return Tuple[Array[float], Array[float]]: (min, max) joint position limits, where each is an n-DOF length array
        """
        return (self.joint_lower_limits, self.joint_upper_limits)

    @property
    def joint_velocity_limits(self):
        """
        :return Tuple[Array[float], Array[float]]: (min, max) joint velocity limits, where each is an n-DOF length array
        """
        return (
            -np.array([j.max_velocity for j in self._joints.values()]),
            np.array([j.max_velocity for j in self._joints.values()]),
        )

    @property
    def joint_torque_limits(self):
        """
        :return Tuple[Array[float], Array[float]]: (min, max) joint torque limits, where each is an n-DOF length array
        """
        return (
            -np.array([j.max_torque for j in self._joints.values()]),
            np.array([j.max_torque for j in self._joints.values()]),
        )

    @property
    def joint_positions(self):
        """
        :return Array[float]: n-DOF length array of this robot's joint positions
        """
        return deepcopy(self._joint_state["unnormalized"]["position"])

    @property
    def joint_velocities(self):
        """
        :return Array[float]: n-DOF length array of this robot's joint velocities
        """
        return deepcopy(self._joint_state["unnormalized"]["velocity"])

    @property
    def joint_torques(self):
        """
        :return Array[float]: n-DOF length array of this robot's joint torques
        """
        return deepcopy(self._joint_state["unnormalized"]["torque"])

    @property
    def joint_positions_normalized(self):
        """
        :return Array[float]: n-DOF length array of this robot's normalized joint positions in range [-1, 1]
        """
        return deepcopy(self._joint_state["normalized"]["position"])

    @property
    def joint_velocities_normalized(self):
        """
        :return Array[float]: n-DOF length array of this robot's normalized joint velocities in range [-1, 1]
        """
        return deepcopy(self._joint_state["normalized"]["velocity"])

    @property
    def joint_torques_normalized(self):
        """
        :return Array[float]: n-DOF length array of this robot's normalized joint torques in range [-1, 1]
        """
        return deepcopy(self._joint_state["normalized"]["torque"])

    @property
    def joint_at_limits(self):
        """
        :return Array[float]: n-DOF length array specifying whether joint is at its limit,
            with 1.0 --> at limit, otherwise 0.0
        """
        return deepcopy(self._joint_state["at_limits"])

    @property
    def joint_has_limits(self):
        """
        :return Array[bool]: n-DOF length array specifying whether joint has a limit or not
        """
        return np.array([j.has_limit for j in self._joints.values()])

    @property
    def category(self):
        """
        :return str: Semantic category for robots
        """
        return "agent"

    @property
    @abstractmethod
    def model_name(self):
        """
        :return str: robot model name
        """
        raise NotImplementedError

    @property
    def action_dim(self):
        """
        :return int: Dimension of action space for this robot. By default,
            is the sum over all controller action dimensions
        """
        return sum([controller.command_dim for controller in self._controllers.values()])

    @property
    def action_space(self):
        """
        Action space for this robot.

        :return gym.space: Action space, either discrete (Discrete) or continuous (Box)
        """
        return deepcopy(self._action_space)

    @property
    @abstractmethod
    def controller_order(self):
        """
        :return Tuple[str]: Ordering of the actions, corresponding to the controllers. e.g., ["base", "arm", "gripper"],
            to denote that the action vector should be interpreted as first the base action, then arm command, then
            gripper command
        """
        raise NotImplementedError

    @property
    def controller_action_idx(self):
        """
        :return: Dict[str, Array[int]]: Mapping from controller names (e.g.: head, base, arm, etc.) to corresponding
            indices in the action vector
        """
        dic = {}
        idx = 0
        for controller in self.controller_order:
            cmd_dim = self._controllers[controller].command_dim
            dic[controller] = np.arange(idx, idx + cmd_dim)
            idx += cmd_dim

        return dic

    @property
    def control_limits(self):
        """
        :return: Dict[str, Any]: Keyword-mapped limits for this robot. Dict contains:
            position: (min, max) joint limits, where min and max are N-DOF arrays
            velocity: (min, max) joint velocity limits, where min and max are N-DOF arrays
            torque: (min, max) joint torque limits, where min and max are N-DOF arrays
            has_limit: (n_dof,) array where each element is True if that corresponding joint has a position limit
                (otherwise, joint is assumed to be limitless)
        """
        return {
            "position": (self.joint_lower_limits, self.joint_upper_limits),
            "velocity": (-self.max_joint_velocities, self.max_joint_velocities),
            "torque": (-self.max_joint_torques, self.max_joint_torques),
            "has_limit": self.joint_has_limits,
        }

    @property
    def default_proprio_obs(self):
        """
        :return Array[str]: Default proprioception observations to use
        """
        return []

    @property
    @abstractmethod
    def default_joint_pos(self):
        """
        :return Array[float]: Default joint positions for this robot
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def _default_controller_config(self):
        """
        :return Dict[str, Any]: default nested dictionary mapping controller name(s) to specific controller
            configurations for this robot. Note that the order specifies the sequence of actions to be received
            from the environment.

            Expected structure is as follows:
                group1:
                    controller_name1:
                        controller_name1_params
                        ...
                    controller_name2:
                        ...
                group2:
                    ...

            The @group keys specify the control type for various aspects of the robot, e.g.: "head", "arm", "base", etc.
            @controller_name keys specify the supported controllers for that group. A default specification MUST be
            specified for each controller_name. e.g.: IKController, DifferentialDriveController, JointController, etc.
        """
        return {}

    @property
    @abstractmethod
    def _default_controllers(self):
        """
        :return Dict[str, str]: Maps robot group (e.g. base, arm, etc.) to default controller class name to use
            (e.g. IKController, JointController, etc.)
        """
        return {}

    @property
    def joint_ids(self):
        """
        :return: Array[int], joint IDs for this robot
        """
        return np.array([joint.joint_id for joint in self._joints.values()])

    @property
    def joint_damping(self):
        """
        :return: Array[float], joint damping values for this robot
        """
        return np.array([joint.get_info().jointDamping for joint in self._joints.values()])

    @property
    def joint_lower_limits(self):
        """
        :return: Array[float], minimum values for this robot's joints. If joint does not have a range, returns -1000
            for that joint
        """
        return np.array([joint.lower_limit if joint.has_limit else -1000.0 for joint in self._joints.values()])

    @property
    def joint_upper_limits(self):
        """
        :return: Array[float], maximum values for this robot's joints. If joint does not have a range, returns 1000
            for that joint
        """
        return np.array([joint.upper_limit if joint.has_limit else 1000.0 for joint in self._joints.values()])

    @property
    def joint_range(self):
        """
        :return: Array[float], joint range values for this robot's joints
        """
        return self.joint_upper_limits - self.joint_lower_limits

    @property
    def max_joint_velocities(self):
        """
        :return: Array[float], maximum velocities for this robot's joints
        """
        return np.array([joint.max_velocity for joint in self._joints.values()])

    @property
    def max_joint_torques(self):
        """
        :return: Array[float], maximum torques for this robot's joints
        """
        return np.array([joint.max_torque for joint in self._joints.values()])

    @property
    def disabled_collision_pairs(self):
        """
        :return Tuple[Tuple[str, str]]: List of collision pairs to disable. Default is None (empty list)
        """
        return []

    @property
    @abstractmethod
    def model_file(self):
        """
        :return str: absolute path to robot model's URDF / MJCF file
        """
        raise NotImplementedError

    def force_wakeup(self):
        for link in self._links.values():
            link.force_wakeup()

    def keep_still(self):
        """
        Keep the robot still. Apply zero velocity to all joints.
        """
        for joint in self._joints.values():
            joint.set_vel(0.0)


class RobotLink:
    """
    Body part (link) of Robots
    """

    def __init__(self, link_name, link_id, body_id):
        """
        :param link_name: str, name of the link corresponding to @link_id
        :param link_id: int, ID of this link within the link(s) found in the body corresponding to @body_id
        :param body_id: Robot body ID containing this link
        """
        # Store args and initialize state
        self.link_name = link_name
        self.link_id = link_id
        self.body_id = body_id
        self.initial_pos, self.initial_quat = self.get_position_orientation()
        self.movement_cid = -1

    def get_name(self):
        """
        Get name of this link
        """
        return self.link_name

    def get_position_orientation(self):
        """
        Get pose of this link

        :return Tuple[Array[float], Array[float]]: pos (x,y,z) cartesian coordinates, quat (x,y,z,w)
            orientation in quaternion form of this link
        """
        if self.link_id == -1:
            pos, quat = p.getBasePositionAndOrientation(self.body_id)
        else:
            _, _, _, _, pos, quat = p.getLinkState(self.body_id, self.link_id)
        return np.array(pos), np.array(quat)

    def get_position(self):
        """
        :return Array[float]: (x,y,z) cartesian coordinates of this link
        """
        return self.get_position_orientation()[0]

    def get_orientation(self):
        """
        :return Array[float]: (x,y,z,w) orientation in quaternion form of this link
        """
        return self.get_position_orientation()[1]

    def get_rpy(self):
        """
        :return Array[float]: (r,p,y) orientation in euler form of this link
        """
        return np.array(p.getEulerFromQuaternion(self.get_orientation()))

    def set_position(self, pos):
        """
        Sets the link's position

        :param pos: Array[float], corresponding to (x,y,z) cartesian coordinates to set
        """
        old_quat = self.get_orientation()
        self.set_position_orientation(pos, old_quat)

    def set_orientation(self, quat):
        """
        Set the link's global orientation

        :param quat: Array[float], corresponding to (x,y,z,w) quaternion orientation to set
        """
        old_pos = self.get_position()
        self.set_position_orientation(old_pos, quat)

    def set_position_orientation(self, pos, quat):
        """
        Set model's global position and orientation. Note: only supported if this is the base link (ID = -1!)

        :param pos: Array[float], corresponding to (x,y,z) global cartesian coordinates to set
        :param quat: Array[float], corresponding to (x,y,z,w) global quaternion orientation to set
        """
        assert self.link_id == -1, "Can only set pose for a base link (id = -1)! Got link id: {}.".format(self.link_id)
        p.resetBasePositionAndOrientation(self.body_id, pos, quat)

    def get_velocity(self):
        """
        Get velocity of this link

        :return Tuple[Array[float], Array[float]]: linear (x,y,z) velocity, angular (ax,ay,az)
            velocity of this link
        """
        if self.link_id == -1:
            lin, ang = p.getBaseVelocity(self.body_id)
        else:
            _, _, _, _, _, _, lin, ang = p.getLinkState(self.body_id, self.link_id, computeLinkVelocity=1)
        return np.array(lin), np.array(ang)

    def get_linear_velocity(self):
        """
        Get linear velocity of this link

        :return Array[float]: linear (x,y,z) velocity of this link
        """
        return self.get_velocity()[0]

    def get_angular_velocity(self):
        """
        Get angular velocity of this link

        :return Array[float]: angular (ax,ay,az) velocity of this link
        """
        return self.get_velocity()[1]

    def contact_list(self):
        """
        Get contact points of the body part

        :return Array[ContactPoints]: list of contact points seen by this link
        """
        return p.getContactPoints(self.body_id, -1, self.link_id, -1)

    def force_wakeup(self):
        """
        Forces a wakeup for this robot. Defaults to no-op.
        """
        p.changeDynamics(self.body_id, self.link_id, activationState=p.ACTIVATION_STATE_WAKE_UP)


class RobotJoint:
    """
    Joint of a robot
    """

    def __init__(self, joint_name, joint_id, body_id):
        """
        :param joint_name: str, name of the joint corresponding to @joint_id
        :param joint_id: int, ID of this joint within the joint(s) found in the body corresponding to @body_id
        :param body_id: Robot body ID containing this link
        """
        # Store args and initialize state
        self.joint_name = joint_name
        self.joint_id = joint_id
        self.body_id = body_id

        # read joint type and joint limit from the URDF file
        # lower_limit, upper_limit, max_velocity, max_torque = <limit lower=... upper=... velocity=... effort=.../>
        # "effort" is approximately torque (revolute) / force (prismatic), but not exactly (ref: http://wiki.ros.org/pr2_controller_manager/safety_limits).
        # if <limit /> does not exist, the following will be the default value
        # lower_limit, upper_limit, max_velocity, max_torque = 0.0, -1.0, 0.0, 0.0
        (
            _,
            _,
            self.joint_type,
            _,
            _,
            _,
            _,
            _,
            self.lower_limit,
            self.upper_limit,
            self.max_torque,
            self.max_velocity,
            _,
            _,
            _,
            _,
            _,
        ) = p.getJointInfo(self.body_id, self.joint_id)

        # if joint torque and velocity limits cannot be found in the model file, set a default value for them
        if self.max_torque == 0.0:
            self.max_torque = 100.0
        if self.max_velocity == 0.0:
            # if max_velocity and joint limit are missing for a revolute joint,
            # it's likely to be a wheel joint and a high max_velocity is usually supported.
            self.max_velocity = 15.0 if self.joint_type == p.JOINT_REVOLUTE and not self.has_limit else 1.0

    def __str__(self):
        return "idx: {}, name: {}".format(self.joint_id, self.joint_name)

    def get_state(self):
        """
        Get the current state of the joint

        :return Tuple[float, float, float]: (joint_pos, joint_vel, joint_tor) observed for this joint
        """
        x, vx, _, trq = p.getJointState(self.body_id, self.joint_id)
        return x, vx, trq

    def get_relative_state(self):
        """
        Get the normalized current state of the joint

        :return Tuple[float, float, float]: Normalized (joint_pos, joint_vel, joint_tor) observed for this joint
        """
        pos, vel, trq = self.get_state()

        # normalize position to [-1, 1]
        if self.has_limit:
            mean = (self.lower_limit + self.upper_limit) / 2.0
            magnitude = (self.upper_limit - self.lower_limit) / 2.0
            pos = (pos - mean) / magnitude

        # (trying to) normalize velocity to [-1, 1]
        vel /= self.max_velocity

        # (trying to) normalize torque / force to [-1, 1]
        trq /= self.max_torque

        return pos, vel, trq

    def get_info(self):
        return get_joint_info(self.body_id, self.joint_id)

    def set_pos(self, pos):
        """
        Set position of joint (in metric space)

        :param pos: float, desired position for this joint, in metric space
        """
        if self.has_limit:
            pos = np.clip(pos, self.lower_limit, self.upper_limit)
        p.setJointMotorControl2(self.body_id, self.joint_id, p.POSITION_CONTROL, targetPosition=pos)

    def set_vel(self, vel):
        """
        Set velocity of joint (in metric space)

        :param vel: float, desired velocity for this joint, in metric space
        """
        vel = np.clip(vel, -self.max_velocity, self.max_velocity)
        p.setJointMotorControl2(self.body_id, self.joint_id, p.VELOCITY_CONTROL, targetVelocity=vel)

    def set_torque(self, torque):
        """
        Set torque of joint (in metric space)

        :param torque: float, desired torque for this joint, in metric space
        """
        torque = np.clip(torque, -self.max_torque, self.max_torque)
        p.setJointMotorControl2(
            bodyIndex=self.body_id,
            jointIndex=self.joint_id,
            controlMode=p.TORQUE_CONTROL,
            force=torque,
        )

    def reset_state(self, pos, vel):
        """
        Reset pos and vel of joint in metric space

        :param pos: float, desired position for this joint, in metric space
        :param vel: float, desired velocity for this joint, in metric space
        """
        p.resetJointState(self.body_id, self.joint_id, targetValue=pos, targetVelocity=vel)
        self.disable_motor()

    def disable_motor(self):
        """
        Disable the motor of this joint
        """
        p.setJointMotorControl2(
            self.body_id,
            self.joint_id,
            controlMode=p.POSITION_CONTROL,
            targetPosition=0,
            targetVelocity=0,
            positionGain=0.1,
            velocityGain=0.1,
            force=0,
        )

    @property
    def has_limit(self):
        """
        :return bool: True if this joint has a limit, else False
        """
        return self.lower_limit < self.upper_limit
