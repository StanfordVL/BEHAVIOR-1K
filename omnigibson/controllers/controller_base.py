from collections.abc import Iterable
from enum import IntEnum

import numpy as np

from omnigibson.utils.python_utils import classproperty, assert_valid_key, Serializable, Registerable, Recreatable

# Global dicts that will contain mappings
REGISTERED_CONTROLLERS = dict()
REGISTERED_LOCOMOTION_CONTROLLERS = dict()
REGISTERED_MANIPULATION_CONTROLLERS = dict()
REGISTERED_GRIPPER_CONTROLLERS = dict()


def register_locomotion_controller(cls):
    if cls.__name__ not in REGISTERED_LOCOMOTION_CONTROLLERS:
        REGISTERED_LOCOMOTION_CONTROLLERS[cls.__name__] = cls


def register_manipulation_controller(cls):
    if cls.__name__ not in REGISTERED_MANIPULATION_CONTROLLERS:
        REGISTERED_MANIPULATION_CONTROLLERS[cls.__name__] = cls


def register_gripper_controller(cls):
    if cls.__name__ not in REGISTERED_GRIPPER_CONTROLLERS:
        REGISTERED_GRIPPER_CONTROLLERS[cls.__name__] = cls


class IsGraspingState(IntEnum):
    TRUE = 1
    UNKNOWN = 0
    FALSE = -1


# Define macros
class ControlType:
    NONE = -1
    POSITION = 0
    VELOCITY = 1
    EFFORT = 2
    _MAPPING = {
        "none": NONE,
        "position": POSITION,
        "velocity": VELOCITY,
        "effort": EFFORT,
    }
    VALID_TYPES = set(_MAPPING.values())
    VALID_TYPES_STR = set(_MAPPING.keys())

    @classmethod
    def get_type(cls, type_str):
        """
        Args:
            type_str (str): One of "position", "velocity", or "effort" (any case), and maps it
                to the corresponding type

        Returns:
            ControlType: control type corresponding to the associated string
        """
        assert_valid_key(key=type_str.lower(), valid_keys=cls._MAPPING, name="control type")
        return cls._MAPPING[type_str.lower()]


class BaseController(Serializable, Registerable, Recreatable):
    """
    An abstract class with interface for mapping specific types of commands to deployable control signals.
    """

    def __init__(
        self,
        control_freq,
        control_limits,
        dof_idx,
        command_input_limits="default",
        command_output_limits="default",
    ):
        """
        Args:
            control_freq (int): controller loop frequency
            control_limits (Dict[str, Tuple[Array[float], Array[float]]]): The min/max limits to the outputted
                control signal. Should specify per-dof type limits, i.e.:

                "position": [[min], [max]]
                "velocity": [[min], [max]]
                "effort": [[min], [max]]
                "has_limit": [...bool...]

                Values outside of this range will be clipped, if the corresponding joint index in has_limit is True.
            dof_idx (Array[int]): specific dof indices controlled by this robot. Used for inferring
                controller-relevant values during control computations
            command_input_limits (None or "default" or Tuple[float, float] or Tuple[Array[float], Array[float]]):
                if set, is the min/max acceptable inputted command. Values outside this range will be clipped.
                If None, no clipping will be used. If "default", range will be set to (-1, 1)
            command_output_limits (None or "default" or Tuple[float, float] or Tuple[Array[float], Array[float]]):
                if set, is the min/max scaled command. If both this value and @command_input_limits is not None,
                then all inputted command values will be scaled from the input range to the output range.
                If either is None, no scaling will be used. If "default", then this range will automatically be set
                to the @control_limits entry corresponding to self.control_type
        """
        # Store arguments
        self._control_freq = control_freq
        self._control_limits = {}
        for motor_type in {"position", "velocity", "effort"}:
            if motor_type not in control_limits:
                continue

            self._control_limits[ControlType.get_type(motor_type)] = [
                np.array(control_limits[motor_type][0]),
                np.array(control_limits[motor_type][1]),
            ]
        assert "has_limit" in control_limits, "Expected has_limit specified in control_limits, but does not exist."
        self._dof_has_limits = control_limits["has_limit"]
        self._dof_idx = np.array(dof_idx, dtype=int)

        # Initialize some other variables that will be filled in during runtime
        self._control = None
        self._command = None
        self._command_scale_factor = None
        self._command_output_transform = None
        self._command_input_transform = None

        # Standardize command input / output limits to be (min_array, max_array)
        command_input_limits = (-1.0, 1.0) if command_input_limits == "default" else command_input_limits
        command_output_limits = (
            (
                np.array(self._control_limits[self.control_type][0])[self.dof_idx],
                np.array(self._control_limits[self.control_type][1])[self.dof_idx],
            )
            if command_output_limits == "default"
            else command_output_limits
        )
        self._command_input_limits = (
            None
            if command_input_limits is None
            else (
                self.nums2array(command_input_limits[0], self.command_dim),
                self.nums2array(command_input_limits[1], self.command_dim),
            )
        )
        self._command_output_limits = (
            None
            if command_output_limits is None
            else (
                self.nums2array(command_output_limits[0], self.command_dim),
                self.nums2array(command_output_limits[1], self.command_dim),
            )
        )

    def _preprocess_command(self, command):
        """
        Clips + scales inputted @command according to self.command_input_limits and self.command_output_limits.
        If self.command_input_limits is None, then no clipping will occur. If either self.command_input_limits
        or self.command_output_limits is None, then no scaling will occur.

        Args:
            command (Array[float] or float): Inputted command vector

        Returns:
            Array[float]: Processed command vector
        """
        # Make sure command is a np.array
        command = np.array([command]) if type(command) in {int, float} else np.array(command)
        # We only clip and / or scale if self.command_input_limits exists
        if self._command_input_limits is not None:
            # Clip
            command = command.clip(*self._command_input_limits)
            if self._command_output_limits is not None:
                # If we haven't calculated how to scale the command, do that now (once)
                if self._command_scale_factor is None:
                    self._command_scale_factor = abs(
                        self._command_output_limits[1] - self._command_output_limits[0]
                    ) / abs(self._command_input_limits[1] - self._command_input_limits[0])
                    self._command_output_transform = (
                        self._command_output_limits[1] + self._command_output_limits[0]
                    ) / 2.0
                    self._command_input_transform = (self._command_input_limits[1] + self._command_input_limits[0]) / 2.0
                # Scale command
                command = (
                    command - self._command_input_transform
                ) * self._command_scale_factor + self._command_output_transform

        # Return processed command
        return command

    def update_command(self, command):
        """
        Updates inputted @command internally.

        Args:
            command (Array[float]): inputted command to store internally in this controller
        """
        # Sanity check the command
        assert len(command) == self.command_dim, "Commands must be dimension {}, got dim {} instead.".format(
            self.command_dim, len(command)
        )
        # Preprocess and store inputted command
        self._command = self._preprocess_command(np.array(command))

    def clip_control(self, control):
        """
        Clips the inputted @control signal based on @control_limits.

        Args:
            control (Array[float]): control signal to clip

        Returns:
            Array[float]: Clipped control signal
        """
        clipped_control = control.clip(
            self._control_limits[self.control_type][0][self.dof_idx],
            self._control_limits[self.control_type][1][self.dof_idx],
        )
        idx = (
            self._dof_has_limits[self.dof_idx]
            if self.control_type == ControlType.POSITION
            else [True] * self.control_dim
        )
        control[idx] = clipped_control[idx]
        return control

    def step(self, control_dict):
        """
        Take a controller step.

        Args:
            control_dict (Dict[str, Any]): dictionary that should include any relevant keyword-mapped
                states necessary for controller computation

        Returns:
            Array[float]: numpy array of outputted control signals
        """
        control = self._command_to_control(command=self._command, control_dict=control_dict)
        self._control = self.clip_control(control=control)
        return self._control

    def reset(self):
        """
        Resets this controller. Should be implemented by subclass.
        """
        raise NotImplementedError

    def _dump_state(self):
        # Default is no state (empty dict)
        return dict()

    def _load_state(self, state):
        # Default is no state (empty dict), so this is a no-op
        pass

    def _serialize(self, state):
        # Default is no state, so do nothing
        return np.array([])

    def _deserialize(self, state):
        # Default is no state, so do nothing
        return dict(), 0

    def _command_to_control(self, command, control_dict):
        """
        Converts the (already preprocessed) inputted @command into deployable (non-clipped!) control signal.
        Should be implemented by subclass.

        Args:
            command (Array[float]): desired (already preprocessed) command to convert into control signals
            control_dict (Dict[str, Any]): dictionary that should include any relevant keyword-mapped
                states necessary for controller computation

        Returns:
            Array[float]: outputted (non-clipped!) control signal to deploy
        """
        raise NotImplementedError

    @staticmethod
    def nums2array(nums, dim):
        """
        Convert input @nums into numpy array of length @dim. If @nums is a single number, broadcasts it to the
        corresponding dimension size @dim before converting into a numpy array

        Args:
            nums (numeric or Iterable): Either single value or array of numbers
            dim (int): Size of array to broadcast input to

        Returns:
            np.array: Array filled with values specified in @nums
        """
        # First run sanity check to make sure no strings are being inputted
        if isinstance(nums, str):
            raise TypeError("Error: Only numeric inputs are supported for this function, nums2array!")

        # Check if input is an Iterable, if so, we simply convert the input to np.array and return
        # Else, input is a single value, so we map to a numpy array of correct size and return
        return np.array(nums) if isinstance(nums, Iterable) else np.ones(dim) * nums

    @property
    def state_size(self):
        # Default is no state, so return 0
        return 0

    @property
    def control(self):
        """
        Returns:
            n-array: Array of most recent controls deployed by this controller
        """
        return self._control

    @property
    def control_freq(self):
        """
        Returns:
            float: Control frequency (Hz) of this controller
        """
        return self._control_freq

    @property
    def control_dim(self):
        """
        Returns:
            int: Expected size of outputted controls
        """
        return len(self.dof_idx)

    @property
    def control_type(self):
        """
        Returns:
            ControlType: Type of control returned by this controller
        """
        raise NotImplementedError

    @property
    def command_input_limits(self):
        """
        Returns:
            None or 2-tuple: If specified, returns (min, max) command input limits for this controller, where
                @min and @max are numpy float arrays of length self.command_dim. Otherwise, returns None
        """
        return self._command_input_limits

    @property
    def command_output_limits(self):
        """
        Returns:
            None or 2-tuple: If specified, returns (min, max) command output limits for this controller, where
                @min and @max are numpy float arrays of length self.command_dim. Otherwise, returns None
        """
        return self._command_output_limits

    @property
    def command_dim(self):
        """
        Returns:
            int: Expected size of inputted commands
        """
        raise NotImplementedError

    @property
    def dof_idx(self):
        """
        Returns:
            Array[int]: DOF indices corresponding to the specific DOFs being controlled by this robot
        """
        return np.array(self._dof_idx)

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("BaseController")
        return classes

    @classproperty
    def _cls_registry(cls):
        # Global registry
        global REGISTERED_CONTROLLERS
        return REGISTERED_CONTROLLERS


class LocomotionController(BaseController):
    """
    Controller to control locomotion. All implemented controllers that encompass locomotion capabilities should extend
    from this class.
    """

    def __init_subclass__(cls, **kwargs):
        # Register as part of locomotion controllers
        super().__init_subclass__(**kwargs)
        register_locomotion_controller(cls)

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("LocomotionController")
        return classes


class ManipulationController(BaseController):
    """
    Controller to control manipulation. All implemented controllers that encompass manipulation capabilities
    should extend from this class.
    """

    def __init_subclass__(cls, **kwargs):
        # Register as part of manipulation controllers
        super().__init_subclass__(**kwargs)
        register_manipulation_controller(cls)

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("ManipulationController")
        return classes


class GripperController(BaseController):
    """
    Controller to control a gripper. All implemented controllers that encompass gripper capabilities
    should extend from this class.
    """

    def __init_subclass__(cls, **kwargs):
        # Register as part of gripper controllers
        super().__init_subclass__(**kwargs)
        register_gripper_controller(cls)

    def is_grasping(self):
        """
        Checks whether the current state of this gripper being controlled is in a grasping state.
        Should be implemented by subclass.

        Returns:
            IsGraspingState: Grasping state of gripper
        """
        raise NotImplementedError()

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("GripperController")
        return classes
