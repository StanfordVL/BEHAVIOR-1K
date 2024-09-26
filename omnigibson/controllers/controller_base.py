import math
from collections.abc import Iterable
from enum import IntEnum

import torch as th

from omnigibson.utils.python_utils import Recreatable, Registerable, Serializable, assert_valid_key, classproperty

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
                control_limits[motor_type][0],
                control_limits[motor_type][1],
            ]
        assert "has_limit" in control_limits, "Expected has_limit specified in control_limits, but does not exist."
        self._dof_has_limits = control_limits["has_limit"]
        self._dof_idx = dof_idx.int()

        # Generate goal information
        self._goal_shapes = self._get_goal_shapes()
        self._goal_dim = int(sum(th.prod(th.tensor(shape)) for shape in self._goal_shapes.values()))

        # Initialize some other variables that will be filled in during runtime
        self._control = None
        self._goal = None
        self._command_scale_factor = None
        self._command_output_transform = None
        self._command_input_transform = None

        # Standardize command input / output limits to be (min_array, max_array)
        command_input_limits = (
            (-1.0, 1.0)
            if type(command_input_limits) == str and command_input_limits == "default"
            else command_input_limits
        )
        command_output_limits = (
            (
                self._control_limits[self.control_type][0][self.dof_idx],
                self._control_limits[self.control_type][1][self.dof_idx],
            )
            if type(command_output_limits) == str and command_output_limits == "default"
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
        # Make sure command is a th.tensor
        command = th.tensor([command], dtype=th.float32) if type(command) in {int, float} else command
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
                    self._command_input_transform = (
                        self._command_input_limits[1] + self._command_input_limits[0]
                    ) / 2.0
                # Scale command
                command = (
                    command - self._command_input_transform
                ) * self._command_scale_factor + self._command_output_transform

        # Return processed command
        return command

    def _reverse_preprocess_command(self, processed_command):
        """
        Reverses the scaling operation performed by _preprocess_command.
        Note: This method does not reverse the clipping operation as it's not reversible.

        Args:
            processed_command (th.Tensor[float]): Processed command vector

        Returns:
            th.Tensor[float]: Original command vector (before scaling, clipping not reversed)
        """
        # We only reverse the scaling if both input and output limits exist
        if self._command_input_limits is not None and self._command_output_limits is not None:
            # If we haven't calculated how to scale the command, do that now (once)
            if self._command_scale_factor is None:
                self._command_scale_factor = abs(self._command_output_limits[1] - self._command_output_limits[0]) / abs(
                    self._command_input_limits[1] - self._command_input_limits[0]
                )
                self._command_output_transform = (self._command_output_limits[1] + self._command_output_limits[0]) / 2.0
                self._command_input_transform = (self._command_input_limits[1] + self._command_input_limits[0]) / 2.0

            original_command = (
                processed_command - self._command_output_transform
            ) / self._command_scale_factor + self._command_input_transform
        else:
            original_command = processed_command

        return original_command

    def update_goal(self, command, control_dict):
        """
        Updates inputted @command internally, writing any necessary internal variables as needed.

        Args:
            command (Array[float]): inputted command to preprocess and extract relevant goal(s) to store
                internally in this controller
            control_dict (dict): Current state
        """
        # Sanity check the command
        assert (
            len(command) == self.command_dim
        ), f"Commands must be dimension {self.command_dim}, got dim {len(command)} instead."

        # Preprocess and run internal command
        self._goal = self._update_goal(command=self._preprocess_command(command), control_dict=control_dict)

    def _update_goal(self, command, control_dict):
        """
        Updates inputted @command internally, writing any necessary internal variables as needed.

        Args:
            command (Array[float]): inputted (preprocessed!) command and extract relevant goal(s) to store
                internally in this controller
            control_dict (dict): Current state

        Returns:
            dict: Keyword-mapped goals to store internally in this controller
        """
        raise NotImplementedError

    def compute_control(self, goal_dict, control_dict):
        """
        Converts the (already preprocessed) inputted @command into deployable (non-clipped!) control signal.
        Should be implemented by subclass.

        Args:
            goal_dict (Dict[str, Any]): dictionary that should include any relevant keyword-mapped
                goals necessary for controller computation
            control_dict (Dict[str, Any]): dictionary that should include any relevant keyword-mapped
                states necessary for controller computation

        Returns:
            Array[float]: outputted (non-clipped!) control signal to deploy
        """
        raise NotImplementedError

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
        # Generate no-op goal if not specified
        if self._goal is None:
            self._goal = self.compute_no_op_goal(control_dict=control_dict)

        # Compute control, then clip and return
        control = self.compute_control(goal_dict=self._goal, control_dict=control_dict)
        assert (
            len(control) == self.control_dim
        ), f"Control signal must be of length {self.control_dim}, got {len(control)} instead."
        self._control = self.clip_control(control=control)
        return self._control

    def reset(self):
        """
        Resets this controller. Can be extended by subclass
        """
        self._goal = None

    def compute_no_op_goal(self, control_dict):
        """
        Compute no-op goal given the current state @control_dict

        Args:
            control_dict (dict): Current state

        Returns:
            dict: Maps relevant goal keys (from self._goal_shapes.keys()) to relevant goal data to be used
                in controller computations
        """
        raise NotImplementedError

    def compute_no_op_action(self, control_dict):
        """
        Compute a no-op action that updates the goal to match the current position
        Disclaimer: this no-op might cause drift under external load (e.g. when the controller cannot reach the goal position)
        """
        if self._goal is None:
            self._goal = self.compute_no_op_goal(control_dict=control_dict)
        command = self._compute_no_op_action(control_dict=control_dict)
        return self._reverse_preprocess_command(command)

    def _compute_no_op_action(self, control_dict):
        """
        Compute no-op action given the goal
        """
        raise NotImplementedError

    def _dump_state(self):
        # Default is just the command
        return dict(
            goal_is_valid=self._goal is not None,
            goal=self._goal,
        )

    def _load_state(self, state):
        # Make sure every entry in goal is a numpy array
        # Load goal
        self._goal = None if state["goal"] is None else {name: goal_state for name, goal_state in state["goal"].items()}

    def serialize(self, state):
        # Make sure size of the state is consistent, even if we have no goal
        goal_state_flattened = (
            th.cat([goal_state.flatten() for goal_state in self._goal.values()])
            if (state)["goal_is_valid"]
            else th.zeros(self.goal_dim)
        )

        return th.cat([th.tensor([state["goal_is_valid"]]), goal_state_flattened])

    def deserialize(self, state):
        goal_is_valid = bool(state[0])
        if goal_is_valid:
            # Un-flatten all the keys
            idx = 1
            goal = dict()
            for key, shape in self._goal_shapes.items():
                length = math.prod(shape)
                goal[key] = state[idx : idx + length].reshape(shape)
                idx += length
        else:
            goal = None
        state_dict = dict(
            goal_is_valid=goal_is_valid,
            goal=goal,
        )
        return state_dict, self.goal_dim + 1

    def _get_goal_shapes(self):
        """
        Returns:
            dict: Maps keyword in @self.goal to its corresponding numerical shape. This should be static
                and analytically computed prior to any controller steps being taken
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
            th.tensor: Array filled with values specified in @nums
        """
        # First run sanity check to make sure no strings are being inputted
        if isinstance(nums, str):
            raise TypeError("Error: Only numeric inputs are supported for this function, nums2array!")

        # Check if input is an Iterable, if so, we simply convert the input to th.tensor and return
        # Else, input is a single value, so we map to a numpy array of correct size and return
        return (
            nums
            if isinstance(nums, th.Tensor)
            else (
                th.tensor(nums, dtype=th.float32)
                if isinstance(nums, Iterable)
                else th.ones(dim, dtype=th.float32) * nums
            )
        )

    @property
    def state_size(self):
        # Default is goal dim + 1 (for whether the goal is valid or not)
        return self.goal_dim + 1

    @property
    def goal(self):
        """
        Returns:
            dict: Current goal for this controller. Maps relevant goal keys to goal values to be
                used during controller step computations
        """
        return self._goal

    @property
    def goal_dim(self):
        """
        Returns:
            int: Expected size of flattened, internal goals
        """
        return self._goal_dim

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
        return self._dof_idx

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
