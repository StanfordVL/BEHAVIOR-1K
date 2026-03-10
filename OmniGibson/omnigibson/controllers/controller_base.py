import math
from collections.abc import Iterable
from enum import IntEnum

import torch as th

from omnigibson.macros import create_module_macros
from omnigibson.utils.backend_utils import _compute_backend as cb
from omnigibson.utils.python_utils import Recreatable, Registerable, Serializable, assert_valid_key, classproperty
from omnigibson.utils.usd_utils import ControllableObjectViewAPI

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Set default isaac kp / kd for controllers
m.DEFAULT_ISAAC_KP = 1e7
m.DEFAULT_ISAAC_KD = 1e5

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

    Each instance represents a group of controllers that share the same
    (robot kinematic tree pattern, body_part, controller_config) key. Members are added via
    add_member() and assigned a controller_idx. All members are stepped together in a single
    batched call, with per-member state stored in indexed tensors.
    """

    def __init__(
        self,
        control_freq,
        control_limits,
        dof_idx,
        command_input_limits="default",
        command_output_limits="default",
        isaac_kp=None,
        isaac_kd=None,
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
            isaac_kp (None or float or Array[float]): If specified, stiffness gains to apply to the underlying
                isaac DOFs. Can either be a single number or a per-DOF set of numbers.
                Should only be nonzero if self.control_type is position
            isaac_kd (None or float or Array[float]): If specified, damping gains to apply to the underlying
                isaac DOFs. Can either be a single number or a per-DOF set of numbers
                Should only be nonzero if self.control_type is position or velocity
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
        self._dof_idx = cb.as_int(dof_idx)

        # Generate goal information
        self._goal_shapes = self._get_goal_shapes()
        self._goal_dim = int(sum(cb.prod(cb.array(shape)) for shape in self._goal_shapes.values()))

        # Multi-controller batching state
        # List of articulation root paths for each group member
        self._articulation_root_paths = []
        # Batched goals: key -> (N, *shape) tensor, built lazily as members are added
        self._goals = {}
        # Per-member flag indicating whether goal has been set this step
        self._goal_set = []
        # Per-member control enabled mask (1 enabled, 0 disabled)
        self._control_enabled = []
        # Per-member last deployed control tensor (set in _write_control)
        self._controls = []
        # Per-member tombstone mask: 0 = active, 1 = unregistered (permanently ignored)
        self._unregistered_controllers = th.zeros(0, dtype=th.long)
        # Lazy-init: row indices of each member in the shared view (None until first _write_control call)
        self._view_row_indices = None

        # Initialize command scaling variables
        self._command_scale_factor = None
        self._command_output_transform = None
        self._command_input_transform = None

        # Standardize command input / output limits to be (min_array, max_array)
        command_input_limits = (
            self._generate_default_command_input_limits()
            if type(command_input_limits) is str and command_input_limits == "default"
            else command_input_limits
        )
        command_output_limits = (
            self._generate_default_command_output_limits()
            if type(command_output_limits) is str and command_output_limits == "default"
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

        # Set gains
        if self.control_type == ControlType.POSITION:
            # Set default kp / kd values if not specified
            isaac_kp = m.DEFAULT_ISAAC_KP if isaac_kp is None else isaac_kp
            isaac_kd = m.DEFAULT_ISAAC_KD if isaac_kd is None else isaac_kd
        elif self.control_type == ControlType.VELOCITY:
            # No kp should be specified, but kd should be
            assert (
                isaac_kp is None
            ), f"Control type for controller {self.__class__.__name__} is VELOCITY, so no isaac_kp should be set!"
            isaac_kd = m.DEFAULT_ISAAC_KP if isaac_kd is None else isaac_kd
        elif self.control_type == ControlType.EFFORT:
            # Neither kp nor kd should be specified
            assert (
                isaac_kp is None
            ), f"Control type for controller {self.__class__.__name__} is EFFORT, so no isaac_kp should be set!"
            assert (
                isaac_kd is None
            ), f"Control type for controller {self.__class__.__name__} is EFFORT, so no isaac_kd should be set!"
        else:
            raise ValueError(
                f"Expected control type to be one of: [POSITION, VELOCITY, EFFORT], but got: {self.control_type}"
            )

        self._isaac_kp = None if isaac_kp is None else self.nums2array(isaac_kp, self.control_dim)
        self._isaac_kd = None if isaac_kd is None else self.nums2array(isaac_kd, self.control_dim)

    def add_member(self, articulation_root_path, control_enabled=True):
        """
        Register a controller as a member of this controller group.

        Reuses the first tombstoned (unregistered) slot if one exists, otherwise appends a new slot.

        Args:
            articulation_root_path (str): articulation root prim path of the new group member

        Returns:
            int: controller_idx — index into this group's member arrays for the added controller,
                used to access goal, control, and articulation_root_path
        """
        # Reuse the first tombstoned slot if available
        tombstone_indices = (self._unregistered_controllers == 1).nonzero(as_tuple=True)[0]
        if len(tombstone_indices) > 0:
            controller_idx = tombstone_indices[0].item()
            # Reset the reused slot in-place
            self._articulation_root_paths[controller_idx] = articulation_root_path
            self._goal_set[controller_idx] = False
            self._control_enabled[controller_idx] = 1 if control_enabled else 0
            self._controls[controller_idx] = None
            self._unregistered_controllers[controller_idx] = 0
            for key, shape in self._goal_shapes.items():
                self._goals[key][controller_idx] = cb.zeros(shape)
        else:
            # No tombstone available — append a new slot
            controller_idx = len(self._articulation_root_paths)
            self._articulation_root_paths.append(articulation_root_path)
            self._goal_set.append(False)
            self._control_enabled.append(1 if control_enabled else 0)
            self._controls.append(None)
            self._unregistered_controllers = th.cat([self._unregistered_controllers, th.zeros(1, dtype=th.long)])
            for key, shape in self._goal_shapes.items():
                new_row = cb.zeros((1, *shape))
                if key in self._goals:
                    self._goals[key] = cb.cat([self._goals[key], new_row], 0)
                else:
                    self._goals[key] = new_row

        # Always invalidate cached view indices so the next _write_control re-resolves them
        self._view_row_indices = None
        return controller_idx

    @property
    def n_members(self):
        """
        Returns:
            int: Number of controllers registered in this controller group
        """
        return len(self._articulation_root_paths)

    def _generate_default_command_input_limits(self):
        """
        Generates default command input limits based on the control limits

        Returns:
            2-tuple:
                - int or array: min command input limits
                - int or array: max command input limits
        """
        return (-1.0, 1.0)

    def _generate_default_command_output_limits(self):
        """
        Generates default command output limits based on the control limits

        Returns:
            2-tuple:
                - int or array: min command output limits
                - int or array: max command output limits
        """
        return (
            self._control_limits[self.control_type][0][self.dof_idx],
            self._control_limits[self.control_type][1][self.dof_idx],
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
        command = cb.array([command]) if type(command) in {int, float} else command
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

    def update_goal(self, controller_idx, command):
        """
        Updates the goal for controller at @controller_idx with the given @command.

        Args:
            controller_idx (int): index of the controller in this controller group
            command (Array[float]): inputted command to preprocess and extract relevant goal(s)
        """
        # Sanity check the command
        assert (
            len(command) == self.command_dim
        ), f"Commands must be dimension {self.command_dim}, got dim {len(command)} instead."

        preprocessed = self._preprocess_command(command)
        goal_dict = self._update_goal(controller_idx, preprocessed)
        for k, v in goal_dict.items():
            self._goals[k][controller_idx] = v

        self._goal_set[controller_idx] = True

    def _update_goal(self, controller_idx, command):
        """
        Updates the goal for the controller at @controller_idx.

        Args:
            controller_idx (int): index of the controller in this controller group
            command (Array[float]): preprocessed command

        Returns:
            dict: Keyword-mapped goal values for controller at controller_idx
        """
        raise NotImplementedError

    def compute_control(self, goals):
        """
        Converts batched @goals into deployable (non-clipped!) control signals for all member controllers.

        Args:
            goals (Dict[str, Tensor]): batched goals, each value has shape (N, *shape)

        Returns:
            Tensor: (N, control_dim) control signals
        """
        raise NotImplementedError

    def _ensure_view_row_indices(self):
        """Lazy-init: resolve each member's row index in the shared physics view."""
        if self._view_row_indices is None:
            self._view_row_indices = ControllableObjectViewAPI.get_member_view_indices(
                self.routing_path, self._articulation_root_paths
            )

    def _write_control(self):
        """Write batched control signals to Isaac via a single tensor op per control type."""
        self._ensure_view_row_indices()

        # Filter to enabled and non-unregistered members
        enabled_indices = [
            i
            for i, (e, u) in enumerate(zip(self._control_enabled, self._unregistered_controllers))
            if e != 0 and u == 0
        ]
        if not enabled_indices:
            return

        enabled_rows = [self._view_row_indices[i] for i in enabled_indices]
        enabled_controls = cb.stack([self._controls[i] for i in enabled_indices])  # (N_en, K)
        routing_path = self.routing_path

        if self.control_type == ControlType.POSITION:
            ControllableObjectViewAPI.set_all_joint_position_targets(
                routing_path, enabled_rows, enabled_controls, self.dof_idx
            )
            ControllableObjectViewAPI.set_all_joint_velocity_targets(
                routing_path, enabled_rows, cb.zeros(enabled_controls.shape), self.dof_idx
            )
        elif self.control_type == ControlType.VELOCITY:
            ControllableObjectViewAPI.set_all_joint_velocity_targets(
                routing_path, enabled_rows, enabled_controls, self.dof_idx
            )
        elif self.control_type == ControlType.EFFORT:
            ControllableObjectViewAPI.set_all_joint_efforts(routing_path, enabled_rows, enabled_controls, self.dof_idx)

    def clip_control(self, control):
        """
        Clips the inputted @control signal based on @control_limits.

        Args:
            control (Tensor): (N, control_dim) control signal to clip

        Returns:
            Tensor: Clipped (N, control_dim) control signal
        """
        lo = self._control_limits[self.control_type][0][self.dof_idx]
        hi = self._control_limits[self.control_type][1][self.dof_idx]
        # Broadcast limits: (control_dim,) -> (1, control_dim) for N-dim clipping
        clipped_control = control.clip(lo, hi)
        idx = (
            self._dof_has_limits[self.dof_idx]
            if self.control_type == ControlType.POSITION
            else (cb.ones(self.control_dim) > 0)
        )
        control[:, idx] = clipped_control[:, idx]
        return control

    def step(self):
        """
        Take a batched controller step across all member controller.

        For any controller that has not received a goal this step, a no-op goal is computed.
        The control is then computed for all controllers, clipped, written to Isaac, and the
        goal_set flags are reset.
        """
        N = self.n_members
        # If all active (non-unregistered) members are disabled, early return
        if all(self._control_enabled[i] == 0 or self._unregistered_controllers[i] == 1 for i in range(N)):
            return

        # Fill in no-op goals for any controllers that haven't received a goal.
        for i in range(N):
            # Skip control_disabled or unregistered members
            if self._control_enabled[i] == 0 or self._unregistered_controllers[i] == 1:
                continue
            if not self._goal_set[i]:
                no_op = self.compute_no_op_goal(i)
                for k, v in no_op.items():
                    self._goals[k][i] = v
                self._goal_set[i] = True

        # Compute batched control: (N, control_dim)
        control_output = self.compute_control(self._goals)
        assert control_output.shape == (
            N,
            self.control_dim,
        ), f"compute_control must return shape ({N}, {self.control_dim}), got {control_output.shape}"

        # Clip, store per-member controls, then write batched
        control_output = self.clip_control(control_output)
        for i in range(N):
            if self._control_enabled[i] != 0 and self._unregistered_controllers[i] == 0:
                ctrl = control_output[i]
                self._controls[i] = (
                    cb.from_torch(ctrl)
                    if isinstance(ctrl, th.Tensor)
                    else (ctrl if isinstance(ctrl, cb.arr_type) else cb.array(ctrl))
                )
        self._write_control()

    def reset(self, controller_idx):
        """
        Resets the goal state for the controller at @controller_idx. Can be extended by subclass

        Args:
            controller_idx (int): index of the controller in this controller group
        """
        if self._unregistered_controllers[controller_idx] == 1:
            return
        self._goal_set[controller_idx] = False
        self._controls[controller_idx] = None
        for k in self._goals:
            self._goals[k][controller_idx] = cb.zeros(self._goal_shapes[k])

    def unregister_member(self, controller_idx):
        """Mark member at controller_idx as a tombstone (permanently ignored)."""
        self._unregistered_controllers[controller_idx] = 1

    def has_no_active_members(self):
        """Return True if all members have been unregistered."""
        return bool(th.all(self._unregistered_controllers == 1).item())

    def set_control_enabled(self, controller_idx, enabled):
        self._control_enabled[controller_idx] = 1 if enabled else 0

    def compute_no_op_goal(self, controller_idx):
        """
        Compute no-op goal for the controller at @controller_idx.

        Args:
            controller_idx (int): index of the controller in this controller group

        Returns:
            dict: Maps relevant goal keys to goal data for that controller
        """
        raise NotImplementedError

    def compute_no_op_action(self, controller_idx):
        """
        Compute a no-op action that updates the goal to match the current position
        Disclaimer: this no-op might cause drift under external load (e.g. when the controller cannot reach the goal position)

        Args:
            controller_idx (int): index of the controller in this group

        Returns:
            Tensor: no-op action command
        """
        if not self._goal_set[controller_idx]:
            no_op_goal = self.compute_no_op_goal(controller_idx)
            for k, v in no_op_goal.items():
                self._goals[k][controller_idx] = v
        command = self._compute_no_op_command(controller_idx)
        return cb.to_torch(self._reverse_preprocess_command(command))

    def _compute_no_op_command(self, controller_idx):
        """
        Compute no-op command for the controller at @controller_idx.

        Args:
            controller_idx (int): index of the controller in this group

        Returns:
            Array: no-op command
        """
        raise NotImplementedError

    def _dump_state(self, controller_idx: int) -> dict:
        """Dump state for one controller member."""
        goals = {k: cb.to_torch(v[controller_idx]).clone() for k, v in self._goals.items()}
        return {
            "goal_set": bool(self._goal_set[controller_idx]),
            "goals": goals,
        }

    def dump_state(self, controller_idx: int, serialized: bool = False):
        """
        Dumps the state for a single controller member.

        Args:
            controller_idx (int): member index in this controller group
            serialized (bool): whether to return flattened serialized state

        Returns:
            dict or th.Tensor: member state for this controller_idx
        """
        state = self._dump_state(controller_idx=controller_idx)
        return self.serialize(state=state, controller_idx=controller_idx) if serialized else state

    def _load_state(self, controller_idx: int, state: dict):
        """Load state for one controller member."""
        goal_set = state["goal_set"]
        if isinstance(goal_set, th.Tensor):
            goal_set = bool(goal_set.item())
        # Loaded goals can be stale after scene reset / controller reload.
        # Force a fresh no-op goal on next step for this member.
        self._goal_set[controller_idx] = False
        self._controls[controller_idx] = None
        self._unregistered_controllers[controller_idx] = 0  # we won't load a unregistered controller
        for name, val in state["goals"].items():
            if name in self._goals:
                if isinstance(val, th.Tensor):
                    val = cb.from_torch(val)
                elif not isinstance(val, cb.arr_type):
                    val = cb.array(val)
                self._goals[name][controller_idx] = val

    def load_state(self, controller_idx: int, state, serialized: bool = False):
        """
        Loads state for a single controller member.

        Args:
            controller_idx (int): member index in this controller group
            state (dict or th.Tensor): member state payload
            serialized (bool): whether @state is serialized
        """
        if serialized:
            orig_state_len = len(state)
            state, deserialized_items = self.deserialize(state=state, controller_idx=controller_idx)
            assert deserialized_items == orig_state_len, (
                f"Invalid state deserialization occurred! Expected {orig_state_len} total "
                f"values to be deserialized, only {deserialized_items} were."
            )
        self._load_state(controller_idx=controller_idx, state=state)

    def serialize(self, state, controller_idx: int):
        goal_set_tensor = th.tensor([float(state["goal_set"])], dtype=th.float32)
        goals = state["goals"]
        goal_flat = th.cat([v.flatten() for v in goals.values()]) if goals else th.zeros(self.goal_dim)
        return th.cat([goal_set_tensor, goal_flat])

    def deserialize(self, state, controller_idx: int):
        goal_set = bool(state[0].item())
        idx = 1
        goals = {}
        for key, shape in self._goal_shapes.items():
            length = math.prod(shape)
            goals[key] = state[idx : idx + length].reshape(*shape)
            idx += length
        return dict(goal_set=goal_set, goals=goals), idx

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
            if isinstance(nums, cb.arr_type)
            else cb.array(nums)
            if isinstance(nums, Iterable)
            else cb.ones(dim) * nums
        )

    @property
    def state_size(self):
        # goal_set + goal vector for one controller_idx
        return 1 + self.goal_dim

    def get_goal(self, controller_idx):
        """
        Returns the current goal for the controller at @controller_idx.

        Args:
            controller_idx (int): index of the controller in this group

        Returns:
            dict: Maps goal keys to per-controller tensors of shape (*shape).
        """
        return {k: v[controller_idx] for k, v in self._goals.items()}

    @property
    def goal_dim(self):
        """
        Returns:
            int: Expected size of flattened goals for a single group member
        """
        return self._goal_dim

    def get_control(self, controller_idx):
        """
        Returns the last deployed control signal for the controller at @controller_idx.

        Args:
            controller_idx (int): index of the controller in this group

        Returns:
            Tensor or None: last control tensor of shape (control_dim,), or None if not yet stepped.
        """
        return self._controls[controller_idx]

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
    def isaac_kp(self):
        """
        Returns:
            None or Array[float]: Stiffness gains that should be applied to the underlying Isaac joint motors.
                None if not specified.
        """
        return self._isaac_kp

    @property
    def isaac_kd(self):
        """
        Returns:
            None or Array[float]: Stiffness gains that should be applied to the underlying Isaac joint motors.
                None if not specified.
        """
        return self._isaac_kd

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
    def routing_path(self):
        """
        Returns:
            str: Articulation root path of the first member, used as a routing key for
                ControllableObjectViewAPI pattern lookups. All members in a group share the
                same view pattern, so any member's path works; index 0 is canonical.
        """
        return self._articulation_root_paths[0]

    @property
    def dof_idx(self):
        """
        Returns:
            Array[int]: DOF indices corresponding to the specific DOFs being controlled by this controller group
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

    def is_grasping(self, controller_idx):
        """
        Checks whether the current state of this gripper being controlled is in a grasping state.
        Should be implemented by subclass.

        Args:
            controller_idx (int): index of the controller in this group. Used for MultiFingerGripperController.

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
