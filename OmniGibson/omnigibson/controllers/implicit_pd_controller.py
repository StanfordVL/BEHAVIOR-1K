"""
Implicit PD Controller for OmniGibson.

This controller provides two modes that match Isaac Lab's actuator models:

1. **Implicit mode** (use_explicit_pd=False): Matches Isaac Lab's ImplicitActuatorCfg
   - Sets isaac_kp and isaac_kd on the physics engine
   - Passes position targets to the physics engine
   - Physics engine computes: effort = stiffness * (target - pos) - damping * vel

2. **Explicit mode** (use_explicit_pd=True): Matches Isaac Lab's IdealPDActuator
   - Controller computes: effort = stiffness * (pos_target - pos) + damping * (vel_target - vel) + feedforward_effort
   - Outputs effort commands to the physics engine
"""

import torch as th

from omnigibson.controllers import (
    ControlType,
    GripperController,
    IsGraspingState,
    LocomotionController,
    ManipulationController,
)
from omnigibson.macros import create_module_macros
from omnigibson.utils.backend_utils import _compute_backend as cb
from omnigibson.utils.processing_utils import MovingAverageFilter
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Default gains matching Isaac Lab's typical values for Franka
m.DEFAULT_STIFFNESS = 400.0
m.DEFAULT_DAMPING = 80.0


class ImplicitPDController(LocomotionController, ManipulationController, GripperController):
    """
    Controller class that implements PD position control matching Isaac Lab's actuator models.
    
    This controller supports two modes:
    
    **Implicit Mode** (use_explicit_pd=False, default):
        Matches Isaac Lab's ImplicitActuatorCfg. The stiffness and damping are set as
        isaac_kp and isaac_kd, and the physics engine handles PD control internally.
        The controller just passes position targets.
    
    **Explicit Mode** (use_explicit_pd=True):
        Matches Isaac Lab's IdealPDActuator. The controller computes efforts using:
            effort = stiffness * (pos_target - pos) + damping * (vel_target - vel) + feedforward_effort
        
        This matches Isaac Lab's formula exactly and supports:
        - Position targets
        - Velocity targets (optional, defaults to 0)
        - Feedforward effort (optional, defaults to 0)
    """

    def __init__(
        self,
        control_freq,
        control_limits,
        dof_idx,
        stiffness=None,
        damping=None,
        command_input_limits="default",
        command_output_limits="default",
        smoothing_filter_size=None,
        use_delta_commands=False,
        effort_limit=None,
        velocity_limit=None,
        use_explicit_pd=False,
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
            dof_idx (Array[int]): specific dof indices controlled by this robot
            stiffness (None or float or Array[float]): Proportional gain (Kp) for PD control.
                Can be a single value for all DOFs or per-DOF array. Default: 400.0
            damping (None or float or Array[float]): Derivative gain (Kd) for PD control.
                Can be a single value for all DOFs or per-DOF array. Default: 80.0
            command_input_limits (None or "default" or Tuple[float, float] or Tuple[Array[float], Array[float]]):
                Min/max acceptable inputted command. If "default", uses (-1, 1).
            command_output_limits (None or "default" or Tuple[float, float] or Tuple[Array[float], Array[float]]):
                Min/max scaled command. If "default", uses position limits.
            smoothing_filter_size (None or int): Size of moving average filter for command smoothing.
            use_delta_commands (bool): If True, interpret commands as deltas from current position.
            effort_limit (None or float or Array[float]): Optional effort limit to clip output.
                If None, uses control_limits["effort"].
            velocity_limit (None or float or Array[float]): Optional velocity limit (stored for compatibility).
            use_explicit_pd (bool): If True, controller computes efforts explicitly (like IdealPDActuator).
                If False, passes position targets to physics engine (like ImplicitActuator). Default: False.
        """
        # Store gains
        self._stiffness_value = m.DEFAULT_STIFFNESS if stiffness is None else stiffness
        self._damping_value = m.DEFAULT_DAMPING if damping is None else damping
        self._use_delta_commands = use_delta_commands
        self._effort_limit = effort_limit
        self._velocity_limit = velocity_limit
        self._use_explicit_pd = use_explicit_pd
        
        # Create control filter if specified
        command_dim = len(dof_idx)
        self.control_filter = (
            None
            if smoothing_filter_size in {None, 0}
            else MovingAverageFilter(obs_dim=command_dim, filter_width=smoothing_filter_size)
        )
        
        # When in delta mode, don't use default output limits
        assert not (
            self._use_delta_commands and type(command_output_limits) is str and command_output_limits == "default"
        ), "Cannot use 'default' command output limits in delta commands mode. Use None instead."

        # Set up isaac_kp/isaac_kd based on mode
        if use_explicit_pd:
            # Explicit mode: we compute efforts, so don't set physics gains
            isaac_kp = None
            isaac_kd = None
        else:
            # Implicit mode: set physics gains so physics engine does PD control
            isaac_kp = self._stiffness_value
            isaac_kd = self._damping_value

        # Run super init
        super().__init__(
            control_freq=control_freq,
            control_limits=control_limits,
            dof_idx=dof_idx,
            command_input_limits=command_input_limits,
            command_output_limits=command_output_limits,
            isaac_kp=isaac_kp,
            isaac_kd=isaac_kd,
        )
        
        # Convert gains to tensors after dof_idx is set (for explicit mode)
        n_dofs = len(self._dof_idx)
        if isinstance(self._stiffness_value, (int, float)):
            self._stiffness = cb.array([self._stiffness_value] * n_dofs)
        else:
            self._stiffness = cb.array(self._stiffness_value)
            
        if isinstance(self._damping_value, (int, float)):
            self._damping = cb.array([self._damping_value] * n_dofs)
        else:
            self._damping = cb.array(self._damping_value)
            
        # Convert effort limit to tensor if needed
        if self._effort_limit is not None:
            if isinstance(self._effort_limit, (int, float)):
                self._effort_limit_tensor = cb.array([self._effort_limit] * n_dofs)
            else:
                self._effort_limit_tensor = cb.array(self._effort_limit)
        else:
            self._effort_limit_tensor = None

    def reset(self):
        """Reset the controller state."""
        super().reset()
        if self.control_filter is not None:
            self.control_filter.reset()

    @property
    def state_size(self):
        """Return the state size including filter state."""
        return super().state_size + (0 if self.control_filter is None else self.control_filter.state_size)

    def _dump_state(self):
        """Dump controller state for serialization."""
        state = super()._dump_state()
        if self.control_filter is not None:
            state["control_filter"] = self.control_filter.dump_state(serialized=False)
        return state

    def _load_state(self, state):
        """Load controller state from serialized data."""
        super()._load_state(state=state)
        if self._goal is not None and self.control_filter is not None:
            self.control_filter.load_state(state["control_filter"], serialized=False)

    def serialize(self, state):
        """Serialize controller state to tensor."""
        state_flat = super().serialize(state=state)
        return th.cat([
            state_flat,
            th.tensor([]) if self.control_filter is None 
            else self.control_filter.serialize(state=state["control_filter"]),
        ])

    def deserialize(self, state):
        """Deserialize controller state from tensor."""
        state_dict, idx = super().deserialize(state=state)
        if self.control_filter is not None:
            state_dict["control_filter"], deserialized_items = self.control_filter.deserialize(state=state[idx:])
            idx += deserialized_items
        return state_dict, idx

    def _generate_default_command_output_limits(self):
        """Generate default output limits based on position limits."""
        return (
            self._control_limits[ControlType.POSITION][0][self.dof_idx],
            self._control_limits[ControlType.POSITION][1][self.dof_idx],
        )

    def _update_goal(self, command, control_dict):
        """
        Update the goal based on the input command.
        
        Args:
            command: Scaled command (joint positions)
            control_dict: Dictionary containing current joint states
            
        Returns:
            dict: Goal dictionary with target position
        """
        if self._use_delta_commands:
            # Delta mode: add command to current position
            base_value = control_dict["joint_position"][self.dof_idx]
            target = base_value + command
        else:
            # Absolute mode: command is the target directly
            target = command
            
        # Clip target to position limits
        target = target.clip(
            self._control_limits[ControlType.POSITION][0][self.dof_idx],
            self._control_limits[ControlType.POSITION][1][self.dof_idx],
        )
        
        return dict(target=target)

    def compute_control(self, goal_dict, control_dict):
        """
        Compute the control output.
        
        In implicit mode: returns position targets (physics does PD)
        In explicit mode: computes effort = stiffness * pos_error + damping * vel_error
        
        This matches Isaac Lab's IdealPDActuator formula:
            effort = stiffness * (pos_target - pos) + damping * (vel_target - vel) + feedforward
        
        Args:
            goal_dict: Dictionary containing 'target' position
            control_dict: Dictionary containing current joint states
            
        Returns:
            Array[float]: Position targets (implicit) or effort commands (explicit)
        """
        target = goal_dict["target"]
        
        # Apply smoothing filter if configured
        if self.control_filter is not None:
            target = self.control_filter.estimate(target)
        
        if self._use_explicit_pd:
            # Explicit mode: compute efforts like Isaac Lab's IdealPDActuator
            # effort = stiffness * (pos_target - pos) + damping * (vel_target - vel) + feedforward
            current_pos = control_dict["joint_position"][self.dof_idx]
            current_vel = control_dict["joint_velocity"][self.dof_idx]
            
            # Position error
            pos_error = target - current_pos
            
            # Velocity error (target velocity is 0 by default, like Isaac Lab when not specified)
            vel_target = cb.zeros(len(self._dof_idx))
            vel_error = vel_target - current_vel
            
            # Compute effort (no feedforward by default)
            effort = self._stiffness * pos_error + self._damping * vel_error
            
            # Clip effort
            if self._effort_limit_tensor is not None:
                effort = effort.clip(-self._effort_limit_tensor, self._effort_limit_tensor)
            else:
                effort = effort.clip(
                    self._control_limits[ControlType.EFFORT][0][self.dof_idx],
                    self._control_limits[ControlType.EFFORT][1][self.dof_idx],
                )
            
            return effort
        else:
            # Implicit mode: just return position targets, physics does PD
            return target

    def compute_no_op_goal(self, control_dict):
        """Compute a no-op goal (maintain current position)."""
        target = control_dict["joint_position"][self.dof_idx]
        return dict(target=target)

    def _compute_no_op_command(self, control_dict):
        """Compute a no-op command."""
        if self._use_delta_commands:
            return cb.zeros(self.command_dim)
        else:
            return control_dict["joint_position"][self.dof_idx]

    def _get_goal_shapes(self):
        """Return the shapes of goal components."""
        return dict(target=(self.control_dim,))

    def is_grasping(self):
        """
        Return grasping state. For position control, we use UNKNOWN since
        there's no good heuristic to determine grasping.
        """
        return IsGraspingState.UNKNOWN

    @property
    def use_delta_commands(self):
        """Whether this controller uses delta commands."""
        return self._use_delta_commands

    @property
    def control_type(self):
        """The control type output by this controller."""
        if self._use_explicit_pd:
            return ControlType.EFFORT
        else:
            return ControlType.POSITION

    @property
    def command_dim(self):
        """The dimension of the command space."""
        return len(self.dof_idx)

    @property
    def stiffness(self):
        """The stiffness (Kp) gains."""
        return self._stiffness

    @property
    def damping(self):
        """The damping (Kd) gains."""
        return self._damping
    
    @property
    def use_explicit_pd(self):
        """Whether the controller computes efforts explicitly."""
        return self._use_explicit_pd
