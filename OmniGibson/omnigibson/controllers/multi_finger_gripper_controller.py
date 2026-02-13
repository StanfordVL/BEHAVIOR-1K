import torch as th

from omnigibson.controllers.controller_base import BaseController, ControlType, IsGraspingState
from omnigibson.macros import create_module_macros
from omnigibson.utils.backend_utils import _compute_backend as cb
from omnigibson.utils.processing_utils import MovingAverageFilter
from omnigibson.utils.python_utils import assert_valid_key
from copy import deepcopy

VALID_MODES = {
    "binary",
    "smooth",
    "independent",
}


# Create settings for this module
m = create_module_macros(module_path=__file__)

# is_grasping heuristics parameters
m.POS_TOLERANCE = 0.002  # arbitrary heuristic
m.VEL_TOLERANCE = 0.02  # arbitrary heuristic


class MultiFingerGripperController(BaseController):
    """
    Controller class for multi finger gripper control. This either interprets an input as a binary
    command (open / close), continuous command (open / close with scaled velocities), or per-joint continuous command

    Each controller step consists of the following:
        1. Clip + Scale inputted command according to @command_input_limits and @command_output_limits
        2a. Convert command into gripper joint control signals
        2b. Clips the resulting control by the motor limits
    """
    @classmethod
    def _process_config(cls, controller_id: str, input_config: dict):
        config = deepcopy(input_config)
        assert_valid_key(key=config["motor_type"].lower(), valid_keys=ControlType.VALID_TYPES_STR, name="motor_type")
        config["motor_type"] = config["motor_type"].lower()
        assert_valid_key(key=config.get("mode", "binary"), valid_keys=VALID_MODES, name="mode for multi finger gripper")
        config["mode"] = config.get("mode", "binary")
        config["inverted"] = config.get("inverted", False)
        config["limit_tolerance"] = config.get("limit_tolerance", 0.001)
        config["open_qpos"] = (
            None if config.get("open_qpos", None) is None else cb.array(config.get("open_qpos"))
        )
        config["closed_qpos"] = (
            None if config.get("closed_qpos", None) is None else cb.array(config.get("closed_qpos"))
        )

        if config["mode"] == "binary":
            config["command_output_limits"] = "default"

        return super()._process_config(controller_id, config)

    @classmethod
    def _init_state(cls, controller_id: str):
        config = cls._configs[controller_id]
        cls._state[controller_id]["is_grasping"] = IsGraspingState.FALSE
        cls._state[controller_id]["vel_filter"] = MovingAverageFilter(obs_dim=len(cls.dof_idx(controller_id)), filter_width=5)

    @classmethod
    def _generate_default_command_output_limits(cls, controller_id: str):
        config = cls._configs[controller_id]
        command_output_limits = super()._generate_default_command_output_limits(controller_id)
        if config["mode"] == "binary":
            command_output_limits = (-1.0, 1.0)
        elif config["mode"] == "smooth":
            command_output_limits = (
                cb.mean(command_output_limits[0]),
                cb.mean(command_output_limits[1]),
            )
        elif config["mode"] == "independent":
            pass
        else:
            raise ValueError(f"Invalid mode {config['mode']}")
        return command_output_limits

    @classmethod
    def reset(cls, controller_id: str):
        super().reset(controller_id=controller_id)
        cls._state[controller_id]["vel_filter"].reset()
        cls._state[controller_id]["is_grasping"] = IsGraspingState.FALSE

    @classmethod
    def _preprocess_command(cls, controller_id: str, command):
        config = cls._configs[controller_id]
        if config["mode"] != "independent":
            command = (
                cb.array([command] * cls.command_dim(controller_id=controller_id))
                if type(command) in {int, float}
                else cb.array([command[0]] * cls.command_dim(controller_id=controller_id))
            )
        if config["inverted"]:
            command = config["command_input_limits"][1] - (command - config["command_input_limits"][0])
        return super()._preprocess_command(controller_id, command=command)

    @classmethod
    def _update_goal(cls, controller_id: str, command, control_dict):
        return dict(target=command)

    @classmethod
    def compute_control(cls, controller_id: str, control_dict):
        config = cls._configs[controller_id]
        target = cls._goals[controller_id]["target"]
        joint_pos = control_dict["joint_position"][cls.dof_idx(controller_id)]
        if config["mode"] == "binary":
            should_open = target[0] >= 0.0 if not config["inverted"] else target[0] > 0.0
            if should_open:
                u = (
                    config["control_limits"][ControlType.get_type(config["motor_type"])][1][cls.dof_idx(controller_id)]
                    if config["open_qpos"] is None
                    else config["open_qpos"]
                )
            else:
                u = (
                    config["control_limits"][ControlType.get_type(config["motor_type"])][0][cls.dof_idx(controller_id)]
                    if config["closed_qpos"] is None
                    else config["closed_qpos"]
                )
        else:
            # Use continuous signal. Make sure to go from command to control dim.
            u = cb.full((cls.control_dim(controller_id=controller_id),), target[0]) if len(target) == 1 else target


        # If we're near the joint limits and we're using velocity / torque control, we zero out the action
        if config["motor_type"] in {"velocity", "torque"}:
            violate_upper_limit = (
                joint_pos > config["control_limits"][ControlType.POSITION][1][cls.dof_idx(controller_id)] - config["limit_tolerance"]
            )
            violate_lower_limit = (
                joint_pos < config["control_limits"][ControlType.POSITION][0][cls.dof_idx(controller_id)] + config["limit_tolerance"]
            )
            violation = cb.logical_or(violate_upper_limit * (u > 0), violate_lower_limit * (u < 0))
            u *= ~violation
        cls._update_grasping_state(controller_id=controller_id, control_dict=control_dict)
        return u

    @classmethod
    def _update_grasping_state(cls, controller_id: str, control_dict):
        config = cls._configs[controller_id]
        """
        Updates internal inferred grasping state of the gripper being controlled by this gripper controller

        Args:
            control_dict (dict): dictionary that should include any relevant keyword-mapped
                states necessary for controller computation. Must include the following keys:

                    joint_position: Array of current joint positions
                    joint_velocity: Array of current joint velocities
        """
        # Update velocity history
        finger_vel = cls._state[controller_id]["vel_filter"].estimate(control_dict["joint_velocity"][cls.dof_idx(controller_id)])

        # Calculate grasping state based on mode of this controller
        # Independent mode of MultiFingerGripperController does not have any good heuristics to determine is_grasping
        if config["mode"] == "independent":
            is_grasping = IsGraspingState.UNKNOWN

        # No control has been issued before -- we assume not grasping
        elif cls.control(controller_id) is None:
            is_grasping = IsGraspingState.FALSE

        #  Different values in the command for non-independent mode - cannot use heuristics
        elif not cb.all(cls.control(controller_id) == cls.control(controller_id)[0]):
            is_grasping = IsGraspingState.UNKNOWN

        # Joint position tolerance for is_grasping heuristics checking is smaller than or equal to the gripper
        # controller's tolerance of zero-ing out velocities, which makes the heuristics invalid.
        elif not m.POS_TOLERANCE > config["limit_tolerance"]:
            is_grasping = IsGraspingState.UNKNOWN

        else:
            finger_pos = control_dict["joint_position"][cls.dof_idx(controller_id)]

            # For joint position control, if the desired positions are the same as the current positions, is_grasping unknown
            if config["motor_type"] == "position" and cb.abs(finger_pos - cls.control(controller_id)).mean() < m.POS_TOLERANCE:
                is_grasping = IsGraspingState.UNKNOWN

            # For joint velocity / torque control, if the desired velocities / torques are zeros, is_grasping unknown
            elif config["motor_type"] in {"velocity", "torque"} and cb.abs(cls.control(controller_id)).mean() < m.VEL_TOLERANCE:
                is_grasping = IsGraspingState.UNKNOWN

            # Otherwise, the last control signal intends to "move" the gripper
            else:
                min_pos = config["control_limits"][ControlType.POSITION][0][cls.dof_idx(controller_id)]
                max_pos = config["control_limits"][ControlType.POSITION][1][cls.dof_idx(controller_id)]

                # Make sure we don't have any invalid values (i.e.: fingers should be within the limits)
                finger_pos = finger_pos.clip(min_pos, max_pos)

                # Check distance from both ends of the joint limits
                dist_from_lower_limit = finger_pos - min_pos
                dist_from_upper_limit = max_pos - finger_pos

                # If either of the joint positions are not near the joint limits with some tolerance (m.POS_TOLERANCE)
                valid_grasp_pos = (
                    dist_from_lower_limit.mean() > m.POS_TOLERANCE or dist_from_upper_limit.mean() > m.POS_TOLERANCE
                )

                # And the joint velocities are close to zero with some tolerance (m.VEL_TOLERANCE)
                valid_grasp_vel = cb.all(cb.abs(finger_vel) < m.VEL_TOLERANCE)

                # Then the gripper is grasping something, which stops the gripper from reaching its desired state
                is_grasping = IsGraspingState.TRUE if valid_grasp_pos and valid_grasp_vel else IsGraspingState.FALSE

        # Store calculated state
        cls._state[controller_id]["is_grasping"] = is_grasping

    @classmethod
    def compute_no_op_goal(cls, controller_id: str, control_dict):
        config = cls._configs[controller_id]
        if config["mode"] == "binary":
            goal_sign = -1 if cls.is_grasping(controller_id) == IsGraspingState.TRUE else 1
            if config["inverted"]:
                goal_sign = -1 * goal_sign
            target = cb.array([goal_sign])
        else:
            if config["motor_type"] == "position":
                target = control_dict["joint_position"][cls.dof_idx(controller_id)]
            elif config["motor_type"] == "velocity":
                target = cb.zeros(cls.command_dim(controller_id))
            else:
                raise ValueError("Cannot compute noop action for effort motor type.")

            # Convert to binary / smooth mode if necessary
            if config["mode"] == "smooth":
                target = cb.mean(target, dim=-1, keepdim=True)

        return dict(target=target)

    @classmethod
    def _compute_no_op_command(cls, controller_id: str, control_dict):
        config = cls._configs[controller_id]
        # Take care of the special case of binary control
        if config["mode"] == "binary":
            command_val = -1 if cls.is_grasping(controller_id) == IsGraspingState.TRUE else 1
            if config["inverted"]:
                command_val = -1 * command_val
            return cb.array([command_val])

        if config["motor_type"] == "position":
            command = control_dict["joint_position"][cls.dof_idx(controller_id)]
        elif config["motor_type"] == "velocity":
            command = cb.zeros(cls.command_dim(controller_id))
        else:
            raise ValueError("Cannot compute noop action for effort motor type.")

        # Convert to binary / smooth mode if necessary
        if config["mode"] == "smooth":
            command = cb.mean(command, dim=-1, keepdim=True)
        return command

    @classmethod
    def _get_goal_shapes(cls, controller_id: str):
        return dict(target=(cls.command_dim(controller_id),))

    @classmethod
    def is_grasping(cls, controller_id: str):
        return cls._state[controller_id]["is_grasping"]

    @classmethod
    def control_type(cls, controller_id: str):
        config = cls._configs[controller_id]
        return ControlType.get_type(type_str=config["motor_type"])

    @classmethod
    def command_dim(cls, controller_id: str):
        config = cls._configs[controller_id]
        if config["mode"] == "independent":
            return len(cls.dof_idx(controller_id))
        return 1

    @classmethod
    def _dump_state(cls, controller_id: str):
        state = super()._dump_state(controller_id=controller_id)
        state["vel_filter"] = cls._state[controller_id]["vel_filter"].dump_state(serialized=False)
        return state

    @classmethod
    def _load_state(cls, controller_id: str, state):
        super()._load_state(controller_id=controller_id, state=state)
        if cls._goals[controller_id] is not None:
            cls._state[controller_id]["vel_filter"].load_state(state["vel_filter"], serialized=False)

    @classmethod
    def serialize(cls, controller_id: str, state):
        state_flat = super().serialize(controller_id=controller_id, state=state)
        return th.cat([state_flat, cls._state[controller_id]["vel_filter"].serialize(state=state["vel_filter"])])

    @classmethod
    def deserialize(cls, controller_id: str, state):
        state_dict, idx = super().deserialize(controller_id=controller_id, state=state)
        state_dict["vel_filter"], deserialized_items = cls._state[controller_id]["vel_filter"].deserialize(
            state=state[idx:]
        )
        idx += deserialized_items
        return state_dict, idx
