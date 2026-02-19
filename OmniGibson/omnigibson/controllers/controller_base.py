import math
from collections.abc import Iterable
from copy import deepcopy
from enum import IntEnum

import numpy as np
import torch as th
from numba import jit

import omnigibson.utils.transform_utils as TT
import omnigibson.utils.transform_utils_np as NT
from omnigibson.macros import create_module_macros
from omnigibson.utils.backend_utils import _compute_backend as cb
from omnigibson.utils.backend_utils import add_compute_function
from omnigibson.utils.geometry_utils import wrap_angle
from omnigibson.utils.processing_utils import MovingAverageFilter
from omnigibson.utils.python_utils import Recreatable, Serializable, assert_valid_key, torch_compile
from omnigibson.utils.ui_utils import create_module_logger

log = create_module_logger(module_name=__name__)

m = create_module_macros(module_path=__file__)
m.DEFAULT_ISAAC_KP = 1e7
m.DEFAULT_ISAAC_KD = 1e5
m.DEFAULT_JOINT_POS_KP = 50.0
m.DEFAULT_JOINT_POS_DAMPING_RATIO = 1.0
m.DEFAULT_JOINT_VEL_KP = 2.0
m.POS_TOLERANCE = 0.002
m.VEL_TOLERANCE = 0.02


class ControllerType(IntEnum):
    DifferentialDriveController = 0
    HolonomicBaseJointController = 1
    InverseKinematicsController = 2
    JointController = 3
    MultiFingerGripperController = 4
    NullJointController = 5
    OperationalSpaceController = 6


class IsGraspingState(IntEnum):
    TRUE = 1
    UNKNOWN = 0
    FALSE = -1


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
        assert_valid_key(key=type_str.lower(), valid_keys=cls._MAPPING, name="control type")
        return cls._MAPPING[type_str.lower()]


# Mode constants
IK_MODE_COMMAND_DIMS = {
    "absolute_pose": 6,
    "pose_absolute_ori": 6,
    "pose_delta_ori": 6,
    "position_fixed_ori": 3,
    "position_compliant_ori": 3,
}
IK_MODES = set(IK_MODE_COMMAND_DIMS.keys())

OSC_MODE_COMMAND_DIMS = {
    "absolute_pose": 6,
    "pose_absolute_ori": 6,
    "pose_delta_ori": 6,
    "position_fixed_ori": 3,
    "position_compliant_ori": 3,
}
OSC_MODES = set(OSC_MODE_COMMAND_DIMS.keys())

GRIPPER_MODES = {"binary", "smooth", "independent"}

_JOINT_TYPES = frozenset({
    ControllerType.JointController,
    ControllerType.NullJointController,
    ControllerType.HolonomicBaseJointController,
    ControllerType.InverseKinematicsController,
})


class Controller(Serializable, Recreatable):
    """
    Unified singleton controller class. All controller logic lives here, dispatched
    by ControllerType stored in _types[controller_id].
    """

    _types = {}
    _configs = {}
    _goals = {}
    _controls = {}
    _state = {}
    _robots = {}
    _control_dicts = {}
    _ROBOT_CONTROL_STEP_CACHE = {}

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    @classmethod
    def register(cls, controller_id: str, config: dict, robot=None, controller_type=None):
        cls._types[controller_id] = controller_type
        cls._configs[controller_id] = cls._process_config(controller_id, config)
        cls._goals.setdefault(controller_id, None)
        cls._controls.setdefault(controller_id, None)
        if controller_id not in cls._state:
            cls._state[controller_id] = {}
        if robot is not None:
            cls._robots[controller_id] = robot
        cls._init_state(controller_id=controller_id)

    @classmethod
    def unregister(cls, controller_id: str):
        cls._types.pop(controller_id, None)
        cls._configs.pop(controller_id, None)
        cls._goals.pop(controller_id, None)
        cls._controls.pop(controller_id, None)
        cls._state.pop(controller_id, None)
        cls._robots.pop(controller_id, None)
        cls._control_dicts.pop(controller_id, None)

    # -------------------------------------------------------------------------
    # Config processing
    # -------------------------------------------------------------------------

    @classmethod
    def _process_config(cls, controller_id: str, input_config: dict):
        config = deepcopy(input_config)
        ctype = cls._types[controller_id]

        # Phase 1: type-specific pre-processing (before joint processing)
        if ctype == ControllerType.InverseKinematicsController:
            cls._process_config_ik(config)
        elif ctype == ControllerType.HolonomicBaseJointController:
            cls._process_config_holonomic(config)
        elif ctype == ControllerType.OperationalSpaceController:
            cls._process_config_osc(config)
        elif ctype == ControllerType.DifferentialDriveController:
            cls._process_config_dd(config)
        elif ctype == ControllerType.MultiFingerGripperController:
            cls._process_config_gripper(config)

        # Phase 2: joint processing for joint-based types
        if ctype in _JOINT_TYPES:
            cls._process_config_joint(config)

        # Phase 3: base processing (common to all)
        return cls._process_config_base(controller_id, config)
    
    @classmethod
    def _process_config_joint(cls, config):
        motor_type = config["motor_type"].lower()
        assert_valid_key(key=motor_type, valid_keys=ControlType.VALID_TYPES_STR, name="motor_type")
        config["motor_type"] = motor_type
        config["use_delta_commands"] = config.get("use_delta_commands", False)
        config["compute_delta_in_quat_space"] = (
            [] if config.get("compute_delta_in_quat_space", None) is None else config["compute_delta_in_quat_space"]
        )

        pos_kp = config.get("pos_kp", None)
        pos_damping_ratio = config.get("pos_damping_ratio", None)
        vel_kp = config.get("vel_kp", None)
        if motor_type == "position":
            pos_kp = m.DEFAULT_JOINT_POS_KP if pos_kp is None else pos_kp
            pos_damping_ratio = m.DEFAULT_JOINT_POS_DAMPING_RATIO if pos_damping_ratio is None else pos_damping_ratio
        elif motor_type == "velocity":
            vel_kp = m.DEFAULT_JOINT_VEL_KP if vel_kp is None else vel_kp
            assert pos_damping_ratio is None, "Cannot set pos_damping_ratio for JointController with motor_type=velocity!"
        else:
            assert pos_kp is None, "Cannot set pos_kp for JointController with motor_type=effort!"
            assert pos_damping_ratio is None, "Cannot set pos_damping_ratio for JointController with motor_type=effort!"
            assert vel_kp is None, "Cannot set vel_kp for JointController with motor_type=effort!"

        config["pos_kp"] = pos_kp
        config["pos_kd"] = None if pos_kp is None or pos_damping_ratio is None else 2 * math.sqrt(pos_kp) * pos_damping_ratio
        config["vel_kp"] = vel_kp
        config["use_impedances"] = config.get("use_impedances", False)
        config["use_gravity_compensation"] = config.get("use_gravity_compensation", False)
        config["use_cc_compensation"] = config.get("use_cc_compensation", True)

        if config["use_gravity_compensation"]:
            log.warning(
                "JointController is using gravity compensation. This is an experimental feature that only works on "
                "fixed base robots. We do not recommend enabling this."
            )

        if config["use_delta_commands"] and config.get("command_output_limits", "default") == "default":
            raise AssertionError(
                "Cannot use 'default' command output limits in delta commands mode of JointController. Try None instead."
            )

    @classmethod
    def _process_config_holonomic(cls, config):
        assert len(config["dof_idx"]) == 3, f"Expected 3 DOFs for holonomic base control, got {len(config['dof_idx'])}"
        config["use_delta_commands"] = False
        config["compute_delta_in_quat_space"] = None

    @classmethod
    def _process_config_ik(cls, config):
        config["motor_type"] = "position"
        config["use_delta_commands"] = False
        config["mode"] = config.get("mode", "pose_delta_ori")
        assert config["mode"] in IK_MODES, f"Invalid ik mode specified! Valid options are: {IK_MODES}, got: {config['mode']}"
        config["use_impedances"] = config.get("use_impedances", False)
        config["smoothing_filter_size"] = config.get("smoothing_filter_size", None)
        config["workspace_pose_limiter"] = config.get("workspace_pose_limiter", None)
        config["condition_on_current_position"] = config.get("condition_on_current_position", True)
        config["reset_joint_pos"] = config["reset_joint_pos"][config["dof_idx"]]

        command_input_limits = config.get("command_input_limits", "default")
        command_output_limits = config.get("command_output_limits", (
            (-0.2, -0.2, -0.2, -0.5, -0.5, -0.5),
            (0.2, 0.2, 0.2, 0.5, 0.5, 0.5),
        ))

        if config["mode"] == "absolute_pose":
            assert command_input_limits is None, "command_input_limits should be None if using absolute_pose mode!"
            assert command_output_limits is None, "command_output_limits should be None if using absolute_pose mode!"

        if config["mode"] == "pose_absolute_ori":
            if command_input_limits is not None:
                if type(command_input_limits) is str and command_input_limits == "default":
                    command_input_limits = [
                        cb.array([-1.0, -1.0, -1.0, -math.pi, -math.pi, -math.pi]),
                        cb.array([1.0, 1.0, 1.0, math.pi, math.pi, math.pi]),
                    ]
                else:
                    command_input_limits[0][3:] = cb.array([-math.pi] * len(command_input_limits[0][3:]))
                    command_input_limits[1][3:] = cb.array([math.pi] * len(command_input_limits[1][3:]))
            if command_output_limits is not None:
                if not isinstance(command_output_limits, str) and isinstance(command_output_limits, Iterable):
                    command_output_limits = [
                        cb.array(command_output_limits[0]),
                        cb.array(command_output_limits[1]),
                    ]
                if type(command_output_limits) is str and command_output_limits == "default":
                    command_output_limits = [
                        cb.array([-1.0, -1.0, -1.0, -math.pi, -math.pi, -math.pi]),
                        cb.array([1.0, 1.0, 1.0, math.pi, math.pi, math.pi]),
                    ]
                else:
                    command_output_limits[0][3:] = cb.array([-math.pi] * len(command_output_limits[0][3:]))
                    command_output_limits[1][3:] = cb.array([math.pi] * len(command_output_limits[1][3:]))
        config["command_input_limits"] = command_input_limits
        config["command_output_limits"] = command_output_limits

    @classmethod
    def _process_config_osc(cls, config):
        control_dim = len(config["dof_idx"])
        config["use_gravity_compensation"] = config.get("use_gravity_compensation", False)
        config["use_cc_compensation"] = config.get("use_cc_compensation", True)
        if config["use_gravity_compensation"]:
            log.warning(
                "OperationalSpaceController is using gravity compensation. This is an experimental feature that only works on "
                "fixed base robots. We do not recommend enabling this."
            )

        kp = config.get("kp", 150.0)
        damping_ratio = config.get("damping_ratio", 1.0)
        kp_null = config.get("kp_null", 10.0)
        config["kp"] = cls.nums2array(nums=kp, dim=6) if kp is not None else None
        config["damping_ratio"] = damping_ratio
        config["kp_null"] = cls.nums2array(nums=kp_null, dim=control_dim) if kp_null is not None else None
        config["kd_null"] = 2 * cb.sqrt(config["kp_null"]) if kp_null is not None else None
        config["kp_limits"] = cb.array(config.get("kp_limits", (10.0, 300.0)))
        config["damping_ratio_limits"] = cb.array(config.get("damping_ratio_limits", (0.0, 2.0)))
        config["kp_null_limits"] = cb.array(config.get("kp_null_limits", (0.0, 50.0)))

        config["variable_kp"] = config["kp"] is None
        config["variable_damping_ratio"] = config["damping_ratio"] is None
        config["variable_kp_null"] = config["kp_null"] is None
        assert True not in {
            config["variable_kp"],
            config["variable_damping_ratio"],
            config["variable_kp_null"],
        }, "Variable gains with OSC is not supported yet!"

        mode = config.get("mode", "pose_delta_ori")
        assert_valid_key(key=mode, valid_keys=OSC_MODES, name="OSC mode")
        config["mode"] = mode

        command_input_limits = config.get("command_input_limits", "default")
        command_output_limits = config.get("command_output_limits", ((-0.2, -0.2, -0.2, -0.5, -0.5, -0.5), (0.2, 0.2, 0.2, 0.5, 0.5, 0.5)))

        if mode == "absolute_pose":
            assert command_input_limits is None, "command_input_limits should be None if using absolute_pose mode!"
            assert command_output_limits is None, "command_output_limits should be None if using absolute_pose mode!"

        if mode == "pose_absolute_ori":
            if command_input_limits is not None:
                if type(command_input_limits) is str and command_input_limits == "default":
                    command_input_limits = [
                        [-1.0, -1.0, -1.0, -math.pi, -math.pi, -math.pi],
                        [1.0, 1.0, 1.0, math.pi, math.pi, math.pi],
                    ]
                else:
                    command_input_limits[0][3:] = -math.pi
                    command_input_limits[1][3:] = math.pi
            if command_output_limits is not None:
                if type(command_output_limits) is str and command_output_limits == "default":
                    command_output_limits = [
                        [-1.0, -1.0, -1.0, -math.pi, -math.pi, -math.pi],
                        [1.0, 1.0, 1.0, math.pi, math.pi, math.pi],
                    ]
                else:
                    command_output_limits[0][3:] = -math.pi
                    command_output_limits[1][3:] = math.pi

        is_input_limits_numeric = not (command_input_limits is None or isinstance(command_input_limits, str))
        is_output_limits_numeric = not (command_output_limits is None or isinstance(command_output_limits, str))
        if is_input_limits_numeric:
            command_input_limits = [cls.nums2array(lim, dim=6) for lim in command_input_limits]
        if is_output_limits_numeric:
            command_output_limits = [cls.nums2array(lim, dim=6) for lim in command_output_limits]

        config["command_dim"] = OSC_MODE_COMMAND_DIMS[mode]
        for variable_gain, gain_limits, dim in zip(
            (config["variable_kp"], config["variable_damping_ratio"], config["variable_kp_null"]),
            (config["kp_limits"], config["damping_ratio_limits"], config["kp_null_limits"]),
            (6, 6, control_dim),
        ):
            if variable_gain:
                if is_input_limits_numeric:
                    command_input_limits = [
                        cb.cat([lim, cls.nums2array(nums=val, dim=dim)])
                        for lim, val in zip(command_input_limits, (-1, 1))
                    ]
                if is_output_limits_numeric:
                    command_output_limits = [
                        cb.cat([lim, cls.nums2array(nums=val, dim=dim)])
                        for lim, val in zip(command_output_limits, gain_limits)
                    ]
                config["command_dim"] += dim

        config["decouple_pos_ori"] = config.get("decouple_pos_ori", False)
        config["workspace_pose_limiter"] = config.get("workspace_pose_limiter", None)
        reset_joint_pos = config.get("reset_joint_pos")
        config["reset_joint_pos"] = reset_joint_pos[config["dof_idx"]]

        config["command_input_limits"] = command_input_limits
        config["command_output_limits"] = command_output_limits

    @classmethod
    def _process_config_dd(cls, config):
        config["wheel_radius"] = config["wheel_radius"]
        config["wheel_axle_halflength"] = config["wheel_axle_length"] / 2.0

        command_output_limits = config.get("command_output_limits", "default")
        if type(command_output_limits) is str and command_output_limits == "default":
            control_limits = config["control_limits"]
            dof_idx = config["dof_idx"]
            min_vels = control_limits["velocity"][0][dof_idx]
            assert min_vels[0] == min_vels[1], "Differential drive requires both wheel joints to have same min velocities!"
            max_vels = control_limits["velocity"][1][dof_idx]
            assert max_vels[0] == max_vels[1], "Differential drive requires both wheel joints to have same max velocities!"
            assert abs(min_vels[0]) == abs(max_vels[0]), "Differential drive requires both wheel joints to have same min and max absolute velocities!"
            max_lin_vel = max_vels[0] * config["wheel_radius"]
            max_ang_vel = max_lin_vel * 2.0 / config["wheel_axle_halflength"]
            config["command_output_limits"] = ((-max_lin_vel, -max_ang_vel), (max_lin_vel, max_ang_vel))

    @classmethod
    def _process_config_gripper(cls, config):
        assert_valid_key(key=config["motor_type"].lower(), valid_keys=ControlType.VALID_TYPES_STR, name="motor_type")
        config["motor_type"] = config["motor_type"].lower()
        assert_valid_key(key=config.get("mode", "binary"), valid_keys=GRIPPER_MODES, name="mode for multi finger gripper")
        config["mode"] = config.get("mode", "binary")
        config["inverted"] = config.get("inverted", False)
        config["limit_tolerance"] = config.get("limit_tolerance", 0.001)
        config["open_qpos"] = None if config.get("open_qpos", None) is None else cb.array(config.get("open_qpos"))
        config["closed_qpos"] = None if config.get("closed_qpos", None) is None else cb.array(config.get("closed_qpos"))

        if config["mode"] == "binary":
            config["command_output_limits"] = "default"

    @classmethod
    def _process_config_base(cls, controller_id: str, config):
        config["dof_idx"] = cb.as_int(config["dof_idx"])
        config["command_input_limits"] = config.get("command_input_limits", "default")
        config["command_output_limits"] = config.get("command_output_limits", "default")

        had_existing_config = controller_id in cls._configs
        if not had_existing_config:
            cls._configs[controller_id] = config

        control_limits = {}
        for motor_type in {"position", "velocity", "effort"}:
            if motor_type not in config["control_limits"]:
                continue
            control_limits[ControlType.get_type(motor_type)] = [
                config["control_limits"][motor_type][0],
                config["control_limits"][motor_type][1],
            ]
        assert "has_limit" in config["control_limits"], "Expected has_limit specified in control_limits, but does not exist."
        control_limits["has_limit"] = config["control_limits"]["has_limit"]
        config["control_limits"] = control_limits
        config["dof_has_limits"] = control_limits["has_limit"]

        config["goal_shapes"] = cls._get_goal_shapes(controller_id)
        config["goal_dim"] = int(sum(cb.prod(cb.array(shape)) for shape in config["goal_shapes"].values()))

        cls._controls[controller_id] = None
        cls._goals[controller_id] = None
        config["command_scale_factor"] = None
        config["command_output_transform"] = None
        config["command_input_transform"] = None

        command_input_limits = config["command_input_limits"]
        command_output_limits = config["command_output_limits"]
        if type(command_input_limits) is str and command_input_limits == "default":
            command_input_limits = cls._generate_default_command_input_limits()
        if type(command_output_limits) is str and command_output_limits == "default":
            command_output_limits = cls._generate_default_command_output_limits(controller_id)

        command_dim = cls.command_dim(controller_id)
        config["command_input_limits"] = (
            None
            if command_input_limits is None
            else (
                cls.nums2array(command_input_limits[0], command_dim),
                cls.nums2array(command_input_limits[1], command_dim),
            )
        )
        config["command_output_limits"] = (
            None
            if command_output_limits is None
            else (
                cls.nums2array(command_output_limits[0], command_dim),
                cls.nums2array(command_output_limits[1], command_dim),
            )
        )

        isaac_kp = config.get("isaac_kp", None)
        isaac_kd = config.get("isaac_kd", None)
        ct = cls.control_type(controller_id)
        if ct == ControlType.POSITION:
            isaac_kp = m.DEFAULT_ISAAC_KP if isaac_kp is None else isaac_kp
            isaac_kd = m.DEFAULT_ISAAC_KD if isaac_kd is None else isaac_kd
        elif ct == ControlType.VELOCITY:
            assert isaac_kp is None, f"Control type for controller {controller_id} is VELOCITY, so no isaac_kp should be set!"
            isaac_kd = m.DEFAULT_ISAAC_KP if isaac_kd is None else isaac_kd
        elif ct == ControlType.EFFORT:
            assert isaac_kp is None, f"Control type for controller {controller_id} is EFFORT, so no isaac_kp should be set!"
            assert isaac_kd is None, f"Control type for controller {controller_id} is EFFORT, so no isaac_kd should be set!"
        else:
            raise ValueError(f"Expected control type to be one of: [POSITION, VELOCITY, EFFORT], but got: {ct}")

        control_dim = cls.control_dim(controller_id)
        config["isaac_kp"] = None if isaac_kp is None else cls.nums2array(isaac_kp, control_dim)
        config["isaac_kd"] = None if isaac_kd is None else cls.nums2array(isaac_kd, control_dim)

        if not had_existing_config:
            cls._configs.pop(controller_id, None)
        
        return config

    # -------------------------------------------------------------------------
    # State initialization
    # -------------------------------------------------------------------------

    @classmethod
    def _init_state(cls, controller_id: str):
        ctype = cls._types[controller_id]
        if ctype == ControllerType.InverseKinematicsController:
            config = cls._configs[controller_id]
            cls._state[controller_id]["fixed_quat_target"] = None
            cls._state[controller_id]["control_filter"] = (
                None if config.get("smoothing_filter_size", None) in {None, 0}
                else MovingAverageFilter(obs_dim=len(cls.dof_idx(controller_id)), filter_width=config["smoothing_filter_size"])
            )
        elif ctype == ControllerType.OperationalSpaceController:
            cls._state[controller_id]["fixed_quat_target"] = None
        elif ctype == ControllerType.MultiFingerGripperController:
            cls._state[controller_id]["is_grasping"] = IsGraspingState.FALSE
            cls._state[controller_id]["vel_filter"] = MovingAverageFilter(obs_dim=len(cls.dof_idx(controller_id)), filter_width=5)
        elif ctype == ControllerType.NullJointController:
            config = cls._configs[controller_id]
            default_goal = config.get("default_goal", None)
            if default_goal is None:
                default_goal = cb.zeros(len(config["dof_idx"]))
            cls._state[controller_id]["default_goal"] = cb.array(default_goal)

    # -------------------------------------------------------------------------
    # Default limits
    # -------------------------------------------------------------------------

    @classmethod
    def _generate_default_command_input_limits(cls):
        return (-1.0, 1.0)

    @classmethod
    def _generate_default_command_output_limits(cls, controller_id: str):
        config = cls._configs[controller_id]
        ctype = cls._types[controller_id]

        if ctype in _JOINT_TYPES:
            motor_type = config["motor_type"]
            return (
                config["control_limits"][ControlType.get_type(motor_type)][0][config["dof_idx"]],
                config["control_limits"][ControlType.get_type(motor_type)][1][config["dof_idx"]],
            )
        elif ctype == ControllerType.MultiFingerGripperController:
            base_limits = (
                config["control_limits"][cls.control_type(controller_id)][0][config["dof_idx"]],
                config["control_limits"][cls.control_type(controller_id)][1][config["dof_idx"]],
            )
            if config["mode"] == "binary":
                return (-1.0, 1.0)
            elif config["mode"] == "smooth":
                return (cb.mean(base_limits[0]), cb.mean(base_limits[1]))
            elif config["mode"] == "independent":
                return base_limits
            else:
                raise ValueError(f"Invalid mode {config['mode']}")
        else:
            return (
                config["control_limits"][cls.control_type(controller_id)][0][config["dof_idx"]],
                config["control_limits"][cls.control_type(controller_id)][1][config["dof_idx"]],
            )

    # -------------------------------------------------------------------------
    # Goal management
    # -------------------------------------------------------------------------

    @classmethod
    def set_goals(cls, controller_id: str, goals):
        cls._goals[controller_id] = goals

    @classmethod
    def update_goal(cls, controller_id: str, command, control_dict):
        assert (
            len(command) == cls.command_dim(controller_id)
        ), f"Commands must be dimension {cls.command_dim(controller_id)}, got dim {len(command)} instead."
        cls._goals[controller_id] = cls._update_goal(controller_id, cls._preprocess_command(controller_id, command), control_dict)

    @classmethod
    def _update_goal(cls, controller_id, command, control_dict):
        ctype = cls._types[controller_id]
        if ctype in (ControllerType.JointController, ControllerType.NullJointController):
            return cls._update_goal_joint(controller_id, command, control_dict)
        elif ctype == ControllerType.HolonomicBaseJointController:
            return cls._update_goal_holonomic(controller_id, command, control_dict)
        elif ctype == ControllerType.InverseKinematicsController:
            return cls._update_goal_ik(controller_id, command, control_dict)
        elif ctype == ControllerType.OperationalSpaceController:
            return cls._update_goal_osc(controller_id, command, control_dict)
        elif ctype == ControllerType.DifferentialDriveController:
            return dict(vel=command)
        elif ctype == ControllerType.MultiFingerGripperController:
            return dict(target=command)
        raise ValueError(f"Unknown controller type: {ctype}")

    @classmethod
    def _update_goal_joint(cls, controller_id, command, control_dict):
        config = cls._configs[controller_id]
        if config["use_delta_commands"]:
            base_value = control_dict[f"joint_{config['motor_type']}"][cls.dof_idx(controller_id)]
            target = base_value + command
            for rx_ind, ry_ind, rz_ind in config["compute_delta_in_quat_space"]:
                start_rots = base_value[[rx_ind, ry_ind, rz_ind]]
                delta_rots = command[[rx_ind, ry_ind, rz_ind]]
                _, end_quat = cb.T.pose_transform(
                    cb.zeros(3), cb.T.euler2quat(delta_rots), cb.zeros(3), cb.T.euler2quat(start_rots)
                )
                end_rots = cb.T.quat2euler(end_quat)
                target[[rx_ind, ry_ind, rz_ind]] = end_rots
        else:
            target = command

        target = target.clip(
            config["control_limits"][ControlType.get_type(config["motor_type"])][0][cls.dof_idx(controller_id)],
            config["control_limits"][ControlType.get_type(config["motor_type"])][1][cls.dof_idx(controller_id)],
        )
        return dict(target=target)

    @classmethod
    def _update_goal_holonomic(cls, controller_id, command, control_dict):
        base_pose = cb.T.pose2mat((control_dict["root_pos"], control_dict["root_quat"]))
        canonical_pose = cb.T.pose2mat((control_dict["canonical_pos"], control_dict["canonical_quat"]))
        canonical_to_base_pose = cb.T.pose_inv(canonical_pose) @ base_pose

        if cls._configs[controller_id]["motor_type"] == "position":
            command_in_base_frame = cb.as_float32(cb.eye(4))
            command_in_base_frame[:2, 3] = command[:2]
            command_in_canonical_frame = canonical_to_base_pose @ command_in_base_frame
            position = command_in_canonical_frame[:2, 3]
            rz_joint_pos = control_dict["joint_position"][cls.dof_idx(controller_id)][2:3]
            delta_joint_pos = wrap_angle(command[2])
            new_joint_pos = rz_joint_pos + delta_joint_pos
            command = cb.cat([position, new_joint_pos])
        else:
            command_in_base_frame = cb.as_float32(cb.eye(4))
            command_in_base_frame[:2, 3] = command[:2]
            canonical_to_base_pose_rotation = cb.as_float32(cb.eye(4))
            canonical_to_base_pose_rotation[:3, :3] = canonical_to_base_pose[:3, :3]
            command_in_canonical_frame = canonical_to_base_pose_rotation @ command_in_base_frame
            linear_velocity = command_in_canonical_frame[:2, 3]
            angular_velocity = command[2:3]
            command = cb.cat([linear_velocity, angular_velocity])

        return cls._update_goal_joint(controller_id, command=command, control_dict=control_dict)

    @classmethod
    def _update_goal_ik(cls, controller_id, command, control_dict):
        config = cls._configs[controller_id]
        pos_relative = control_dict[f"{config['task_name']}_pos_relative"]
        quat_relative = control_dict[f"{config['task_name']}_quat_relative"]

        if config["mode"] == "absolute_pose":
            target_pos = command[:3]
        else:
            dpos = command[:3]
            target_pos = pos_relative + dpos

        if config["mode"] == "position_fixed_ori":
            if cls._state[controller_id]["fixed_quat_target"] is None:
                cls._state[controller_id]["fixed_quat_target"] = (
                    quat_relative if (cls._goals[controller_id] is None) else cls._goals[controller_id]["target_quat"]
                )
            target_quat = cls._state[controller_id]["fixed_quat_target"]
        elif config["mode"] == "position_compliant_ori":
            target_quat = quat_relative
        elif config["mode"] in ("pose_absolute_ori", "absolute_pose"):
            target_quat = cb.T.axisangle2quat(command[3:6])
        else:
            dori = cb.T.quat2mat(cb.T.axisangle2quat(command[3:6]))
            target_quat = cb.T.mat2quat(dori @ cb.T.quat2mat(quat_relative))

        if config.get("workspace_pose_limiter", None) is not None:
            target_pos, target_quat = config["workspace_pose_limiter"](target_pos, target_quat, control_dict)

        return dict(
            target_pos=cb.as_float32(target_pos),
            target_ori_mat=cb.as_float32(cb.T.quat2mat(target_quat)),
        )

    @classmethod
    def _update_goal_osc(cls, controller_id, command, control_dict):
        config = cls._configs[controller_id]
        pos_relative = cb.copy(control_dict[f"{config['task_name']}_pos_relative"])
        quat_relative = cb.copy(control_dict[f"{config['task_name']}_quat_relative"])

        if config["mode"] == "absolute_pose":
            target_pos = command[:3]
        else:
            dpos = command[:3]
            target_pos = pos_relative + dpos

        if config["mode"] == "position_fixed_ori":
            if cls._state[controller_id]["fixed_quat_target"] is None:
                cls._state[controller_id]["fixed_quat_target"] = (
                    quat_relative if (cls._goals[controller_id] is None) else cls._goals[controller_id]["target_quat"]
                )
            target_quat = cls._state[controller_id]["fixed_quat_target"]
        elif config["mode"] == "position_compliant_ori":
            target_quat = quat_relative
        elif config["mode"] in ("pose_absolute_ori", "absolute_pose"):
            target_quat = cb.T.axisangle2quat(command[3:6])
        else:
            dori = cb.T.quat2mat(cb.T.axisangle2quat(command[3:6]))
            target_quat = cb.T.mat2quat(dori @ cb.T.quat2mat(quat_relative))

        if config["workspace_pose_limiter"] is not None:
            target_pos, target_quat = config["workspace_pose_limiter"](target_pos, target_quat, control_dict)

        gains = None
        if gains is not None:
            cls._update_variable_gains(controller_id=controller_id, gains=gains)

        return dict(
            target_pos=cb.as_float32(target_pos),
            target_ori_mat=cb.as_float32(cb.T.quat2mat(target_quat)),
        )

    # -------------------------------------------------------------------------
    # Command preprocessing
    # -------------------------------------------------------------------------

    @classmethod
    def _preprocess_command(cls, controller_id, command):
        ctype = cls._types[controller_id]

        if ctype == ControllerType.NullJointController:
            return cb.array(cls._state[controller_id]["default_goal"])

        if ctype == ControllerType.MultiFingerGripperController:
            config = cls._configs[controller_id]
            if config["mode"] != "independent":
                command = (
                    cb.array([command] * cls.command_dim(controller_id=controller_id))
                    if type(command) in {int, float}
                    else cb.array([command[0]] * cls.command_dim(controller_id=controller_id))
                )
            if config["inverted"]:
                command = config["command_input_limits"][1] - (command - config["command_input_limits"][0])

        return cls._preprocess_command_base(controller_id, command)

    @classmethod
    def _preprocess_command_base(cls, controller_id, command):
        config = cls._configs[controller_id]
        command = cb.array([command]) if type(command) in {int, float} else command
        if config["command_input_limits"] is not None:
            command = command.clip(*config["command_input_limits"])
            if config["command_output_limits"] is not None:
                if config["command_scale_factor"] is None:
                    config["command_scale_factor"] = abs(
                        config["command_output_limits"][1] - config["command_output_limits"][0]
                    ) / abs(config["command_input_limits"][1] - config["command_input_limits"][0])
                    config["command_output_transform"] = (
                        config["command_output_limits"][1] + config["command_output_limits"][0]
                    ) / 2.0
                    config["command_input_transform"] = (
                        config["command_input_limits"][1] + config["command_input_limits"][0]
                    ) / 2.0
                command = (
                    command - config["command_input_transform"]
                ) * config["command_scale_factor"] + config["command_output_transform"]
        return command
    
    @classmethod
    def _reverse_preprocess_command(cls, controller_id, processed_command):
        config = cls._configs[controller_id]
        if config["command_input_limits"] is not None and config["command_output_limits"] is not None:
            if config["command_scale_factor"] is None:
                config["command_scale_factor"] = abs(config["command_output_limits"][1] - config["command_output_limits"][0]) / abs(
                    config["command_input_limits"][1] - config["command_input_limits"][0]
                )
                config["command_output_transform"] = (config["command_output_limits"][1] + config["command_output_limits"][0]) / 2.0
                config["command_input_transform"] = (config["command_input_limits"][1] + config["command_input_limits"][0]) / 2.0
            original_command = (
                processed_command - config["command_output_transform"]
            ) / config["command_scale_factor"] + config["command_input_transform"]
        else:
            original_command = processed_command
        return original_command
    
    # -------------------------------------------------------------------------
    # Control computation
    # -------------------------------------------------------------------------

    @classmethod
    def compute_control(cls, controller_id, control_dict, goal_dict=None):
        ctype = cls._types[controller_id]
        if ctype in _JOINT_TYPES:
            if ctype == ControllerType.InverseKinematicsController:
                return cls._compute_control_ik(controller_id, control_dict)
            return cls._compute_control_joint(controller_id, control_dict, goal_dict=goal_dict)
        elif ctype == ControllerType.OperationalSpaceController:
            return cls._compute_control_osc(controller_id, control_dict)
        elif ctype == ControllerType.DifferentialDriveController:
            return cls._compute_control_dd(controller_id, control_dict)
        elif ctype == ControllerType.MultiFingerGripperController:
            return cls._compute_control_gripper(controller_id, control_dict)
        raise ValueError(f"Unknown controller type: {ctype}")

    @classmethod
    def _compute_control_joint(cls, controller_id, control_dict, goal_dict=None):
        config = cls._configs[controller_id]
        base_value = control_dict[f"joint_{config['motor_type']}"][cls.dof_idx(controller_id)]
        target = cls._goals[controller_id]["target"] if goal_dict is None else goal_dict["target"]

        if config["use_impedances"]:
            if config["motor_type"] == "position":
                position_error = target - base_value
                vel_pos_error = -control_dict["joint_velocity"][cls.dof_idx(controller_id)]
                u = position_error * config["pos_kp"] + vel_pos_error * config["pos_kd"]
            elif config["motor_type"] == "velocity":
                velocity_error = target - base_value
                u = velocity_error * config["vel_kp"]
            else:
                u = target

            u = cb.get_custom_method("compute_joint_torques")(u, control_dict["mass_matrix"], cls.dof_idx(controller_id))

            if config["use_gravity_compensation"]:
                u += control_dict["gravity_force"][cls.dof_idx(controller_id)]
            if config["use_cc_compensation"]:
                u += control_dict["cc_force"][cls.dof_idx(controller_id)]
        else:
            u = target

        return u

    @classmethod
    def _compute_control_ik(cls, controller_id, control_dict):
        config = cls._configs[controller_id]
        goal_dict = cls._goals[controller_id]
        q = control_dict["joint_position"][cls.dof_idx(controller_id)]
        j_eef = control_dict[f"{config['task_name']}_jacobian_relative"][:, cls.dof_idx(controller_id)]
        ee_pos = control_dict[f"{config['task_name']}_pos_relative"]
        ee_quat = control_dict[f"{config['task_name']}_quat_relative"]

        target_joint_pos = cb.get_custom_method("compute_ik_qpos")(
            q=q,
            j_eef=j_eef,
            ee_pos=cb.as_float32(ee_pos),
            ee_mat=cb.as_float32(cb.T.quat2mat(ee_quat)),
            goal_pos=goal_dict["target_pos"],
            goal_ori_mat=goal_dict["target_ori_mat"],
            q_lower_limit=config["control_limits"][ControlType.get_type("position")][0][cls.dof_idx(controller_id)],
            q_upper_limit=config["control_limits"][ControlType.get_type("position")][1][cls.dof_idx(controller_id)],
        )

        if cls._state[controller_id]["control_filter"] is not None:
            target_joint_pos = cls._state[controller_id]["control_filter"].estimate(target_joint_pos)

        return cls._compute_control_joint(controller_id=controller_id, control_dict=control_dict, goal_dict=dict(target=target_joint_pos))

    @classmethod
    def _compute_control_osc(cls, controller_id, control_dict):
        goal_dict = cls._goals[controller_id]
        config = cls._configs[controller_id]
        kp = config["kp"]
        damping_ratio = config["damping_ratio"]
        kd = 2 * cb.sqrt(kp) * damping_ratio

        dof_idxs_mat = tuple(cb.meshgrid(cls.dof_idx(controller_id), cls.dof_idx(controller_id)))
        q = control_dict["joint_position"][cls.dof_idx(controller_id)]
        qd = control_dict["joint_velocity"][cls.dof_idx(controller_id)]
        mm = control_dict["mass_matrix"][dof_idxs_mat]
        j_eef = control_dict[f"{config['task_name']}_jacobian_relative"][:, cls.dof_idx(controller_id)]
        ee_pos = control_dict[f"{config['task_name']}_pos_relative"]
        ee_quat = control_dict[f"{config['task_name']}_quat_relative"]
        ee_vel = cb.cat([
            control_dict[f"{config['task_name']}_lin_vel_relative"],
            control_dict[f"{config['task_name']}_ang_vel_relative"],
        ])
        base_lin_vel = control_dict["root_rel_lin_vel"]
        base_ang_vel = control_dict["root_rel_ang_vel"]

        u = cb.get_custom_method("compute_osc_torques")(
            q=q, qd=qd, mm=mm, j_eef=j_eef,
            ee_pos=cb.as_float32(ee_pos),
            ee_mat=cb.as_float32(cb.T.quat2mat(ee_quat)),
            ee_lin_vel=cb.as_float32(ee_vel[:3]),
            ee_ang_vel_err=cb.as_float32(
                cb.T.quat2axisangle(
                    cb.T.quat_multiply(cb.T.axisangle2quat(-ee_vel[3:]), cb.T.axisangle2quat(base_ang_vel))
                )
            ),
            goal_pos=goal_dict["target_pos"],
            goal_ori_mat=goal_dict["target_ori_mat"],
            kp=kp, kd=kd,
            kp_null=config["kp_null"],
            kd_null=config["kd_null"],
            rest_qpos=config["reset_joint_pos"],
            control_dim=cls.control_dim(controller_id),
            decouple_pos_ori=config["decouple_pos_ori"],
            base_lin_vel=cb.as_float32(base_lin_vel),
            base_ang_vel=cb.as_float32(base_ang_vel),
        ).flatten()

        if config["use_gravity_compensation"]:
            u += control_dict["gravity_force"][cls.dof_idx(controller_id)]
        if config["use_cc_compensation"]:
            u += control_dict["cc_force"][cls.dof_idx(controller_id)]
        return u

    @classmethod
    def _compute_control_dd(cls, controller_id, control_dict):
        goal_dict = cls._goals[controller_id]
        config = cls._configs[controller_id]
        lin_vel, ang_vel = goal_dict["vel"]
        left_wheel_joint_vel = (lin_vel - ang_vel * config["wheel_axle_halflength"]) / config["wheel_radius"]
        right_wheel_joint_vel = (lin_vel + ang_vel * config["wheel_axle_halflength"]) / config["wheel_radius"]
        return cb.array([left_wheel_joint_vel, right_wheel_joint_vel])

    @classmethod
    def _compute_control_gripper(cls, controller_id, control_dict):
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
            u = cb.full((cls.control_dim(controller_id=controller_id),), target[0]) if len(target) == 1 else target

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

    # -------------------------------------------------------------------------
    # Clip, step, reset
    # -------------------------------------------------------------------------

    @classmethod
    def clip_control(cls, controller_id, control):
        config = cls._configs[controller_id]
        clipped_control = control.clip(
            config["control_limits"][cls.control_type(controller_id)][0][config["dof_idx"]],
            config["control_limits"][cls.control_type(controller_id)][1][config["dof_idx"]],
        )
        idx = (
            config["dof_has_limits"][config["dof_idx"]]
            if cls.control_type(controller_id) == ControlType.POSITION
            else [True] * cls.control_dim(controller_id)
        )
        control[idx] = clipped_control[idx]
        return control
    
    @classmethod
    def step(cls, controller_id, control_dict):
        if cls._goals[controller_id] is None:
            cls._goals[controller_id] = cls.compute_no_op_goal(controller_id=controller_id, control_dict=control_dict)
        control = cls.compute_control(controller_id=controller_id, control_dict=control_dict)
        assert (
            len(control) == cls.control_dim(controller_id)
        ), f"Control signal must be of length {cls.control_dim(controller_id)}, got {len(control)} instead."
        cls._controls[controller_id] = cls.clip_control(controller_id, control)
        return cls._controls[controller_id]

    @classmethod
    def reset(cls, controller_id: str):
        cls._goals[controller_id] = None
        ctype = cls._types[controller_id]
        if ctype == ControllerType.InverseKinematicsController:
            if cls._state[controller_id]["control_filter"] is not None:
                cls._state[controller_id]["control_filter"].reset()
            cls._state[controller_id]["fixed_quat_target"] = None
        elif ctype == ControllerType.OperationalSpaceController:
            cls._state[controller_id]["fixed_quat_target"] = None
            cls._clear_variable_gains(controller_id=controller_id)
        elif ctype == ControllerType.MultiFingerGripperController:
            cls._state[controller_id]["vel_filter"].reset()
            cls._state[controller_id]["is_grasping"] = IsGraspingState.FALSE

    # -------------------------------------------------------------------------
    # Batching
    # -------------------------------------------------------------------------

    @classmethod
    def step_batch(cls, controller_ids, controller_type):
        for cid in controller_ids:
            if cls._goals[cid] is None:
                cls._goals[cid] = cls.compute_no_op_goal(cid, cls._control_dicts[cid])

        if controller_type in (ControllerType.JointController, ControllerType.NullJointController, ControllerType.HolonomicBaseJointController):
            return cls._step_batch_joint(controller_ids)
        elif controller_type == ControllerType.InverseKinematicsController:
            return cls._step_batch_ik(controller_ids)
        elif controller_type == ControllerType.OperationalSpaceController:
            return cls._step_batch_osc(controller_ids)
        elif controller_type == ControllerType.DifferentialDriveController:
            return cls._step_batch_dd(controller_ids)
        elif controller_type == ControllerType.MultiFingerGripperController:
            return cls._step_batch_gripper(controller_ids)

        # Fallback: sequential
        results = []
        for cid in controller_ids:
            control = cls.compute_control(controller_id=cid, control_dict=cls._control_dicts[cid])
            control = cls.clip_control(cid, control)
            cls._controls[cid] = control
            results.append(control)
        return results
    
    @classmethod
    def _step_batch_joint(cls, controller_ids):
        impedance_indices = []
        non_impedance_indices = []
        for idx, cid in enumerate(controller_ids):
            if cls._configs[cid]["use_impedances"]:
                impedance_indices.append(idx)
            else:
                non_impedance_indices.append(idx)

        results = [None] * len(controller_ids)

        for idx in non_impedance_indices:
            cid = controller_ids[idx]
            u = cb.copy(cls._goals[cid]["target"])
            u = cls.clip_control(cid, u)
            cls._controls[cid] = u
            results[idx] = u

        if impedance_indices:
            imp_cids = [controller_ids[idx] for idx in impedance_indices]
            N = len(imp_cids)
            dims = [cls.control_dim(cid) for cid in imp_cids]
            max_dim = max(dims)

            targets = cb.zeros((N, max_dim))
            base_values = cb.zeros((N, max_dim))
            velocities = cb.zeros((N, max_dim))
            gain = cb.zeros((N, max_dim))
            damping = cb.zeros((N, max_dim))
            gravity = cb.zeros((N, max_dim))
            cc = cb.zeros((N, max_dim))
            mass_matrices = cb.zeros((N, max_dim, max_dim))
            is_effort = [False] * N

            for i, cid in enumerate(imp_cids):
                config = cls._configs[cid]
                d = dims[i]
                dof_idx = cls.dof_idx(cid)
                cd = cls._control_dicts[cid]

                targets[i, :d] = cls._goals[cid]["target"]
                base_values[i, :d] = cd[f"joint_{config['motor_type']}"][dof_idx]
                velocities[i, :d] = cd["joint_velocity"][dof_idx]

                if config["motor_type"] == "position":
                    gain[i, :d] = config["pos_kp"]
                    damping[i, :d] = config["pos_kd"]
                elif config["motor_type"] == "velocity":
                    gain[i, :d] = config["vel_kp"]
                else:
                    is_effort[i] = True

                mass_matrices[i, :d, :d] = cd["mass_matrix"][dof_idx][:, dof_idx]
                if config["use_gravity_compensation"]:
                    gravity[i, :d] = cd["gravity_force"][dof_idx]
                if config["use_cc_compensation"]:
                    cc[i, :d] = cd["cc_force"][dof_idx]

            u = (targets - base_values) * gain + (-velocities) * damping
            for i in range(N):
                if is_effort[i]:
                    u[i] = targets[i]

            u = (mass_matrices @ u[..., None])[..., 0]
            u = u + gravity + cc

            for i, (cid, idx) in enumerate(zip(imp_cids, impedance_indices)):
                d = dims[i]
                control = u[i, :d]
                control = cls.clip_control(cid, control)
                cls._controls[cid] = control
                results[idx] = control

        return results

    @classmethod
    def _step_batch_ik(cls, controller_ids):
        N = len(controller_ids)
        dims = [cls.control_dim(cid) for cid in controller_ids]
        max_dim = max(dims)

        q = cb.zeros((N, max_dim))
        j_eef = cb.zeros((N, 6, max_dim))
        ee_pos = cb.zeros((N, 3))
        ee_mat = cb.zeros((N, 3, 3))
        goal_pos = cb.zeros((N, 3))
        goal_ori_mat = cb.zeros((N, 3, 3))
        q_lower = cb.zeros((N, max_dim))
        q_upper = cb.zeros((N, max_dim))

        for i, cid in enumerate(controller_ids):
            config = cls._configs[cid]
            d = dims[i]
            dof_idx = cls.dof_idx(cid)
            cd = cls._control_dicts[cid]

            q[i, :d] = cd["joint_position"][dof_idx]
            j_eef[i, :, :d] = cd[f"{config['task_name']}_jacobian_relative"][:, dof_idx]
            ee_pos[i] = cd[f"{config['task_name']}_pos_relative"]
            ee_quat = cd[f"{config['task_name']}_quat_relative"]
            ee_mat[i] = cb.as_float32(cb.T.quat2mat(ee_quat))
            goal_pos[i] = cls._goals[cid]["target_pos"]
            goal_ori_mat[i] = cls._goals[cid]["target_ori_mat"]
            q_lower[i, :d] = config["control_limits"][ControlType.get_type("position")][0][dof_idx]
            q_upper[i, :d] = config["control_limits"][ControlType.get_type("position")][1][dof_idx]

        target_batch = cb.get_custom_method("compute_ik_qpos_batch")(
            q=q, j_eef=j_eef,
            ee_pos=cb.as_float32(ee_pos), ee_mat=cb.as_float32(ee_mat),
            goal_pos=cb.as_float32(goal_pos), goal_ori_mat=cb.as_float32(goal_ori_mat),
            q_lower_limit=q_lower, q_upper_limit=q_upper,
        )

        results = []
        for i, cid in enumerate(controller_ids):
            d = dims[i]
            target_joint_pos = target_batch[i, :d]

            if cls._state[cid]["control_filter"] is not None:
                target_joint_pos = cls._state[cid]["control_filter"].estimate(target_joint_pos)

            config = cls._configs[cid]
            if config["use_impedances"]:
                cd = cls._control_dicts[cid]
                dof_idx = cls.dof_idx(cid)
                base_value = cd[f"joint_{config['motor_type']}"][dof_idx]
                if config["motor_type"] == "position":
                    u = (target_joint_pos - base_value) * config["pos_kp"] + (-cd["joint_velocity"][dof_idx]) * config["pos_kd"]
                elif config["motor_type"] == "velocity":
                    u = (target_joint_pos - base_value) * config["vel_kp"]
                else:
                    u = target_joint_pos
                u = cb.get_custom_method("compute_joint_torques")(u, cd["mass_matrix"], dof_idx)
                if config["use_gravity_compensation"]:
                    u += cd["gravity_force"][dof_idx]
                if config["use_cc_compensation"]:
                    u += cd["cc_force"][dof_idx]
            else:
                u = target_joint_pos

            u = cls.clip_control(cid, u)
            cls._controls[cid] = u
            results.append(u)

        return results

    @classmethod
    def _step_batch_osc(cls, controller_ids):
        N = len(controller_ids)
        dims = [cls.control_dim(cid) for cid in controller_ids]
        max_dim = max(dims)

        q = cb.zeros((N, max_dim))
        qd = cb.zeros((N, max_dim))
        mm = cb.zeros((N, max_dim, max_dim))
        for i in range(N):
            for j in range(max_dim):
                mm[i, j, j] = 1.0
        j_eef_batch = cb.zeros((N, 6, max_dim))
        ee_pos_batch = cb.zeros((N, 3))
        ee_mat_batch = cb.zeros((N, 3, 3))
        ee_lin_vel_batch = cb.zeros((N, 3))
        ee_ang_vel_err_batch = cb.zeros((N, 3))
        goal_pos_batch = cb.zeros((N, 3))
        goal_ori_mat_batch = cb.zeros((N, 3, 3))
        kp_batch = cb.zeros((N, 6))
        kd_batch = cb.zeros((N, 6))
        kp_null_batch = cb.zeros((N, max_dim))
        kd_null_batch = cb.zeros((N, max_dim))
        rest_qpos_batch = cb.zeros((N, max_dim))
        base_lin_vel_batch = cb.zeros((N, 3))
        base_ang_vel_batch = cb.zeros((N, 3))
        gravity = cb.zeros((N, max_dim))
        cc_force = cb.zeros((N, max_dim))
        decouple_flags = []

        for i, cid in enumerate(controller_ids):
            config = cls._configs[cid]
            d = dims[i]
            dof_idx = cls.dof_idx(cid)
            cd = cls._control_dicts[cid]
            goal_dict = cls._goals[cid]

            q[i, :d] = cd["joint_position"][dof_idx]
            qd[i, :d] = cd["joint_velocity"][dof_idx]
            mm[i, :d, :d] = cd["mass_matrix"][dof_idx][:, dof_idx]
            j_eef_batch[i, :, :d] = cd[f"{config['task_name']}_jacobian_relative"][:, dof_idx]
            ee_pos_batch[i] = cd[f"{config['task_name']}_pos_relative"]
            ee_quat = cd[f"{config['task_name']}_quat_relative"]
            ee_mat_batch[i] = cb.as_float32(cb.T.quat2mat(ee_quat))

            ee_lin_vel_batch[i] = cb.as_float32(cd[f"{config['task_name']}_lin_vel_relative"])
            ee_ang_vel = cd[f"{config['task_name']}_ang_vel_relative"]
            base_ang_vel = cd["root_rel_ang_vel"]
            ee_ang_vel_err_batch[i] = cb.as_float32(
                cb.T.quat2axisangle(
                    cb.T.quat_multiply(cb.T.axisangle2quat(-ee_ang_vel), cb.T.axisangle2quat(base_ang_vel))
                )
            )

            goal_pos_batch[i] = goal_dict["target_pos"]
            goal_ori_mat_batch[i] = goal_dict["target_ori_mat"]

            kp = config["kp"]
            kd_val = 2 * cb.sqrt(kp) * config["damping_ratio"]
            kp_batch[i] = kp
            kd_batch[i] = kd_val
            kp_null_batch[i, :d] = config["kp_null"]
            kd_null_batch[i, :d] = config["kd_null"]
            rest_qpos_batch[i, :d] = config["reset_joint_pos"]

            base_lin_vel_batch[i] = cb.as_float32(cd["root_rel_lin_vel"])
            base_ang_vel_batch[i] = cb.as_float32(base_ang_vel)

            decouple_flags.append(config["decouple_pos_ori"])

            if config["use_gravity_compensation"]:
                gravity[i, :d] = cd["gravity_force"][dof_idx]
            if config["use_cc_compensation"]:
                cc_force[i, :d] = cd["cc_force"][dof_idx]

        all_decouple = all(decouple_flags)
        no_decouple = not any(decouple_flags)

        if no_decouple or all_decouple:
            u = cb.get_custom_method("compute_osc_torques_batch")(
                q=q, qd=qd, mm=mm, j_eef=j_eef_batch,
                ee_pos=ee_pos_batch, ee_mat=ee_mat_batch,
                ee_lin_vel=ee_lin_vel_batch, ee_ang_vel_err=ee_ang_vel_err_batch,
                goal_pos=goal_pos_batch, goal_ori_mat=goal_ori_mat_batch,
                kp=kp_batch, kd=kd_batch,
                kp_null=kp_null_batch, kd_null=kd_null_batch,
                rest_qpos=rest_qpos_batch, max_dim=max_dim,
                decouple_pos_ori=all_decouple,
                base_lin_vel=base_lin_vel_batch, base_ang_vel=base_ang_vel_batch,
            )
        else:
            u = cb.zeros((N, max_dim))
            for flag_val in [False, True]:
                indices = [i for i in range(N) if decouple_flags[i] == flag_val]
                if not indices:
                    continue
                idx_arr = cb.as_int(cb.array(indices))
                u_group = cb.get_custom_method("compute_osc_torques_batch")(
                    q=q[idx_arr], qd=qd[idx_arr], mm=mm[idx_arr], j_eef=j_eef_batch[idx_arr],
                    ee_pos=ee_pos_batch[idx_arr], ee_mat=ee_mat_batch[idx_arr],
                    ee_lin_vel=ee_lin_vel_batch[idx_arr], ee_ang_vel_err=ee_ang_vel_err_batch[idx_arr],
                    goal_pos=goal_pos_batch[idx_arr], goal_ori_mat=goal_ori_mat_batch[idx_arr],
                    kp=kp_batch[idx_arr], kd=kd_batch[idx_arr],
                    kp_null=kp_null_batch[idx_arr], kd_null=kd_null_batch[idx_arr],
                    rest_qpos=rest_qpos_batch[idx_arr], max_dim=max_dim,
                    decouple_pos_ori=flag_val,
                    base_lin_vel=base_lin_vel_batch[idx_arr], base_ang_vel=base_ang_vel_batch[idx_arr],
                )
                for j_idx, orig_idx in enumerate(indices):
                    u[orig_idx] = u_group[j_idx]

        u = u + gravity + cc_force

        results = []
        for i, cid in enumerate(controller_ids):
            d = dims[i]
            control = u[i, :d]
            control = cls.clip_control(cid, control)
            cls._controls[cid] = control
            results.append(control)

        return results

    @classmethod
    def _step_batch_dd(cls, controller_ids):
        N = len(controller_ids)

        vels = cb.zeros((N, 2))
        wheel_radius = cb.zeros(N)
        half_axle = cb.zeros(N)
        for i, cid in enumerate(controller_ids):
            vels[i] = cls._goals[cid]["vel"]
            config = cls._configs[cid]
            wheel_radius[i] = config["wheel_radius"]
            half_axle[i] = config["wheel_axle_halflength"]

        lin_vel = vels[:, 0]
        ang_vel = vels[:, 1]

        left = (lin_vel - ang_vel * half_axle) / wheel_radius
        right = (lin_vel + ang_vel * half_axle) / wheel_radius

        u_batch = cb.zeros((N, 2))
        u_batch[:, 0] = left
        u_batch[:, 1] = right

        results = []
        for i, cid in enumerate(controller_ids):
            control = u_batch[i]
            control = cls.clip_control(cid, control)
            cls._controls[cid] = control
            results.append(control)

        return results

    @classmethod
    def _step_batch_gripper(cls, controller_ids):
        results = []
        for cid in controller_ids:
            control = cls.compute_control(controller_id=cid, control_dict=cls._control_dicts[cid])
            control = cls.clip_control(cid, control)
            cls._controls[cid] = control
            results.append(control)
        return results

    # -------------------------------------------------------------------------
    # Step management
    # -------------------------------------------------------------------------

    @classmethod
    def begin_controller_step(cls):
        Controller._ROBOT_CONTROL_STEP_CACHE.clear()

    @classmethod
    def step_controller_class(cls):
        cls._control_dicts.clear()

        active_ids = []
        for controller_id in list(cls._configs.keys()):
            cls._goals.setdefault(controller_id, None)
            cls._controls.setdefault(controller_id, None)
            robot = cls._robots.get(controller_id, None)
            if robot is None:
                continue
            if not robot.control_enabled:
                continue
            if robot._articulation_view_direct is None or not robot._articulation_view_direct.initialized:
                continue

            robot_name = robot.name
            cache = Controller._ROBOT_CONTROL_STEP_CACHE.get(robot_name, None)
            if cache is None:
                control_dict = robot.get_control_dict()
                u_vec = cb.zeros(robot.n_dof)
                u_type_vec = cb.array([ControlType.EFFORT] * robot.n_dof)
                cache = {
                    "robot": robot,
                    "control_dict": control_dict,
                    "u_vec": u_vec,
                    "u_type_vec": u_type_vec,
                }
                Controller._ROBOT_CONTROL_STEP_CACHE[robot_name] = cache

            cls._control_dicts[controller_id] = cache["control_dict"]
            active_ids.append(controller_id)

        if not active_ids:
            return

        # Group by controller type
        type_groups = {}
        for cid in active_ids:
            ctype = cls._types[cid]
            type_groups.setdefault(ctype, []).append(cid)

        # Batch compute per type
        all_results = {}
        for ctype, cids in type_groups.items():
            controls = cls.step_batch(cids, ctype)
            for cid, control in zip(cids, controls):
                all_results[cid] = control

        # Scatter
        for controller_id in active_ids:
            control = all_results[controller_id]
            robot_name = cls._robots[controller_id].name
            cache = Controller._ROBOT_CONTROL_STEP_CACHE[robot_name]
            idx = cls.dof_idx(controller_id)
            cache["u_vec"][idx] = control
            cache["u_type_vec"][idx] = cls.control_type(controller_id)

    @classmethod
    def deploy_controller_step(cls):
        for cache in Controller._ROBOT_CONTROL_STEP_CACHE.values():
            robot = cache["robot"]
            control, control_type = robot._postprocess_control(
                control=cache["u_vec"], control_type=cache["u_type_vec"]
            )
            robot.deploy_control(control=control, control_type=control_type)
        Controller._ROBOT_CONTROL_STEP_CACHE.clear()

    @classmethod
    def apply_action(cls, controller_id: str, action):
        robot = cls._robots[controller_id]
        cls.update_goal(
            controller_id=controller_id,
            command=action,
            control_dict=robot.get_control_dict(),
        )

    # -------------------------------------------------------------------------
    # No-op goals / commands
    # -------------------------------------------------------------------------

    @classmethod
    def compute_no_op_goal(cls, controller_id: str, control_dict):
        ctype = cls._types[controller_id]
        config = cls._configs[controller_id]

        if ctype in (ControllerType.JointController, ControllerType.HolonomicBaseJointController):
            if config["motor_type"] == "position":
                target = control_dict[f"joint_{config['motor_type']}"][cls.dof_idx(controller_id)]
            else:
                target = cb.zeros(cls.control_dim(controller_id))
            return dict(target=target)
        elif ctype == ControllerType.NullJointController:
            return dict(target=cls._state[controller_id]["default_goal"])
        elif ctype == ControllerType.InverseKinematicsController:
            return dict(
                target_pos=cb.as_float32(control_dict[f"{config['task_name']}_pos_relative"]),
                target_ori_mat=cb.as_float32(cb.T.quat2mat(control_dict[f"{config['task_name']}_quat_relative"])),
            )
        elif ctype == ControllerType.OperationalSpaceController:
            target_pos = cb.copy(control_dict[f"{config['task_name']}_pos_relative"])
            target_quat = cb.copy(control_dict[f"{config['task_name']}_quat_relative"])
            return dict(
                target_pos=cb.as_float32(target_pos),
                target_ori_mat=cb.as_float32(cb.T.quat2mat(target_quat)),
            )
        elif ctype == ControllerType.DifferentialDriveController:
            return dict(vel=cb.zeros(2))
        elif ctype == ControllerType.MultiFingerGripperController:
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
                if config["mode"] == "smooth":
                    target = cb.mean(target, dim=-1, keepdim=True)
            return dict(target=target)
        raise ValueError(f"Unknown controller type: {ctype}")

    @classmethod
    def compute_no_op_action(cls, controller_id: str, control_dict):
        if cls._goals[controller_id] is None:
            cls._goals[controller_id] = cls.compute_no_op_goal(controller_id=controller_id, control_dict=control_dict)
        command = cls._compute_no_op_command(controller_id=controller_id, control_dict=control_dict)
        return cb.to_torch(cls._reverse_preprocess_command(controller_id=controller_id, processed_command=command))

    @classmethod
    def _compute_no_op_command(cls, controller_id: str, control_dict):
        ctype = cls._types[controller_id]
        config = cls._configs[controller_id]

        if ctype == ControllerType.NullJointController:
            return cb.array([])
        elif ctype in (ControllerType.JointController,):
            if config["motor_type"] == "position":
                if config["use_delta_commands"]:
                    return cb.zeros(cls.command_dim(controller_id))
                return control_dict["joint_position"][cls.dof_idx(controller_id)]
            if config["motor_type"] == "velocity":
                if config["use_delta_commands"]:
                    return -control_dict["joint_velocity"][cls.dof_idx(controller_id)]
                return cb.zeros(cls.command_dim(controller_id))
            raise ValueError("Cannot compute noop action for effort motor type.")
        elif ctype == ControllerType.HolonomicBaseJointController:
            return cb.zeros(cls.command_dim(controller_id))
        elif ctype == ControllerType.InverseKinematicsController:
            pos_relative = control_dict[f"{config['task_name']}_pos_relative"]
            quat_relative = control_dict[f"{config['task_name']}_quat_relative"]
            command = cb.zeros(6)
            mode = config["mode"]
            if mode == "absolute_pose":
                command[:3] = pos_relative
            if mode in ("pose_absolute_ori", "absolute_pose"):
                command[3:] = cb.T.quat2axisangle(quat_relative)
            return command
        elif ctype == ControllerType.OperationalSpaceController:
            pos_relative = control_dict[f"{config['task_name']}_pos_relative"]
            quat_relative = control_dict[f"{config['task_name']}_quat_relative"]
            command = cb.zeros(6)
            if config["mode"] == "absolute_pose":
                command[:3] = pos_relative
            if config["mode"] in ("pose_absolute_ori", "absolute_pose"):
                command[3:] = cb.T.quat2axisangle(quat_relative)
            return command
        elif ctype == ControllerType.DifferentialDriveController:
            return cb.zeros(2)
        elif ctype == ControllerType.MultiFingerGripperController:
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
            if config["mode"] == "smooth":
                command = cb.mean(command, dim=-1, keepdim=True)
            return command
        raise ValueError(f"Unknown controller type: {ctype}")

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    @classmethod
    def _dump_state(cls, controller_id: str):
        goal = cls._goals[controller_id]
        state = dict(
            goal_is_valid=goal is not None,
            goal=None if goal is None else {k: cb.to_torch(v) for k, v in goal.items()},
        )
        ctype = cls._types[controller_id]
        if ctype == ControllerType.InverseKinematicsController:
            if cls._state[controller_id]["control_filter"] is not None:
                state["control_filter"] = cls._state[controller_id]["control_filter"].dump_state(serialized=False)
        elif ctype == ControllerType.MultiFingerGripperController:
            state["vel_filter"] = cls._state[controller_id]["vel_filter"].dump_state(serialized=False)
        return state

    @classmethod
    def _load_state(cls, controller_id: str, state):
        if state["goal"] is None:
            cls._goals[controller_id] = None
        else:
            goal = dict()
            for name, goal_state in state["goal"].items():
                if isinstance(goal_state, th.Tensor):
                    goal[name] = cb.from_torch(goal_state)
                else:
                    goal[name] = goal_state
            cls._goals[controller_id] = goal
    
        ctype = cls._types[controller_id]
        if ctype == ControllerType.InverseKinematicsController:
            if cls._goals[controller_id] is not None:
                if cls._configs[controller_id]["mode"] == "position_fixed_ori":
                    cls._state[controller_id]["fixed_quat_target"] = cls._goals[controller_id]["target_quat"]
                if cls._state[controller_id]["control_filter"] is not None:
                    cls._state[controller_id]["control_filter"].load_state(state["control_filter"], serialized=False)
        elif ctype == ControllerType.OperationalSpaceController:
            if cls._goals[controller_id] is not None and cls._configs[controller_id]["mode"] == "position_fixed_ori":
                cls._state[controller_id]["fixed_quat_target"] = cls._goals[controller_id]["target_quat"]
        elif ctype == ControllerType.MultiFingerGripperController:
            if cls._goals[controller_id] is not None:
                cls._state[controller_id]["vel_filter"].load_state(state["vel_filter"], serialized=False)

    @classmethod
    def dump_state(cls, controller_id: str, serialized=False):
        state = cls._dump_state(controller_id=controller_id)
        return cls.serialize(controller_id=controller_id, state=state) if serialized else state

    @classmethod
    def load_state(cls, controller_id: str, state, serialized=False):
        if serialized:
            orig_state_len = len(state)
            state, deserialized_items = cls.deserialize(controller_id=controller_id, state=state)
            assert deserialized_items == orig_state_len, (
                f"Invalid state deserialization occurred! Expected {orig_state_len} total "
                f"values to be deserialized, only {deserialized_items} were."
            )
        cls._load_state(controller_id=controller_id, state=state)

    @classmethod
    def serialize(cls, controller_id: str, state):
        goal_is_valid = state["goal_is_valid"]
        goal_state_flattened = (
            th.cat([goal_state.flatten() for goal_state in state["goal"].values()])
            if goal_is_valid
            else th.zeros(cls._configs[controller_id]["goal_dim"])
        )
        state_flat = th.cat([th.tensor([goal_is_valid]), goal_state_flattened])

        ctype = cls._types[controller_id]
        if ctype == ControllerType.InverseKinematicsController:
            state_flat = th.cat([
                state_flat,
                (
                    th.tensor([])
                    if cls._state[controller_id]["control_filter"] is None
                    else cls._state[controller_id]["control_filter"].serialize(state=state["control_filter"])
                ),
            ])
        elif ctype == ControllerType.MultiFingerGripperController:
            state_flat = th.cat([state_flat, cls._state[controller_id]["vel_filter"].serialize(state=state["vel_filter"])])
        return state_flat

    @classmethod
    def deserialize(cls, controller_id: str, state):
        goal_is_valid = bool(state[0])
        if goal_is_valid:
            idx = 1
            goal = dict()
            for key, shape in cls._configs[controller_id]["goal_shapes"].items():
                length = math.prod(shape)
                goal[key] = state[idx : idx + length].reshape(shape)
                idx += length
        else:
            goal = None
        state_dict = dict(goal_is_valid=goal_is_valid, goal=goal)
        idx = cls._configs[controller_id]["goal_dim"] + 1

        ctype = cls._types[controller_id]
        if ctype == ControllerType.InverseKinematicsController:
            if cls._state[controller_id]["control_filter"] is not None:
                state_dict["control_filter"], deserialized_items = cls._state[controller_id]["control_filter"].deserialize(
                    state=state[idx:]
                )
                idx += deserialized_items
        elif ctype == ControllerType.MultiFingerGripperController:
            state_dict["vel_filter"], deserialized_items = cls._state[controller_id]["vel_filter"].deserialize(
                state=state[idx:]
            )
            idx += deserialized_items
        return state_dict, idx

    # -------------------------------------------------------------------------
    # Property accessors
    # -------------------------------------------------------------------------

    @classmethod
    def _get_goal_shapes(cls, controller_id: str):
        ctype = cls._types[controller_id]
        if ctype in (ControllerType.JointController, ControllerType.NullJointController, ControllerType.HolonomicBaseJointController):
            return dict(target=(cls.control_dim(controller_id),))
        elif ctype == ControllerType.InverseKinematicsController:
            return dict(target_pos=(3,), target_ori_mat=(3, 3), target_quat=(4,))
        elif ctype == ControllerType.OperationalSpaceController:
            return dict(target_pos=(3,), target_ori_mat=(3, 3))
        elif ctype == ControllerType.DifferentialDriveController:
            return dict(vel=(2,))
        elif ctype == ControllerType.MultiFingerGripperController:
            return dict(target=(cls.command_dim(controller_id),))
        raise ValueError(f"Unknown controller type: {ctype}")

    @staticmethod
    def nums2array(nums, dim):
        if isinstance(nums, str):
            raise TypeError("Error: Only numeric inputs are supported for this function, nums2array!")
        return (
            nums
            if isinstance(nums, cb.arr_type)
            else cb.array(nums)
            if isinstance(nums, Iterable)
            else cb.ones(dim) * nums
        )
    
    @classmethod
    def state_size(cls, controller_id: str):
        size = cls.goal_dim(controller_id) + 1
        ctype = cls._types[controller_id]
        if ctype == ControllerType.InverseKinematicsController:
            control_filter = cls._state[controller_id]["control_filter"]
            size += 0 if control_filter is None else control_filter.state_size
        return size
    
    @classmethod
    def goal(cls, controller_id: str):
        return cls._goals[controller_id]
    
    @classmethod
    def goal_dim(cls, controller_id: str):
        return cls._configs[controller_id]["goal_dim"]
    
    @classmethod
    def control(cls, controller_id: str):
        return cls._controls[controller_id]
    
    @classmethod
    def control_freq(cls, controller_id: str):
        return cls._configs[controller_id]["control_freq"]
    
    @classmethod
    def control_dim(cls, controller_id: str):
        return len(cls._configs[controller_id]["dof_idx"])

    @classmethod
    def control_type(cls, controller_id: str):
        ctype = cls._types[controller_id]
        config = cls._configs[controller_id]
        if ctype in _JOINT_TYPES:
            return ControlType.EFFORT if config["use_impedances"] else ControlType.get_type(type_str=config["motor_type"])
        elif ctype == ControllerType.OperationalSpaceController:
            return ControlType.EFFORT
        elif ctype == ControllerType.DifferentialDriveController:
            return ControlType.VELOCITY
        elif ctype == ControllerType.MultiFingerGripperController:
            return ControlType.get_type(type_str=config["motor_type"])
        raise ValueError(f"Unknown controller type: {ctype}")

    @classmethod
    def command_dim(cls, controller_id: str):
        ctype = cls._types[controller_id]
        config = cls._configs[controller_id]
        if ctype == ControllerType.NullJointController:
            return 0
        elif ctype in (ControllerType.JointController, ControllerType.HolonomicBaseJointController):
            return len(cls.dof_idx(controller_id))
        elif ctype == ControllerType.InverseKinematicsController:
            return IK_MODE_COMMAND_DIMS[config["mode"]]
        elif ctype == ControllerType.OperationalSpaceController:
            return config["command_dim"]
        elif ctype == ControllerType.DifferentialDriveController:
            return 2
        elif ctype == ControllerType.MultiFingerGripperController:
            if config["mode"] == "independent":
                return len(cls.dof_idx(controller_id))
            return 1
        raise ValueError(f"Unknown controller type: {ctype}")
    
    @classmethod
    def isaac_kp(cls, controller_id: str):
        return cls._configs[controller_id]["isaac_kp"]

    @classmethod
    def isaac_kd(cls, controller_id: str):
        return cls._configs[controller_id]["isaac_kd"]
    
    @classmethod
    def command_input_limits(cls, controller_id: str):
        return cls._configs[controller_id]["command_input_limits"]

    @classmethod
    def command_output_limits(cls, controller_id: str):
        return cls._configs[controller_id]["command_output_limits"]

    @classmethod
    def dof_idx(cls, controller_id: str):
        return cls._configs[controller_id]["dof_idx"]
    
    # -------------------------------------------------------------------------
    # Type-specific utilities
    # -------------------------------------------------------------------------

    @classmethod
    def is_grasping(cls, controller_id: str):
        ctype = cls._types[controller_id]
        if ctype == ControllerType.MultiFingerGripperController:
            return cls._state[controller_id]["is_grasping"]
        return IsGraspingState.UNKNOWN

    @classmethod
    def use_delta_commands(cls, controller_id: str):
        return cls._configs[controller_id].get("use_delta_commands", False)
    
    @classmethod
    def motor_type(cls, controller_id: str):
        return cls._configs[controller_id].get("motor_type", None)

    @classmethod
    def update_default_goal(cls, controller_id: str, target):
        assert cls._types[controller_id] == ControllerType.NullJointController
        assert (
            len(target) == cls.control_dim(controller_id)
        ), f"Default goal must be length: {cls.control_dim(controller_id)}, got length: {len(target)}"
        cls._state[controller_id]["default_goal"] = cb.array(target)

    @classmethod
    def _clear_variable_gains(cls, controller_id: str):
        config = cls._configs[controller_id]
        if config["variable_kp"]:
            config["kp"] = None
        if config["variable_damping_ratio"]:
            config["damping_ratio"] = None
        if config["variable_kp_null"]:
            config["kp_null"] = None
            config["kd_null"] = None
    
    @classmethod
    def _update_variable_gains(cls, controller_id: str, gains):
        config = cls._configs[controller_id]
        idx = 0
        if config["variable_kp"]:
            config["kp"] = gains[:, idx : idx + 6]
            idx += 6
        if config["variable_damping_ratio"]:
            config["damping_ratio"] = gains[:, idx : idx + 6]
            idx += 6
        if config["variable_kp_null"]:
            config["kp_null"] = gains[:, idx : idx + cls.control_dim(controller_id)]
            config["kd_null"] = 2 * cb.sqrt(config["kp_null"])
            idx += cls.control_dim(controller_id)

    @classmethod
    def _update_grasping_state(cls, controller_id: str, control_dict):
        config = cls._configs[controller_id]
        finger_vel = cls._state[controller_id]["vel_filter"].estimate(control_dict["joint_velocity"][cls.dof_idx(controller_id)])

        if config["mode"] == "independent":
            is_grasping = IsGraspingState.UNKNOWN
        elif cls.control(controller_id) is None:
            is_grasping = IsGraspingState.FALSE
        elif not cb.all(cls.control(controller_id) == cls.control(controller_id)[0]):
            is_grasping = IsGraspingState.UNKNOWN
        elif not m.POS_TOLERANCE > config["limit_tolerance"]:
            is_grasping = IsGraspingState.UNKNOWN
        else:
            finger_pos = control_dict["joint_position"][cls.dof_idx(controller_id)]

            if config["motor_type"] == "position" and cb.abs(finger_pos - cls.control(controller_id)).mean() < m.POS_TOLERANCE:
                is_grasping = IsGraspingState.UNKNOWN
            elif config["motor_type"] in {"velocity", "torque"} and cb.abs(cls.control(controller_id)).mean() < m.VEL_TOLERANCE:
                is_grasping = IsGraspingState.UNKNOWN
            else:
                min_pos = config["control_limits"][ControlType.POSITION][0][cls.dof_idx(controller_id)]
                max_pos = config["control_limits"][ControlType.POSITION][1][cls.dof_idx(controller_id)]
                finger_pos = finger_pos.clip(min_pos, max_pos)
                dist_from_lower_limit = finger_pos - min_pos
                dist_from_upper_limit = max_pos - finger_pos
                valid_grasp_pos = (
                    dist_from_lower_limit.mean() > m.POS_TOLERANCE or dist_from_upper_limit.mean() > m.POS_TOLERANCE
                )
                valid_grasp_vel = cb.all(cb.abs(finger_vel) < m.VEL_TOLERANCE)
                is_grasping = IsGraspingState.TRUE if valid_grasp_pos and valid_grasp_vel else IsGraspingState.FALSE

        cls._state[controller_id]["is_grasping"] = is_grasping



# =============================================================================
# JIT Functions: Joint Controller
# =============================================================================

@torch_compile
def _compute_joint_torques_torch(u: th.Tensor, mm: th.Tensor, dof_idx: th.Tensor):
    dof_idxs_mat = th.meshgrid(dof_idx, dof_idx, indexing="xy")
    return mm[dof_idxs_mat] @ u


@jit(nopython=True)
def numba_ix(arr, rows, cols):
    one_d_index = np.zeros(len(rows) * len(cols), dtype=np.int32)
    for i, r in enumerate(rows):
        start = i * len(cols)
        one_d_index[start : start + len(cols)] = cols + arr.shape[1] * r
    arr_1d = arr.reshape((arr.shape[0] * arr.shape[1], 1))
    slice_1d = np.take(arr_1d, one_d_index)
    return slice_1d.reshape((len(rows), len(cols)))


@jit(nopython=True)
def _compute_joint_torques_numpy(u, mm, dof_idx):
    return numba_ix(mm, dof_idx, dof_idx) @ u


add_compute_function(
    name="compute_joint_torques", np_function=_compute_joint_torques_numpy, th_function=_compute_joint_torques_torch
)


# =============================================================================
# JIT Functions: IK Controller
# =============================================================================

@th.jit.script
def _compute_ik_qpos_torch(
    q: th.Tensor, j_eef: th.Tensor, ee_pos: th.Tensor, ee_mat: th.Tensor,
    goal_pos: th.Tensor, goal_ori_mat: th.Tensor, q_lower_limit: th.Tensor, q_upper_limit: th.Tensor,
):
    pos_err = goal_pos - ee_pos
    ori_err = TT.orientation_error(goal_ori_mat, ee_mat)
    err = th.cat([pos_err, ori_err])
    j_eef_pinv = th.linalg.pinv(j_eef)
    delta_j = j_eef_pinv @ err
    target_joint_pos = q + delta_j
    return target_joint_pos.clip(min=q_lower_limit, max=q_upper_limit)


@jit(nopython=True)
def _compute_ik_qpos_numpy(
    q, j_eef, ee_pos, ee_mat, goal_pos, goal_ori_mat, q_lower_limit, q_upper_limit,
):
    pos_err = goal_pos - ee_pos
    ori_err = NT.orientation_error(goal_ori_mat, ee_mat).astype(np.float32)
    err = np.concatenate((pos_err, ori_err))
    j_eef_pinv = np.linalg.pinv(j_eef)
    delta_j = j_eef_pinv @ err
    target_joint_pos = q + delta_j
    return target_joint_pos.clip(q_lower_limit, q_upper_limit)


add_compute_function(name="compute_ik_qpos", np_function=_compute_ik_qpos_numpy, th_function=_compute_ik_qpos_torch)


def _compute_ik_qpos_batch_torch(
    q: th.Tensor, j_eef: th.Tensor, ee_pos: th.Tensor, ee_mat: th.Tensor,
    goal_pos: th.Tensor, goal_ori_mat: th.Tensor, q_lower_limit: th.Tensor, q_upper_limit: th.Tensor,
):
    pos_err = goal_pos - ee_pos
    ori_err = TT.orientation_error(goal_ori_mat, ee_mat)
    err = th.cat([pos_err, ori_err], dim=-1)
    j_eef_pinv = th.linalg.pinv(j_eef)
    delta_j = (j_eef_pinv @ err.unsqueeze(-1)).squeeze(-1)
    target_joint_pos = q + delta_j
    return target_joint_pos.clip(min=q_lower_limit, max=q_upper_limit)


def _compute_ik_qpos_batch_numpy(
    q, j_eef, ee_pos, ee_mat, goal_pos, goal_ori_mat, q_lower_limit, q_upper_limit,
):
    pos_err = goal_pos - ee_pos
    ori_err = NT.orientation_error(goal_ori_mat, ee_mat).astype(np.float32)
    err = np.concatenate([pos_err, ori_err], axis=-1)
    j_eef_pinv = np.linalg.pinv(j_eef)
    delta_j = (j_eef_pinv @ err[..., None])[..., 0]
    target_joint_pos = q + delta_j
    return target_joint_pos.clip(q_lower_limit, q_upper_limit)


add_compute_function(
    name="compute_ik_qpos_batch", np_function=_compute_ik_qpos_batch_numpy, th_function=_compute_ik_qpos_batch_torch
)


# =============================================================================
# JIT Functions: OSC Controller
# =============================================================================

@th.jit.script
def _compute_osc_torques_torch(
    q: th.Tensor, qd: th.Tensor, mm: th.Tensor, j_eef: th.Tensor,
    ee_pos: th.Tensor, ee_mat: th.Tensor, ee_lin_vel: th.Tensor, ee_ang_vel_err: th.Tensor,
    goal_pos: th.Tensor, goal_ori_mat: th.Tensor,
    kp: th.Tensor, kd: th.Tensor, kp_null: th.Tensor, kd_null: th.Tensor,
    rest_qpos: th.Tensor, control_dim: int, decouple_pos_ori: bool,
    base_lin_vel: th.Tensor, base_ang_vel: th.Tensor,
):
    mm_inv = th.linalg.inv(mm)
    pos_err = goal_pos - ee_pos
    ori_err = TT.orientation_error(goal_ori_mat, ee_mat)
    err = th.cat((pos_err, ori_err))
    lin_vel_err = base_lin_vel + th.linalg.cross(base_ang_vel, ee_pos) - ee_lin_vel
    vel_err = th.cat((lin_vel_err, ee_ang_vel_err))
    err = th.unsqueeze(kp * err + kd * vel_err, dim=-1)
    m_eef_inv = j_eef @ mm_inv @ j_eef.T
    m_eef = th.linalg.inv(m_eef_inv)

    if decouple_pos_ori:
        m_eef_pos_inv = j_eef[:3, :] @ mm_inv @ j_eef[:3, :].T
        m_eef_ori_inv = j_eef[3:, :] @ mm_inv @ j_eef[3:, :].T
        m_eef_pos = th.linalg.inv(m_eef_pos_inv)
        m_eef_ori = th.linalg.inv(m_eef_ori_inv)
        wrench_pos = m_eef_pos @ err[:3, :]
        wrench_ori = m_eef_ori @ err[3:, :]
        wrench = th.cat((wrench_pos, wrench_ori))
    else:
        wrench = m_eef @ err

    u = j_eef.T @ wrench

    if rest_qpos is not None:
        j_eef_inv = m_eef @ j_eef @ mm_inv
        u_null = kd_null * -qd + kp_null * wrap_angle(rest_qpos - q)
        u_null = mm @ th.unsqueeze(u_null, dim=-1)
        u += (th.eye(control_dim, dtype=th.float32) - j_eef.T @ j_eef_inv) @ u_null

    return u


@jit(nopython=True)
def _compute_osc_torques_numpy(
    q, qd, mm, j_eef, ee_pos, ee_mat, ee_lin_vel, ee_ang_vel_err,
    goal_pos, goal_ori_mat, kp, kd, kp_null, kd_null, rest_qpos,
    control_dim, decouple_pos_ori, base_lin_vel, base_ang_vel,
):
    mm_inv = np.linalg.inv(mm)
    pos_err = goal_pos - ee_pos
    ori_err = NT.orientation_error(goal_ori_mat, ee_mat).astype(np.float32)
    err = np.concatenate((pos_err, ori_err))
    lin_vel_err = base_lin_vel + np.cross(base_ang_vel, ee_pos) - ee_lin_vel
    vel_err = np.concatenate((lin_vel_err, ee_ang_vel_err))
    err = np.expand_dims(kp * err + kd * vel_err, axis=-1)
    m_eef_inv = j_eef @ mm_inv @ j_eef.T
    m_eef = np.linalg.inv(m_eef_inv)

    if decouple_pos_ori:
        m_eef_pos_inv = j_eef[:3, :] @ mm_inv @ j_eef[:3, :].T
        m_eef_ori_inv = j_eef[3:, :] @ mm_inv @ j_eef[3:, :].T
        m_eef_pos = np.linalg.inv(m_eef_pos_inv)
        m_eef_ori = np.linalg.inv(m_eef_ori_inv)
        wrench_pos = m_eef_pos @ err[:3, :]
        wrench_ori = m_eef_ori @ err[3:, :]
        wrench = np.concatenate((wrench_pos, wrench_ori))
    else:
        wrench = m_eef @ err

    u = j_eef.T @ wrench

    if rest_qpos is not None:
        j_eef_inv = m_eef @ j_eef @ mm_inv
        u_null = kd_null * -qd + kp_null * ((rest_qpos - q + np.pi) % (2 * np.pi) - np.pi)
        u_null = mm @ np.expand_dims(u_null, axis=-1).astype(np.float32)
        u += (np.eye(control_dim, dtype=np.float32) - j_eef.T @ j_eef_inv) @ u_null

    return u


add_compute_function(
    name="compute_osc_torques", np_function=_compute_osc_torques_numpy, th_function=_compute_osc_torques_torch
)


def _compute_osc_torques_batch_torch(
    q, qd, mm, j_eef, ee_pos, ee_mat, ee_lin_vel, ee_ang_vel_err,
    goal_pos, goal_ori_mat, kp, kd, kp_null, kd_null, rest_qpos,
    max_dim, decouple_pos_ori, base_lin_vel, base_ang_vel,
):
    mm_inv = th.linalg.inv(mm)
    pos_err = goal_pos - ee_pos
    ori_err = TT.orientation_error(goal_ori_mat, ee_mat)
    lin_vel_err = base_lin_vel + th.linalg.cross(base_ang_vel, ee_pos) - ee_lin_vel
    vel_err = th.cat([lin_vel_err, ee_ang_vel_err], dim=-1)
    task_err = th.cat([pos_err, ori_err], dim=-1)
    err = (kp * task_err + kd * vel_err).unsqueeze(-1)
    j_eef_T = j_eef.transpose(-2, -1)

    if decouple_pos_ori:
        j_pos = j_eef[:, :3, :]
        j_ori = j_eef[:, 3:, :]
        m_eef_pos = th.linalg.inv(j_pos @ mm_inv @ j_pos.transpose(-2, -1))
        m_eef_ori = th.linalg.inv(j_ori @ mm_inv @ j_ori.transpose(-2, -1))
        wrench = th.cat([m_eef_pos @ err[:, :3, :], m_eef_ori @ err[:, 3:, :]], dim=1)
        m_eef = th.linalg.inv(j_eef @ mm_inv @ j_eef_T)
    else:
        m_eef = th.linalg.inv(j_eef @ mm_inv @ j_eef_T)
        wrench = m_eef @ err

    u = j_eef_T @ wrench

    j_eef_inv = m_eef @ j_eef @ mm_inv
    angle_diff = (rest_qpos - q + math.pi) % (2 * math.pi) - math.pi
    u_null = (kd_null * (-qd) + kp_null * angle_diff).unsqueeze(-1)
    u_null = mm @ u_null
    eye = th.eye(max_dim, dtype=th.float32).unsqueeze(0)
    nullspace_proj = eye - j_eef_T @ j_eef_inv
    u = u + nullspace_proj @ u_null

    return u.squeeze(-1)


def _compute_osc_torques_batch_numpy(
    q, qd, mm, j_eef, ee_pos, ee_mat, ee_lin_vel, ee_ang_vel_err,
    goal_pos, goal_ori_mat, kp, kd, kp_null, kd_null, rest_qpos,
    max_dim, decouple_pos_ori, base_lin_vel, base_ang_vel,
):
    mm_inv = np.linalg.inv(mm)
    pos_err = goal_pos - ee_pos
    ori_err = NT.orientation_error(goal_ori_mat, ee_mat).astype(np.float32)
    lin_vel_err = base_lin_vel + np.cross(base_ang_vel, ee_pos) - ee_lin_vel
    vel_err = np.concatenate([lin_vel_err, ee_ang_vel_err], axis=-1)
    task_err = np.concatenate([pos_err, ori_err], axis=-1)
    err = np.expand_dims(kp * task_err + kd * vel_err, axis=-1)
    j_eef_T = np.swapaxes(j_eef, -2, -1)

    if decouple_pos_ori:
        j_pos = j_eef[:, :3, :]
        j_ori = j_eef[:, 3:, :]
        m_eef_pos = np.linalg.inv(j_pos @ mm_inv @ np.swapaxes(j_pos, -2, -1))
        m_eef_ori = np.linalg.inv(j_ori @ mm_inv @ np.swapaxes(j_ori, -2, -1))
        wrench = np.concatenate([m_eef_pos @ err[:, :3, :], m_eef_ori @ err[:, 3:, :]], axis=1)
        m_eef = np.linalg.inv(j_eef @ mm_inv @ j_eef_T)
    else:
        m_eef = np.linalg.inv(j_eef @ mm_inv @ j_eef_T)
        wrench = m_eef @ err

    u = j_eef_T @ wrench

    j_eef_inv = m_eef @ j_eef @ mm_inv
    angle_diff = (rest_qpos - q + np.pi) % (2 * np.pi) - np.pi
    u_null = np.expand_dims(kd_null * (-qd) + kp_null * angle_diff, axis=-1).astype(np.float32)
    u_null = mm @ u_null
    eye = np.expand_dims(np.eye(max_dim, dtype=np.float32), axis=0)
    nullspace_proj = eye - j_eef_T @ j_eef_inv
    u = u + nullspace_proj @ u_null

    return u.squeeze(-1) if u.ndim > 2 else u.reshape(u.shape[0], -1)


add_compute_function(
    name="compute_osc_torques_batch",
    np_function=_compute_osc_torques_batch_numpy,
    th_function=_compute_osc_torques_batch_torch,
)
