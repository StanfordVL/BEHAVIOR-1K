"""
Controller singleton call-flow reference
=========================================

────────────────────────────────────────────────────────────────────────────────
 FLOW 1 — update_goal  (called once per robot per env step, before physics step)
────────────────────────────────────────────────────────────────────────────────

 env._pre_step()
 └─ robot.apply_action(action)                            # fans out flat action
    └─ Controller.update_goal(controller_id, command)      # per-controller slice
          ├─ Controller._preprocess_command(controller_id, command)
          │   ├─ [NullJoint]  replace with default_goal
          │   ├─ [Gripper]    broadcast scalar → command_dim; apply inversion
          │   └─ Controller._preprocess_command_base()     # clip + affine scale
          └─ Controller._update_goal(controller_id, preprocessed_command)
             ├─ [JointController / NullJoint]
             │    └─ _update_goal_joint()      # delta → abs; clip to joint limits
             ├─ [HolonomicBaseJointController]
             │    └─ _update_goal_holonomic()  # base→canonical frame; → _update_goal_joint
             ├─ [InverseKinematicsController]
             │    └─ _update_goal_ik()         # mode dispatch → {target_pos, target_ori_mat}
             ├─ [OperationalSpaceController]
             │    └─ _update_goal_osc()        # same modes as IK → {target_pos, target_ori_mat}
             ├─ [DifferentialDriveController]  → {"vel": [lin, ang]}
             └─ [MultiFingerGripperController] → {"target": command}

 All _update_goal_* methods read current robot state inline via:
   ControllableObjectViewAPI.get_joint_positions / get_link_relative_position_orientation


────────────────────────────────────────────────────────────────────────────────
 FLOW 2 — controller step  (called during physics pre-render, after update_goal)
────────────────────────────────────────────────────────────────────────────────

 simulator._on_pre_render()
 └─ Controller.step()                                # one call replaces begin/step/deploy
    ├─ [per type group in ids_by_type]
    │   ├─ _step_batch_joint(active_cids)            # JointController / Holonomic / Null
    │   ├─ _step_batch_ik(active_cids)               # InverseKinematicsController
    │   ├─ _step_batch_osc(active_cids)              # OperationalSpaceController
    │   ├─ _step_batch_dd(active_cids)               # DifferentialDriveController
    │   └─ _step_batch_gripper(active_cids)          # MultiFingerGripperController
    └─ [deploy per controller directly]
       ├─ ControllableObjectViewAPI.set_joint_position_targets(arpath, positions, indices=dof_idx)
       ├─ ControllableObjectViewAPI.set_joint_velocity_targets(arpath, velocities, indices=dof_idx)
       └─ ControllableObjectViewAPI.set_joint_efforts(arpath, efforts, indices=dof_idx)

────────────────────────────────────────────────────────────────────────────────

"""

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
from omnigibson.utils.usd_utils import ControllableObjectViewAPI

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

_JOINT_TYPES = frozenset(
    {
        ControllerType.JointController,
        ControllerType.NullJointController,
        ControllerType.HolonomicBaseJointController,
        ControllerType.InverseKinematicsController,
    }
)


class Controller(Serializable, Recreatable):
    """
    Unified singleton controller class. All controller logic lives here, dispatched
    by ControllerType stored in types[controller_id].
    """

    type_by_id = {}  # controller_id → ControllerType enum value
    control_type = {}  # controller_id → ControlType output constant (POSITION/VELOCITY/EFFORT); cached at registration time for O(1) lookup
    configs = {}  # controller_id → fully processed config dict (includes structural keys like _mm_start_idx, _task_space_link_name, etc.)
    goals = {}  # controller_id → current internal goal dict (None until the first update_goal call)
    controls = {}  # controller_id → last computed control array (None before the first step)
    state = {}  # controller_id → mutable runtime state dict (filters, grasping state, fixed orientation target, etc.)
    disabled_controllers = set()  # set of controller_ids whose control is disabled (synced with robot.control_enabled)
    articulation_root_paths = {}  # controller_id → articulation root prim path used as key for all ControllableObjectViewAPI calls
    ids_by_type = {}  # ControllerType → ordered list of controller_ids (precomputed at register time for O(1) step dispatch)
    motor_type = {}  # controller_id → motor type string ("position"/"velocity"/"effort"/None)
    use_delta_commands = {}  # controller_id → bool, whether commands are deltas (relative to current state)
    dof_idx = {}  # controller_id → int index array of controlled DOFs into the full robot state
    command_output_limits = {}  # controller_id → (lower, upper) output limits in physical units, or None
    command_input_limits = {}  # controller_id → (lower, upper) normalised input limits, or None
    isaac_kp = {}  # controller_id → per-DOF kp gains for Isaac Sim PD controller, or None
    isaac_kd = {}  # controller_id → per-DOF kd gains for Isaac Sim PD controller, or None
    command_dim = {}  # controller_id → expected command vector length
    control_dim = {}  # controller_id → number of actuated DOFs driven by this controller
    control_freq = {}  # controller_id → control frequency in Hz
    goal_dim = {}  # controller_id → total scalar count of the serialised goal dict

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    @classmethod
    def register(
        cls,
        controller_id,
        config,
        controller_type=None,
        articulation_root_path=None,
        mm_start_idx=0,
        task_space_link_name=None,
        jac_link_row=None,
        jac_col_start=0,
        jac_n_cols=None,
    ):
        """Register a controller with the singleton and store all data needed for inline state access.

        Processes the raw config dict, initialises per-controller state, and stores
        robot-level metadata (articulation root path, DOF count, controller ordering)
        so that the Controller can call ControllableObjectViewAPI directly without
        holding a reference to the robot object.

        Structural values that are expensive or impossible to recompute at each step
        (Jacobian row index, column slicing bounds, mass-matrix start index) are
        embedded directly into the processed config under private underscore keys.

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.
            config (dict): Raw controller configuration dict (will be deep-copied and processed).
            controller_type (ControllerType, optional): The controller type enum value.
            articulation_root_path (str, optional): Articulation root prim path used as the
                key for all ControllableObjectViewAPI calls for this robot.
            mm_start_idx (int): Row/column offset into the generalised mass matrix used to
                skip the 6 virtual-base DOFs for floating-base robots (``0`` for fixed-base,
                ``6`` for floating-base).
            task_space_link_name (str, optional): Name of the task-space link (EEF or trunk tip) used
                by IK and OSC controllers for pose and Jacobian queries. ``None`` for
                joint-level controllers.
            jac_link_row (int, optional): Negative index into the full Jacobian tensor for
                the task link, computed as ``-(n_links - link_idx)``. Negative indexing is
                stable across fixed-base and floating-base topologies.
            jac_col_start (int): First column of the actuated-joint block in the full
                Jacobian (``0`` for fixed-base, ``6`` for floating-base).
            jac_n_cols (int, optional): Number of actuated-joint columns in the Jacobian
                (equal to the robot's total joint count).
        """
        cls.type_by_id[controller_id] = controller_type
        cls.configs[controller_id] = cls._process_config(controller_id, config)
        cls.control_type[controller_id] = cls._compute_control_type(controller_type, cls.configs[controller_id])
        cls.goals[controller_id] = None
        cls.controls[controller_id] = None

        # Precompute per-controller lookup dicts
        cfg = cls.configs[controller_id]
        cls.dof_idx[controller_id] = cfg["dof_idx"]
        cls.control_dim[controller_id] = len(cfg["dof_idx"])
        cls.goal_dim[controller_id] = cfg["goal_dim"]
        cls.control_freq[controller_id] = cfg["control_freq"]
        cls.isaac_kp[controller_id] = cfg["isaac_kp"]
        cls.isaac_kd[controller_id] = cfg["isaac_kd"]
        cls.command_input_limits[controller_id] = cfg["command_input_limits"]
        cls.command_output_limits[controller_id] = cfg["command_output_limits"]
        cls.use_delta_commands[controller_id] = cfg.get("use_delta_commands", False)
        cls.motor_type[controller_id] = cfg.get("motor_type", None)
        cls.command_dim[controller_id] = cls._command_dim_from_config(controller_id, cfg)

        cls.state[controller_id] = {}
        cls._init_state(controller_id=controller_id)

        # Store robot-level info
        assert articulation_root_path is not None, "articulation_root_path of a robot can't be None."
        cls.articulation_root_paths[controller_id] = articulation_root_path
        assert controller_id not in cls.ids_by_type.setdefault(controller_type, [])
        cls.ids_by_type[controller_type].append(controller_id)

        # Embed structural data into config for inline use in step/goal methods
        cfg = cls.configs[controller_id]
        cfg["_mm_start_idx"] = mm_start_idx
        if task_space_link_name is not None:
            cfg.update(
                _task_space_link_name=task_space_link_name,
                _jac_link_row=jac_link_row,
                _jac_col_start=jac_col_start,
                _jac_n_cols=jac_n_cols,
            )

    @classmethod
    def unregister(cls, controller_id: str):
        """Remove a controller and clean up robot-level metadata when the last controller is removed.

        All per-controller dicts (types, configs, goals, controls, state, articulation_root_paths) are purged.

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.
        """
        ctype = cls.type_by_id.get(controller_id)
        if ctype is not None and ctype in cls.ids_by_type:
            cls.ids_by_type[ctype] = [c for c in cls.ids_by_type[ctype] if c != controller_id]
            if not cls.ids_by_type[ctype]:
                cls.ids_by_type.pop(ctype)
        cls.type_by_id.pop(controller_id, None)
        cls.control_type.pop(controller_id, None)
        cls.configs.pop(controller_id, None)
        cls.goals.pop(controller_id, None)
        cls.controls.pop(controller_id, None)
        cls.state.pop(controller_id, None)
        cls.dof_idx.pop(controller_id, None)
        cls.control_dim.pop(controller_id, None)
        cls.goal_dim.pop(controller_id, None)
        cls.control_freq.pop(controller_id, None)
        cls.isaac_kp.pop(controller_id, None)
        cls.isaac_kd.pop(controller_id, None)
        cls.command_input_limits.pop(controller_id, None)
        cls.command_output_limits.pop(controller_id, None)
        cls.use_delta_commands.pop(controller_id, None)
        cls.motor_type.pop(controller_id, None)
        cls.command_dim.pop(controller_id, None)

        cls.articulation_root_paths.pop(controller_id, None)

    @classmethod
    def disable(cls, controller_id: str):
        """Suppress control output for one controller.

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.
        """
        cls.disabled_controllers.add(controller_id)

    @classmethod
    def enable(cls, controller_id: str):
        """Re-enable control output for one controller previously disabled via :meth:`disable`.

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.
        """
        cls.disabled_controllers.discard(controller_id)

    # -------------------------------------------------------------------------
    # Config processing
    # -------------------------------------------------------------------------

    @classmethod
    def _process_config(cls, controller_id: str, input_config: dict):
        """Deep-copy and process a raw config dict through three ordered phases.

        Phase 1: type-specific pre-processing (IK/OSC/DD/Holonomic/Gripper) that
        sets defaults and validates mode-specific fields before joint processing runs.

        Phase 2: joint-level processing (_process_config_joint) for any controller
        that drives joints directly (JointController, NullJointController,
        HolonomicBaseJointController, InverseKinematicsController).

        Phase 3: common base processing (_process_config_base) that converts control
        limits, computes command scaling, and validates Isaac Sim PD gains for all types.

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.
            input_config (dict): Raw controller configuration dict.

        Returns:
            dict: Fully processed configuration dict ready for use in goal/step methods.
        """
        config = deepcopy(input_config)
        ctype = cls.type_by_id[controller_id]

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
        """Validate and normalise fields shared by all joint-based controllers.

        Validates ``motor_type`` and sets gain defaults (``pos_kp``, ``pos_kd``,
        ``vel_kp``) appropriate for the motor type. Also enforces that mutually
        exclusive gain fields are not set together, and raises an error when
        ``use_delta_commands`` is combined with the "default" output limits
        (which would produce nonsensical scaling).

        Args:
            config (dict): Config dict mutated in place.
        """
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
            assert (
                pos_damping_ratio is None
            ), "Cannot set pos_damping_ratio for JointController with motor_type=velocity!"
        else:
            assert pos_kp is None, "Cannot set pos_kp for JointController with motor_type=effort!"
            assert pos_damping_ratio is None, "Cannot set pos_damping_ratio for JointController with motor_type=effort!"
            assert vel_kp is None, "Cannot set vel_kp for JointController with motor_type=effort!"

        config["pos_kp"] = pos_kp
        config["pos_kd"] = (
            None if pos_kp is None or pos_damping_ratio is None else 2 * math.sqrt(pos_kp) * pos_damping_ratio
        )
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
        """Validate and initialise config for a holonomic base controller.

        Asserts that exactly 3 DOFs are specified (x translation, y translation,
        yaw rotation), then disables delta commands and clears the quaternion-space
        delta flag because holonomic commands are always absolute velocity or
        position targets expressed in the canonical frame.

        Args:
            config (dict): Config dict mutated in place.
        """
        assert len(config["dof_idx"]) == 3, f"Expected 3 DOFs for holonomic base control, got {len(config['dof_idx'])}"
        config["use_delta_commands"] = False
        config["compute_delta_in_quat_space"] = None

    @classmethod
    def _process_config_ik(cls, config):
        """Validate and initialise config for an Inverse Kinematics (IK) controller.

        Sets ``motor_type`` to "position" (IK always outputs joint position targets),
        disables delta-command mode, validates the IK mode string, and sets defaults
        for smoothing filter size, workspace pose limiter, and conditioning on the
        current joint position. Also slices ``reset_joint_pos`` down to the controller
        DOFs and processes ``command_input_limits`` / ``command_output_limits`` with
        special handling for ``absolute_pose`` and ``pose_absolute_ori`` modes
        (which require full ±π orientation ranges).

        Args:
            config (dict): Config dict mutated in place.
        """
        config["motor_type"] = "position"
        config["use_delta_commands"] = False
        config["mode"] = config.get("mode", "pose_delta_ori")
        assert (
            config["mode"] in IK_MODES
        ), f"Invalid ik mode specified! Valid options are: {IK_MODES}, got: {config['mode']}"
        config["use_impedances"] = config.get("use_impedances", False)
        config["smoothing_filter_size"] = config.get("smoothing_filter_size", None)
        config["workspace_pose_limiter"] = config.get("workspace_pose_limiter", None)
        config["condition_on_current_position"] = config.get("condition_on_current_position", True)
        config["reset_joint_pos"] = config["reset_joint_pos"][config["dof_idx"]]

        command_input_limits = config.get("command_input_limits", "default")
        command_output_limits = config.get(
            "command_output_limits",
            (
                (-0.2, -0.2, -0.2, -0.5, -0.5, -0.5),
                (0.2, 0.2, 0.2, 0.5, 0.5, 0.5),
            ),
        )

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
        """Validate and initialise config for an Operational Space Controller (OSC).

        Converts scalar gain values to arrays of the correct dimension, sets up
        ``kd = 2 * sqrt(kp) * damping_ratio``, flags variable-gain dimensions,
        validates the OSC mode, and computes the final ``command_dim`` (which grows
        if any gains are variable and included in the command). Processes
        ``command_input_limits`` / ``command_output_limits`` with the same
        orientation-range special cases as the IK config. Slices ``reset_joint_pos``
        to the controller DOFs.

        Note: variable gains are not yet supported and will raise an AssertionError.

        Args:
            config (dict): Config dict mutated in place.
        """
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
        command_output_limits = config.get(
            "command_output_limits", ((-0.2, -0.2, -0.2, -0.5, -0.5, -0.5), (0.2, 0.2, 0.2, 0.5, 0.5, 0.5))
        )

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
        """Validate and initialise config for a Differential Drive controller.

        Computes ``wheel_axle_halflength`` from ``wheel_axle_length`` and, when
        ``command_output_limits`` is "default", derives the maximum linear and
        angular velocity limits from the wheel joint velocity limits and robot
        geometry. Asserts that both wheels have the same symmetric velocity range.

        Args:
            config (dict): Config dict mutated in place.
        """
        config["wheel_radius"] = config["wheel_radius"]
        config["wheel_axle_halflength"] = config["wheel_axle_length"] / 2.0

        command_output_limits = config.get("command_output_limits", "default")
        if type(command_output_limits) is str and command_output_limits == "default":
            control_limits = config["control_limits"]
            dof_idx = config["dof_idx"]
            min_vels = control_limits["velocity"][0][dof_idx]
            assert (
                min_vels[0] == min_vels[1]
            ), "Differential drive requires both wheel joints to have same min velocities!"
            max_vels = control_limits["velocity"][1][dof_idx]
            assert (
                max_vels[0] == max_vels[1]
            ), "Differential drive requires both wheel joints to have same max velocities!"
            assert abs(min_vels[0]) == abs(
                max_vels[0]
            ), "Differential drive requires both wheel joints to have same min and max absolute velocities!"
            max_lin_vel = max_vels[0] * config["wheel_radius"]
            max_ang_vel = max_lin_vel * 2.0 / config["wheel_axle_halflength"]
            config["command_output_limits"] = ((-max_lin_vel, -max_ang_vel), (max_lin_vel, max_ang_vel))

    @classmethod
    def _process_config_gripper(cls, config):
        """Validate and initialise config for a MultiFingerGripper controller.

        Validates ``motor_type`` and ``mode`` (binary / smooth / independent),
        sets ``inverted``, ``limit_tolerance``, and converts ``open_qpos`` /
        ``closed_qpos`` to backend arrays if provided. Forces
        ``command_output_limits`` to "default" for binary mode (since binary
        commands are always ±1).

        Args:
            config (dict): Config dict mutated in place.
        """
        assert_valid_key(key=config["motor_type"].lower(), valid_keys=ControlType.VALID_TYPES_STR, name="motor_type")
        config["motor_type"] = config["motor_type"].lower()
        assert_valid_key(
            key=config.get("mode", "binary"), valid_keys=GRIPPER_MODES, name="mode for multi finger gripper"
        )
        config["mode"] = config.get("mode", "binary")
        config["inverted"] = config.get("inverted", False)
        config["limit_tolerance"] = config.get("limit_tolerance", 0.001)
        config["open_qpos"] = None if config.get("open_qpos", None) is None else cb.array(config.get("open_qpos"))
        config["closed_qpos"] = None if config.get("closed_qpos", None) is None else cb.array(config.get("closed_qpos"))

        if config["mode"] == "binary":
            config["command_output_limits"] = "default"

    @classmethod
    def _process_config_base(cls, controller_id: str, config):
        """Apply common post-processing shared by all controller types.

        Steps performed:
        1. Convert ``dof_idx`` to integer indices.
        2. Resolve "default" command input/output limits via the type-specific
           helpers and broadcast scalar limits to full command-dimension arrays.
        3. Validate and fill Isaac Sim PD gains (``isaac_kp`` / ``isaac_kd``)
           according to the controller's output control type.
        4. Set ``goal_shapes`` and ``goal_dim`` used for serialisation.

        This method is called last in ``_process_config`` and must not be called
        before all type-specific pre-processing is complete.

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.
            config (dict): Partially processed config dict (mutated in place).

        Returns:
            dict: Fully processed config dict (same object as ``config``).
        """
        config["dof_idx"] = cb.as_int(config["dof_idx"])
        config["command_input_limits"] = config.get("command_input_limits", "default")
        config["command_output_limits"] = config.get("command_output_limits", "default")

        cls.configs[controller_id] = config

        control_limits = {}
        for motor_type in {"position", "velocity", "effort"}:
            if motor_type not in config["control_limits"]:
                continue
            control_limits[ControlType.get_type(motor_type)] = [
                config["control_limits"][motor_type][0],
                config["control_limits"][motor_type][1],
            ]
        assert (
            "has_limit" in config["control_limits"]
        ), "Expected has_limit specified in control_limits, but does not exist."
        control_limits["has_limit"] = config["control_limits"]["has_limit"]
        config["control_limits"] = control_limits
        config["dof_has_limits"] = control_limits["has_limit"]

        command_dim = cls._command_dim_from_config(controller_id, config)
        config["goal_shapes"] = cls._get_goal_shapes(controller_id, command_dim, len(config["dof_idx"]))
        config["goal_dim"] = int(sum(cb.prod(cb.array(shape)) for shape in config["goal_shapes"].values()))

        cls.controls[controller_id] = None
        cls.goals[controller_id] = None
        config["command_scale_factor"] = None
        config["command_output_transform"] = None
        config["command_input_transform"] = None

        command_input_limits = config["command_input_limits"]
        command_output_limits = config["command_output_limits"]
        if type(command_input_limits) is str and command_input_limits == "default":
            command_input_limits = (-1.0, 1.0)
        if type(command_output_limits) is str and command_output_limits == "default":
            command_output_limits = cls._generate_default_command_output_limits(controller_id)

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
        ct = cls._compute_control_type(cls.type_by_id[controller_id], config)
        if ct == ControlType.POSITION:
            isaac_kp = m.DEFAULT_ISAAC_KP if isaac_kp is None else isaac_kp
            isaac_kd = m.DEFAULT_ISAAC_KD if isaac_kd is None else isaac_kd
        elif ct == ControlType.VELOCITY:
            assert (
                isaac_kp is None
            ), f"Control type for controller {controller_id} is VELOCITY, so no isaac_kp should be set!"
            isaac_kd = m.DEFAULT_ISAAC_KP if isaac_kd is None else isaac_kd
        elif ct == ControlType.EFFORT:
            assert (
                isaac_kp is None
            ), f"Control type for controller {controller_id} is EFFORT, so no isaac_kp should be set!"
            assert (
                isaac_kd is None
            ), f"Control type for controller {controller_id} is EFFORT, so no isaac_kd should be set!"
        else:
            raise ValueError(f"Expected control type to be one of: [POSITION, VELOCITY, EFFORT], but got: {ct}")

        control_dim = len(config["dof_idx"])
        config["isaac_kp"] = None if isaac_kp is None else cls.nums2array(isaac_kp, control_dim)
        config["isaac_kd"] = None if isaac_kd is None else cls.nums2array(isaac_kd, control_dim)

        return config

    # -------------------------------------------------------------------------
    # State initialization
    # -------------------------------------------------------------------------

    @classmethod
    def _init_state(cls, controller_id: str):
        """Initialise the mutable runtime state dict for a controller.

        State is type-dependent:
        - IK: ``fixed_quat_target`` (None until the first position_fixed_ori command)
          and an optional ``control_filter`` (MovingAverageFilter) for output smoothing.
        - OSC: ``fixed_quat_target`` only.
        - MultiFingerGripper: ``is_grasping`` (IsGraspingState enum) and a velocity
          ``vel_filter`` (MovingAverageFilter with width=5) for grasping detection.
        - NullJoint: ``default_goal`` array (zeros by default, configurable).

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.
        """
        ctype = cls.type_by_id[controller_id]
        if ctype == ControllerType.InverseKinematicsController:
            config = cls.configs[controller_id]
            cls.state[controller_id]["fixed_quat_target"] = None
            cls.state[controller_id]["control_filter"] = (
                None
                if config.get("smoothing_filter_size", None) in {None, 0}
                else MovingAverageFilter(
                    obs_dim=len(cls.dof_idx[controller_id]), filter_width=config["smoothing_filter_size"]
                )
            )
        elif ctype == ControllerType.OperationalSpaceController:
            cls.state[controller_id]["fixed_quat_target"] = None
        elif ctype == ControllerType.MultiFingerGripperController:
            cls.state[controller_id]["is_grasping"] = IsGraspingState.FALSE
            cls.state[controller_id]["vel_filter"] = MovingAverageFilter(
                obs_dim=len(cls.dof_idx[controller_id]), filter_width=5
            )
        elif ctype == ControllerType.NullJointController:
            config = cls.configs[controller_id]
            default_goal = config.get("default_goal", None)
            if default_goal is None:
                default_goal = cb.zeros(len(config["dof_idx"]))
            cls.state[controller_id]["default_goal"] = cb.array(default_goal)

    # -------------------------------------------------------------------------
    # Default limits
    # -------------------------------------------------------------------------

    @classmethod
    def _generate_default_command_output_limits(cls, controller_id: str):
        """Derive default command output limits from the robot's joint control limits.

        For joint-based controllers the limits come from the appropriate motor-type
        slice of ``control_limits``. For multi-finger grippers in binary mode the
        limits are ±1; in smooth mode the scalar mean of the joint limits is used;
        in independent mode the per-joint limits are returned directly.

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.

        Returns:
            tuple: ``(lower_limits_array, upper_limits_array)`` with one entry per
            controlled DOF, in the backend array type.
        """
        config = cls.configs[controller_id]
        ctype = cls.type_by_id[controller_id]

        if ctype in _JOINT_TYPES:
            motor_type = config["motor_type"]
            return (
                config["control_limits"][ControlType.get_type(motor_type)][0][config["dof_idx"]],
                config["control_limits"][ControlType.get_type(motor_type)][1][config["dof_idx"]],
            )
        elif ctype == ControllerType.MultiFingerGripperController:
            ct = cls._compute_control_type(ctype, config)
            base_limits = (
                config["control_limits"][ct][0][config["dof_idx"]],
                config["control_limits"][ct][1][config["dof_idx"]],
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
            ct = cls._compute_control_type(ctype, config)
            return (
                config["control_limits"][ct][0][config["dof_idx"]],
                config["control_limits"][ct][1][config["dof_idx"]],
            )

    # -------------------------------------------------------------------------
    # Goal management
    # -------------------------------------------------------------------------

    @classmethod
    def set_goals(cls, controller_id: str, goals):
        """Directly overwrite the stored goal dict for a controller.

        Bypasses command preprocessing and goal computation — intended for
        state restoration (e.g. ``load_state``) where the goal is already in
        the internal representation.

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.
            goals: Goal dict as returned by one of the ``_update_goal_*`` methods, or None.
        """
        cls.goals[controller_id] = goals

    @classmethod
    def update_goal(cls, controller_id: str, command):
        """Preprocess a raw command and compute/store the new internal goal.

        Validates command dimensionality, passes the command through
        ``_preprocess_command`` (clipping, inversion, scaling), and dispatches
        to the type-specific ``_update_goal_*`` method which converts the
        preprocessed command into the controller's internal goal representation
        (e.g. a target joint position vector, a target EEF pose dict, etc.).

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.
            command: Action array of length ``command_dim(controller_id)``.
        """
        assert (
            len(command) == cls.command_dim[controller_id]
        ), f"Commands must be dimension {cls.command_dim[controller_id]}, got dim {len(command)} instead."
        cls.goals[controller_id] = cls._update_goal(controller_id, cls._preprocess_command(controller_id, command))

    @classmethod
    def _update_goal(cls, controller_id, command):
        """Dispatch a preprocessed command to the type-specific goal-update method.

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.
            command: Preprocessed and scaled command array.

        Returns:
            dict: Internal goal dict whose keys and shapes depend on controller type.
        """
        ctype = cls.type_by_id[controller_id]
        if ctype in (ControllerType.JointController, ControllerType.NullJointController):
            return cls._update_goal_joint(controller_id, command)
        elif ctype == ControllerType.HolonomicBaseJointController:
            return cls._update_goal_holonomic(controller_id, command)
        elif ctype == ControllerType.InverseKinematicsController:
            return cls._update_goal_ik(controller_id, command)
        elif ctype == ControllerType.OperationalSpaceController:
            return cls._update_goal_osc(controller_id, command)
        elif ctype == ControllerType.DifferentialDriveController:
            return dict(vel=command)
        elif ctype == ControllerType.MultiFingerGripperController:
            return dict(target=command)
        raise ValueError(f"Unknown controller type: {ctype}")

    @classmethod
    def _update_goal_joint(cls, controller_id, command):
        """Compute and return the target joint goal for a joint-level controller.

        When ``use_delta_commands`` is True, reads the current joint state from
        ControllableObjectViewAPI and adds the command as a delta, applying
        quaternion-space composition for any rotational joints listed in
        ``compute_delta_in_quat_space``. The resulting target is clipped to the
        configured joint limits before being returned.

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.
            command: Preprocessed command array of length ``control_dim``.

        Returns:
            dict: ``{"target": target_array}`` where ``target_array`` is clipped to limits.
        """
        config = cls.configs[controller_id]
        if config["use_delta_commands"]:
            arpath = cls.articulation_root_paths[controller_id]
            motor_type = config["motor_type"]
            if motor_type == "position":
                base_value = ControllableObjectViewAPI.get_joint_positions(arpath)[cls.dof_idx[controller_id]]
            elif motor_type == "velocity":
                base_value = ControllableObjectViewAPI.get_joint_velocities(arpath, estimate=True)[
                    cls.dof_idx[controller_id]
                ]
            else:
                base_value = ControllableObjectViewAPI.get_joint_efforts(arpath)[cls.dof_idx[controller_id]]
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
            config["control_limits"][ControlType.get_type(config["motor_type"])][0][cls.dof_idx[controller_id]],
            config["control_limits"][ControlType.get_type(config["motor_type"])][1][cls.dof_idx[controller_id]],
        )
        return dict(target=target)

    @classmethod
    def _update_goal_holonomic(cls, controller_id, command):
        """Compute and return the target joint goal for a holonomic base controller.

        Transforms the 3-DOF command (x, y, yaw) from the robot's base frame to the
        canonical (world-aligned) frame. For position motor type, integrates the yaw
        delta onto the current joint yaw reading. The transformed command is then
        passed to ``_update_goal_joint`` for final clipping.

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.
            command: Preprocessed 3-DOF command array ``[x, y, yaw]`` in the base frame.

        Returns:
            dict: ``{"target": target_array}`` in the canonical frame.
        """
        arpath = cls.articulation_root_paths[controller_id]
        root_pos, root_quat = ControllableObjectViewAPI.get_position_orientation(arpath)
        canonical_pos, canonical_quat = ControllableObjectViewAPI.get_root_position_orientation(arpath)
        base_pose = cb.T.pose2mat((root_pos, root_quat))
        canonical_pose = cb.T.pose2mat((canonical_pos, canonical_quat))
        canonical_to_base_pose = cb.T.pose_inv(canonical_pose) @ base_pose

        if cls.configs[controller_id]["motor_type"] == "position":
            command_in_base_frame = cb.as_float32(cb.eye(4))
            command_in_base_frame[:2, 3] = command[:2]
            command_in_canonical_frame = canonical_to_base_pose @ command_in_base_frame
            position = command_in_canonical_frame[:2, 3]
            rz_joint_pos = ControllableObjectViewAPI.get_joint_positions(arpath)[cls.dof_idx[controller_id]][2:3]
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

        return cls._update_goal_joint(controller_id, command=command)

    @classmethod
    def _update_goal_ik(cls, controller_id, command):
        """Compute and return the target EEF pose goal for an IK controller.

        Reads the current EEF position and orientation relative to the robot root
        from ControllableObjectViewAPI. Interprets the 3-6 element command according
        to the configured mode:

        - ``absolute_pose``: command provides the absolute target position and an
          axis-angle orientation in the canonical frame.
        - ``pose_absolute_ori``: delta position + absolute axis-angle orientation.
        - ``pose_delta_ori``: delta position + delta axis-angle orientation applied
          as a left-multiply in SO(3) onto the current orientation.
        - ``position_fixed_ori``: delta position only; orientation is frozen after
          the first command and held thereafter.
        - ``position_compliant_ori``: delta position only; orientation tracks the
          current EEF orientation (no orientation correction).

        Applies ``workspace_pose_limiter`` callback if configured.

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.
            command: Preprocessed command array (3 or 6 elements depending on mode).

        Returns:
            dict: ``{"target_pos": ..., "target_ori_mat": ...}`` with float32 arrays.
        """
        config = cls.configs[controller_id]
        arpath = cls.articulation_root_paths[controller_id]
        pos_relative, quat_relative = ControllableObjectViewAPI.get_link_relative_position_orientation(
            arpath, config["_task_space_link_name"]
        )

        if config["mode"] == "absolute_pose":
            target_pos = command[:3]
        else:
            dpos = command[:3]
            target_pos = pos_relative + dpos

        if config["mode"] == "position_fixed_ori":
            if cls.state[controller_id]["fixed_quat_target"] is None:
                cls.state[controller_id]["fixed_quat_target"] = (
                    quat_relative if (cls.goals[controller_id] is None) else cls.goals[controller_id]["target_quat"]
                )
            target_quat = cls.state[controller_id]["fixed_quat_target"]
        elif config["mode"] == "position_compliant_ori":
            target_quat = quat_relative
        elif config["mode"] in ("pose_absolute_ori", "absolute_pose"):
            target_quat = cb.T.axisangle2quat(command[3:6])
        else:
            dori = cb.T.quat2mat(cb.T.axisangle2quat(command[3:6]))
            target_quat = cb.T.mat2quat(dori @ cb.T.quat2mat(quat_relative))

        if config.get("workspace_pose_limiter", None) is not None:
            target_pos, target_quat = config["workspace_pose_limiter"](target_pos, target_quat)

        return dict(
            target_pos=cb.as_float32(target_pos),
            target_ori_mat=cb.as_float32(cb.T.quat2mat(target_quat)),
        )

    @classmethod
    def _update_goal_osc(cls, controller_id, command):
        """Compute and return the target EEF pose goal for an OSC controller.

        Mirrors ``_update_goal_ik`` in mode semantics but operates on a copy of
        the current EEF pose (to avoid aliasing issues with the ControllableObjectViewAPI
        cache). Also updates variable OSC gains from an embedded gain command if the
        controller is configured for variable gains (currently not supported).

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.
            command: Preprocessed command array (3 or 6 elements depending on mode).

        Returns:
            dict: ``{"target_pos": ..., "target_ori_mat": ...}`` with float32 arrays.
        """
        config = cls.configs[controller_id]
        arpath = cls.articulation_root_paths[controller_id]
        pos_relative_raw, quat_relative_raw = ControllableObjectViewAPI.get_link_relative_position_orientation(
            arpath, config["_task_space_link_name"]
        )
        pos_relative = cb.copy(pos_relative_raw)
        quat_relative = cb.copy(quat_relative_raw)

        if config["mode"] == "absolute_pose":
            target_pos = command[:3]
        else:
            dpos = command[:3]
            target_pos = pos_relative + dpos

        if config["mode"] == "position_fixed_ori":
            if cls.state[controller_id]["fixed_quat_target"] is None:
                cls.state[controller_id]["fixed_quat_target"] = (
                    quat_relative if (cls.goals[controller_id] is None) else cls.goals[controller_id]["target_quat"]
                )
            target_quat = cls.state[controller_id]["fixed_quat_target"]
        elif config["mode"] == "position_compliant_ori":
            target_quat = quat_relative
        elif config["mode"] in ("pose_absolute_ori", "absolute_pose"):
            target_quat = cb.T.axisangle2quat(command[3:6])
        else:
            dori = cb.T.quat2mat(cb.T.axisangle2quat(command[3:6]))
            target_quat = cb.T.mat2quat(dori @ cb.T.quat2mat(quat_relative))

        if config["workspace_pose_limiter"] is not None:
            target_pos, target_quat = config["workspace_pose_limiter"](target_pos, target_quat)

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
        """Apply type-specific preprocessing before base scaling is applied.

        For NullJointController, the command is replaced entirely by the stored
        default goal. For MultiFingerGripperController in non-independent modes,
        the command is broadcast to ``command_dim`` elements; the inverted flag
        reflects the command around the input limit midpoint.

        After type-specific handling, delegates to ``_preprocess_command_base``
        for clipping and input→output range mapping.

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.
            command: Raw command (scalar, list, or array) before any processing.

        Returns:
            Array: Processed command ready to be passed to ``_update_goal``.
        """
        ctype = cls.type_by_id[controller_id]

        if ctype == ControllerType.NullJointController:
            return cb.array(cls.state[controller_id]["default_goal"])

        if ctype == ControllerType.MultiFingerGripperController:
            config = cls.configs[controller_id]
            if config["mode"] != "independent":
                command = (
                    cb.array([command] * cls.command_dim[controller_id])
                    if type(command) in {int, float}
                    else cb.array([command[0]] * cls.command_dim[controller_id])
                )
            if config["inverted"]:
                command = config["command_input_limits"][1] - (command - config["command_input_limits"][0])

        return cls._preprocess_command_base(controller_id, command)

    @classmethod
    def _preprocess_command_base(cls, controller_id, command):
        """Clip the command to input limits and linearly rescale it to output limits.

        If ``command_input_limits`` is set, the command is clipped to that range.
        If both input and output limits are set, a linear affine transform maps
        the input range to the output range, with scaling and offset cached after
        the first call for efficiency.

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.
            command: Command value (scalar, list, or array).

        Returns:
            Array: Clipped and rescaled command.
        """
        config = cls.configs[controller_id]
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
                command = (command - config["command_input_transform"]) * config["command_scale_factor"] + config[
                    "command_output_transform"
                ]
        return command

    @classmethod
    def reverse_preprocess_command(cls, controller_id, processed_command):
        """Invert the input→output linear scaling to recover the original input-space command.

        Used by ``compute_no_op_action`` to express the current no-op goal as a
        command in the normalised input range, so it can be directly re-issued
        to ``update_goal`` without modification.

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.
            processed_command: Command in output (joint / task) space.

        Returns:
            Array: Command in input (normalised) space.
        """
        config = cls.configs[controller_id]
        if config["command_input_limits"] is not None and config["command_output_limits"] is not None:
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
            original_command = (processed_command - config["command_output_transform"]) / config[
                "command_scale_factor"
            ] + config["command_input_transform"]
        else:
            original_command = processed_command
        return original_command

    # -------------------------------------------------------------------------
    # Control computation
    # -------------------------------------------------------------------------
    # Clip, reset
    # -------------------------------------------------------------------------

    @classmethod
    def clip_control(cls, controller_id, control):
        """Clip a computed control signal to the configured hardware joint limits.

        For POSITION control type, clipping is only applied to DOFs that actually
        have hardware limits (``dof_has_limits``). For other control types, all
        DOFs are clipped unconditionally.

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.
            control: Computed control array of length ``control_dim``.

        Returns:
            Array: Control array with out-of-range values clamped (mutates and returns
            the input array).
        """
        config = cls.configs[controller_id]
        clipped_control = control.clip(
            config["control_limits"][cls.control_type[controller_id]][0][config["dof_idx"]],
            config["control_limits"][cls.control_type[controller_id]][1][config["dof_idx"]],
        )
        idx = (
            config["dof_has_limits"][config["dof_idx"]]
            if cls.control_type[controller_id] == ControlType.POSITION
            else [True] * cls.control_dim[controller_id]
        )
        control[idx] = clipped_control[idx]
        return control

    @classmethod
    def reset(cls, controller_id: str):
        """Reset the goal and any persistent runtime state for a controller.

        Clears the stored goal to None and resets type-specific state:
        - IK: resets the smoothing filter and clears ``fixed_quat_target``.
        - OSC: clears ``fixed_quat_target`` and variable gains.
        - MultiFingerGripper: resets the velocity filter and sets
          ``is_grasping`` to FALSE.

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.
        """
        cls.goals[controller_id] = None
        ctype = cls.type_by_id[controller_id]
        if ctype == ControllerType.InverseKinematicsController:
            if cls.state[controller_id]["control_filter"] is not None:
                cls.state[controller_id]["control_filter"].reset()
            cls.state[controller_id]["fixed_quat_target"] = None
        elif ctype == ControllerType.OperationalSpaceController:
            cls.state[controller_id]["fixed_quat_target"] = None
            cls._clear_variable_gains(controller_id=controller_id)
        elif ctype == ControllerType.MultiFingerGripperController:
            cls.state[controller_id]["vel_filter"].reset()
            cls.state[controller_id]["is_grasping"] = IsGraspingState.FALSE

    # -------------------------------------------------------------------------
    # Batching
    # -------------------------------------------------------------------------

    @classmethod
    def _step_batch_joint(cls, controller_ids):
        """Compute joint-level control outputs for a batch of joint controllers.

        Controllers are split into impedance and non-impedance groups:

        - Non-impedance: the stored target is returned directly after clipping.
        - Impedance: batched matrix operations compute the full impedance torque:
          ``u = M(q) * (kp * (q_target - q) - kd * q_dot) + gravity + cc``.
          Gravity and Coriolis/centrifugal forces are added when the respective
          compensation flags are set.

        Both groups write their results to ``controls[cid]``.

        Args:
            controller_ids (list[str]): IDs of JointController / NullJointController /
                HolonomicBaseJointController instances to step.

        Returns:
            list: Ordered control arrays, one per controller ID.
        """
        impedance_indices = []
        non_impedance_indices = []
        for idx, cid in enumerate(controller_ids):
            if cls.configs[cid]["use_impedances"]:
                impedance_indices.append(idx)
            else:
                non_impedance_indices.append(idx)

        results = [None] * len(controller_ids)

        for idx in non_impedance_indices:
            cid = controller_ids[idx]
            u = cb.copy(cls.goals[cid]["target"])
            u = cls.clip_control(cid, u)
            cls.controls[cid] = u
            results[idx] = u

        if impedance_indices:
            imp_cids = [controller_ids[idx] for idx in impedance_indices]
            N = len(imp_cids)
            dims = [cls.control_dim[cid] for cid in imp_cids]
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
                config = cls.configs[cid]
                d = dims[i]
                dof_idx = cls.dof_idx[cid]
                arpath = cls.articulation_root_paths[cid]

                targets[i, :d] = cls.goals[cid]["target"]
                motor_type = config["motor_type"]
                if motor_type == "position":
                    base_values[i, :d] = ControllableObjectViewAPI.get_joint_positions(arpath)[dof_idx]
                    gain[i, :d] = config["pos_kp"]
                    damping[i, :d] = config["pos_kd"]
                elif motor_type == "velocity":
                    base_values[i, :d] = ControllableObjectViewAPI.get_joint_velocities(arpath, estimate=True)[dof_idx]
                    gain[i, :d] = config["vel_kp"]
                else:
                    is_effort[i] = True
                velocities[i, :d] = ControllableObjectViewAPI.get_joint_velocities(arpath, estimate=True)[dof_idx]

                s = config["_mm_start_idx"]
                mm_full = ControllableObjectViewAPI.get_generalized_mass_matrices(arpath)
                mass_matrices[i, :d, :d] = mm_full[s:, s:][dof_idx][:, dof_idx]
                if config["use_gravity_compensation"]:
                    gravity[i, :d] = ControllableObjectViewAPI.get_gravity_compensation_forces(arpath)[dof_idx]
                if config["use_cc_compensation"]:
                    cc[i, :d] = ControllableObjectViewAPI.get_coriolis_and_centrifugal_compensation_forces(arpath)[
                        dof_idx
                    ]

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
                cls.controls[cid] = control
                results[idx] = control

        return results

    @classmethod
    def _step_batch_ik(cls, controller_ids):
        """Compute joint position targets for a batch of IK controllers.

        Gathers joint positions, Jacobians, EEF poses, and goal poses for all
        controllers, pads them to the maximum DOF dimension, then calls the
        registered ``compute_ik_qpos_batch`` JIT function (either PyTorch or NumPy)
        to solve the batch of Jacobian pseudo-inverse IK problems in one kernel call.

        After solving, each controller optionally applies a moving-average smoothing
        filter (``control_filter``) and, if ``use_impedances`` is set, converts the
        joint-position target to a torque via impedance control (same formula as
        ``_step_batch_joint`` impedance path).

        Reads state from ControllableObjectViewAPI using the precomputed Jacobian
        slice indices (``_jac_link_row``, ``_jac_col_start``, ``_jac_n_cols``) and
        mass-matrix start index (``_mm_start_idx``) stored in each controller's config.

        Args:
            controller_ids (list[str]): IDs of InverseKinematicsController instances to step.

        Returns:
            list: Ordered control arrays (joint positions or torques), one per controller ID.
        """
        N = len(controller_ids)
        dims = [cls.control_dim[cid] for cid in controller_ids]
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
            config = cls.configs[cid]
            d = dims[i]
            dof_idx = cls.dof_idx[cid]
            arpath = cls.articulation_root_paths[cid]

            q[i, :d] = ControllableObjectViewAPI.get_joint_positions(arpath)[dof_idx]
            jac_full = ControllableObjectViewAPI.get_relative_jacobian(arpath)
            j_eef[i, :, :d] = jac_full[
                config["_jac_link_row"], :, config["_jac_col_start"] : config["_jac_col_start"] + config["_jac_n_cols"]
            ][:, dof_idx]
            ee_pos_i, ee_quat = ControllableObjectViewAPI.get_link_relative_position_orientation(
                arpath, config["_task_space_link_name"]
            )
            ee_pos[i] = ee_pos_i
            ee_mat[i] = cb.as_float32(cb.T.quat2mat(ee_quat))
            goal_pos[i] = cls.goals[cid]["target_pos"]
            goal_ori_mat[i] = cls.goals[cid]["target_ori_mat"]
            q_lower[i, :d] = config["control_limits"][ControlType.get_type("position")][0][dof_idx]
            q_upper[i, :d] = config["control_limits"][ControlType.get_type("position")][1][dof_idx]

        target_batch = cb.get_custom_method("compute_ik_qpos_batch")(
            q=q,
            j_eef=j_eef,
            ee_pos=cb.as_float32(ee_pos),
            ee_mat=cb.as_float32(ee_mat),
            goal_pos=cb.as_float32(goal_pos),
            goal_ori_mat=cb.as_float32(goal_ori_mat),
            q_lower_limit=q_lower,
            q_upper_limit=q_upper,
        )

        results = []
        for i, cid in enumerate(controller_ids):
            d = dims[i]
            target_joint_pos = target_batch[i, :d]

            if cls.state[cid]["control_filter"] is not None:
                target_joint_pos = cls.state[cid]["control_filter"].estimate(target_joint_pos)

            config = cls.configs[cid]
            if config["use_impedances"]:
                arpath = cls.articulation_root_paths[cid]
                dof_idx = cls.dof_idx[cid]
                motor_type = config["motor_type"]
                if motor_type == "position":
                    base_value = ControllableObjectViewAPI.get_joint_positions(arpath)[dof_idx]
                    u = (target_joint_pos - base_value) * config["pos_kp"] + (
                        -ControllableObjectViewAPI.get_joint_velocities(arpath, estimate=True)[dof_idx]
                    ) * config["pos_kd"]
                elif motor_type == "velocity":
                    base_value = ControllableObjectViewAPI.get_joint_velocities(arpath, estimate=True)[dof_idx]
                    u = (target_joint_pos - base_value) * config["vel_kp"]
                else:
                    u = target_joint_pos
                s = config["_mm_start_idx"]
                mm_full = ControllableObjectViewAPI.get_generalized_mass_matrices(arpath)
                u = cb.get_custom_method("compute_joint_torques")(u, mm_full[s:, s:], dof_idx)
                if config["use_gravity_compensation"]:
                    u += ControllableObjectViewAPI.get_gravity_compensation_forces(arpath)[dof_idx]
                if config["use_cc_compensation"]:
                    u += ControllableObjectViewAPI.get_coriolis_and_centrifugal_compensation_forces(arpath)[dof_idx]
            else:
                u = target_joint_pos

            u = cls.clip_control(cid, u)
            cls.controls[cid] = u
            results.append(u)

        return results

    @classmethod
    def _step_batch_osc(cls, controller_ids):
        """Compute joint torques for a batch of Operational Space Controllers.

        Gathers all tensors needed for the OSC batch kernel: joint positions and
        velocities, mass matrices, Jacobians, EEF poses and velocities, goal poses,
        PD gains, null-space gains and rest poses, and base linear/angular velocities.

        If all controllers share the same ``decouple_pos_ori`` flag, a single
        vectorised ``compute_osc_torques_batch`` call covers the whole batch.
        If the batch is mixed, two sub-calls are made (one per flag value) and
        results are scattered back into the output buffer.

        Optional gravity and Coriolis/centrifugal compensation is added per-controller.
        All results are clipped and stored in ``controls[cid]``.

        Args:
            controller_ids (list[str]): IDs of OperationalSpaceController instances to step.

        Returns:
            list: Ordered torque control arrays, one per controller ID.
        """
        N = len(controller_ids)
        dims = [cls.control_dim[cid] for cid in controller_ids]
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
            config = cls.configs[cid]
            d = dims[i]
            dof_idx = cls.dof_idx[cid]
            arpath = cls.articulation_root_paths[cid]
            goal_dict = cls.goals[cid]

            q[i, :d] = ControllableObjectViewAPI.get_joint_positions(arpath)[dof_idx]
            qd[i, :d] = ControllableObjectViewAPI.get_joint_velocities(arpath, estimate=True)[dof_idx]
            s = config["_mm_start_idx"]
            mm_full = ControllableObjectViewAPI.get_generalized_mass_matrices(arpath)
            mm[i, :d, :d] = mm_full[s:, s:][dof_idx][:, dof_idx]
            jac_full = ControllableObjectViewAPI.get_relative_jacobian(arpath)
            j_eef_batch[i, :, :d] = jac_full[
                config["_jac_link_row"], :, config["_jac_col_start"] : config["_jac_col_start"] + config["_jac_n_cols"]
            ][:, dof_idx]
            ee_pos_i, ee_quat = ControllableObjectViewAPI.get_link_relative_position_orientation(
                arpath, config["_task_space_link_name"]
            )
            ee_pos_batch[i] = ee_pos_i
            ee_mat_batch[i] = cb.as_float32(cb.T.quat2mat(ee_quat))

            ee_lin_vel_batch[i] = cb.as_float32(
                ControllableObjectViewAPI.get_link_relative_linear_velocity(
                    arpath, config["_task_space_link_name"], estimate=True
                )
            )
            ee_ang_vel = ControllableObjectViewAPI.get_link_relative_angular_velocity(
                arpath, config["_task_space_link_name"], estimate=True
            )
            base_ang_vel = ControllableObjectViewAPI.get_relative_angular_velocity(arpath, estimate=True)
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

            base_lin_vel_batch[i] = cb.as_float32(
                ControllableObjectViewAPI.get_relative_linear_velocity(arpath, estimate=True)
            )
            base_ang_vel_batch[i] = cb.as_float32(base_ang_vel)

            decouple_flags.append(config["decouple_pos_ori"])

            if config["use_gravity_compensation"]:
                gravity[i, :d] = ControllableObjectViewAPI.get_gravity_compensation_forces(arpath)[dof_idx]
            if config["use_cc_compensation"]:
                cc_force[i, :d] = ControllableObjectViewAPI.get_coriolis_and_centrifugal_compensation_forces(arpath)[
                    dof_idx
                ]

        all_decouple = all(decouple_flags)
        no_decouple = not any(decouple_flags)

        if no_decouple or all_decouple:
            u = cb.get_custom_method("compute_osc_torques_batch")(
                q=q,
                qd=qd,
                mm=mm,
                j_eef=j_eef_batch,
                ee_pos=ee_pos_batch,
                ee_mat=ee_mat_batch,
                ee_lin_vel=ee_lin_vel_batch,
                ee_ang_vel_err=ee_ang_vel_err_batch,
                goal_pos=goal_pos_batch,
                goal_ori_mat=goal_ori_mat_batch,
                kp=kp_batch,
                kd=kd_batch,
                kp_null=kp_null_batch,
                kd_null=kd_null_batch,
                rest_qpos=rest_qpos_batch,
                max_dim=max_dim,
                decouple_pos_ori=all_decouple,
                base_lin_vel=base_lin_vel_batch,
                base_ang_vel=base_ang_vel_batch,
            )
        else:
            u = cb.zeros((N, max_dim))
            for flag_val in [False, True]:
                indices = [i for i in range(N) if decouple_flags[i] == flag_val]
                if not indices:
                    continue
                idx_arr = cb.as_int(cb.array(indices))
                u_group = cb.get_custom_method("compute_osc_torques_batch")(
                    q=q[idx_arr],
                    qd=qd[idx_arr],
                    mm=mm[idx_arr],
                    j_eef=j_eef_batch[idx_arr],
                    ee_pos=ee_pos_batch[idx_arr],
                    ee_mat=ee_mat_batch[idx_arr],
                    ee_lin_vel=ee_lin_vel_batch[idx_arr],
                    ee_ang_vel_err=ee_ang_vel_err_batch[idx_arr],
                    goal_pos=goal_pos_batch[idx_arr],
                    goal_ori_mat=goal_ori_mat_batch[idx_arr],
                    kp=kp_batch[idx_arr],
                    kd=kd_batch[idx_arr],
                    kp_null=kp_null_batch[idx_arr],
                    kd_null=kd_null_batch[idx_arr],
                    rest_qpos=rest_qpos_batch[idx_arr],
                    max_dim=max_dim,
                    decouple_pos_ori=flag_val,
                    base_lin_vel=base_lin_vel_batch[idx_arr],
                    base_ang_vel=base_ang_vel_batch[idx_arr],
                )
                for j_idx, orig_idx in enumerate(indices):
                    u[orig_idx] = u_group[j_idx]

        u = u + gravity + cc_force

        results = []
        for i, cid in enumerate(controller_ids):
            d = dims[i]
            control = u[i, :d]
            control = cls.clip_control(cid, control)
            cls.controls[cid] = control
            results.append(control)

        return results

    @classmethod
    def _step_batch_dd(cls, controller_ids):
        """Compute wheel velocity targets for a batch of Differential Drive controllers.

        Converts the ``[lin_vel, ang_vel]`` goal into per-wheel angular velocities
        using the standard kinematic equations:
        ``left = (lin_vel - ang_vel * half_axle) / wheel_radius``
        ``right = (lin_vel + ang_vel * half_axle) / wheel_radius``

        Results are clipped to hardware limits and stored in ``controls[cid]``.

        Args:
            controller_ids (list[str]): IDs of DifferentialDriveController instances to step.

        Returns:
            list: Ordered ``[left_vel, right_vel]`` control arrays, one per controller ID.
        """
        N = len(controller_ids)

        vels = cb.zeros((N, 2))
        wheel_radius = cb.zeros(N)
        half_axle = cb.zeros(N)
        for i, cid in enumerate(controller_ids):
            vels[i] = cls.goals[cid]["vel"]
            config = cls.configs[cid]
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
            cls.controls[cid] = control
            results.append(control)

        return results

    @classmethod
    def _step_batch_gripper(cls, controller_ids):
        """Compute finger position / velocity targets for a batch of gripper controllers.

        For each controller:
        - binary mode: maps the sign of the target command to open/closed joint
          limits (or custom ``open_qpos`` / ``closed_qpos`` if provided).
        - smooth mode: broadcasts a scalar target to all fingers.
        - independent mode: passes the target vector through unchanged.

        For velocity and torque motor types, commands that would drive joints
        beyond their soft limits are zeroed out to prevent windup.

        Also calls ``_update_grasping_state`` to update the ``is_grasping``
        classification based on current finger velocities and positions.

        Args:
            controller_ids (list[str]): IDs of MultiFingerGripperController instances to step.

        Returns:
            list: Ordered control arrays, one per controller ID.
        """
        results = []
        for cid in controller_ids:
            config = cls.configs[cid]
            arpath = cls.articulation_root_paths[cid]
            target = cls.goals[cid]["target"]
            joint_pos = ControllableObjectViewAPI.get_joint_positions(arpath)[cls.dof_idx[cid]]
            if config["mode"] == "binary":
                should_open = target[0] >= 0.0 if not config["inverted"] else target[0] > 0.0
                if should_open:
                    u = (
                        config["control_limits"][ControlType.get_type(config["motor_type"])][1][cls.dof_idx[cid]]
                        if config["open_qpos"] is None
                        else config["open_qpos"]
                    )
                else:
                    u = (
                        config["control_limits"][ControlType.get_type(config["motor_type"])][0][cls.dof_idx[cid]]
                        if config["closed_qpos"] is None
                        else config["closed_qpos"]
                    )
            else:
                u = cb.full((cls.control_dim[cid],), target[0]) if len(target) == 1 else target

            if config["motor_type"] in {"velocity", "torque"}:
                violate_upper_limit = (
                    joint_pos
                    > config["control_limits"][ControlType.POSITION][1][cls.dof_idx[cid]] - config["limit_tolerance"]
                )
                violate_lower_limit = (
                    joint_pos
                    < config["control_limits"][ControlType.POSITION][0][cls.dof_idx[cid]] + config["limit_tolerance"]
                )
                violation = cb.logical_or(violate_upper_limit * (u > 0), violate_lower_limit * (u < 0))
                u *= ~violation
            cls._update_grasping_state(controller_id=cid)

            control = cls.clip_control(cid, u)
            cls.controls[cid] = control
            results.append(control)
        return results

    # -------------------------------------------------------------------------
    # Step management
    # -------------------------------------------------------------------------

    @classmethod
    def step(cls):
        """Compute and deploy control outputs for all active controllers in one pass.

        For each controller type group (precomputed in ``ids_by_type`` at register
        time), active (non-disabled) controllers are stepped in batch via the
        type-specific ``_step_batch_*`` method. Results are deployed directly to
        ControllableObjectViewAPI using per-controller DOF indices — no
        intermediate accumulator buffer is needed.
        """
        for ctype, cids in cls.ids_by_type.items():
            active_cids = [cid for cid in cids if cid not in cls.disabled_controllers]
            if not active_cids:
                continue

            for cid in active_cids:
                if cls.goals[cid] is None:
                    cls.goals[cid] = cls.compute_no_op_goal(cid)

            if ctype in (
                ControllerType.JointController,
                ControllerType.NullJointController,
                ControllerType.HolonomicBaseJointController,
            ):
                results = cls._step_batch_joint(active_cids)
            elif ctype == ControllerType.InverseKinematicsController:
                results = cls._step_batch_ik(active_cids)
            elif ctype == ControllerType.OperationalSpaceController:
                results = cls._step_batch_osc(active_cids)
            elif ctype == ControllerType.DifferentialDriveController:
                results = cls._step_batch_dd(active_cids)
            elif ctype == ControllerType.MultiFingerGripperController:
                results = cls._step_batch_gripper(active_cids)
            else:
                raise ValueError(f"Unknown controller type: {ctype}")

            for cid, control in zip(active_cids, results):
                arpath = cls.articulation_root_paths[cid]
                dof_idx = cls.dof_idx[cid]
                ct = cls.control_type[cid]
                if ct == ControlType.POSITION:
                    ControllableObjectViewAPI.set_joint_position_targets(arpath, positions=control, indices=dof_idx)
                    ControllableObjectViewAPI.set_joint_velocity_targets(
                        arpath, velocities=cb.zeros(len(dof_idx)), indices=dof_idx
                    )
                elif ct == ControlType.VELOCITY:
                    ControllableObjectViewAPI.set_joint_velocity_targets(arpath, velocities=control, indices=dof_idx)
                elif ct == ControlType.EFFORT:
                    ControllableObjectViewAPI.set_joint_efforts(arpath, efforts=control, indices=dof_idx)

    # -------------------------------------------------------------------------
    # No-op goals / commands
    # -------------------------------------------------------------------------

    @classmethod
    def compute_no_op_goal(cls, controller_id: str):
        """Compute a goal that keeps the robot in its current state (no movement).

        The semantics depend on controller type:
        - JointController / Holonomic (position motor): hold current joint positions.
        - JointController / Holonomic (velocity/effort motor): zero velocity/effort.
        - NullJointController: use the stored ``default_goal``.
        - IK / OSC: hold the current EEF pose (read from ControllableObjectViewAPI).
        - DifferentialDrive: zero linear and angular velocity.
        - MultiFingerGripper (binary): sign matches current grasping state to avoid
          unintended open/close; (position) holds current finger positions; (velocity) zeros.

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.

        Returns:
            dict: Internal goal dict in the same format as the corresponding
            ``_update_goal_*`` method would return.
        """
        ctype = cls.type_by_id[controller_id]
        config = cls.configs[controller_id]

        if ctype in (ControllerType.JointController, ControllerType.HolonomicBaseJointController):
            if config["motor_type"] == "position":
                arpath = cls.articulation_root_paths[controller_id]
                target = ControllableObjectViewAPI.get_joint_positions(arpath)[cls.dof_idx[controller_id]]
            else:
                target = cb.zeros(cls.control_dim[controller_id])
            return dict(target=target)
        elif ctype == ControllerType.NullJointController:
            return dict(target=cls.state[controller_id]["default_goal"])
        elif ctype == ControllerType.InverseKinematicsController:
            arpath = cls.articulation_root_paths[controller_id]
            pos_rel, quat_rel = ControllableObjectViewAPI.get_link_relative_position_orientation(
                arpath, config["_task_space_link_name"]
            )
            return dict(
                target_pos=cb.as_float32(pos_rel),
                target_ori_mat=cb.as_float32(cb.T.quat2mat(quat_rel)),
            )
        elif ctype == ControllerType.OperationalSpaceController:
            arpath = cls.articulation_root_paths[controller_id]
            pos_rel, quat_rel = ControllableObjectViewAPI.get_link_relative_position_orientation(
                arpath, config["_task_space_link_name"]
            )
            return dict(
                target_pos=cb.as_float32(cb.copy(pos_rel)),
                target_ori_mat=cb.as_float32(cb.T.quat2mat(cb.copy(quat_rel))),
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
                    arpath = cls.articulation_root_paths[controller_id]
                    target = ControllableObjectViewAPI.get_joint_positions(arpath)[cls.dof_idx[controller_id]]
                elif config["motor_type"] == "velocity":
                    target = cb.zeros(cls.command_dim[controller_id])
                else:
                    raise ValueError("Cannot compute noop action for effort motor type.")
                if config["mode"] == "smooth":
                    target = cb.mean(target, dim=-1, keepdim=True)
            return dict(target=target)
        raise ValueError(f"Unknown controller type: {ctype}")

    @classmethod
    def compute_no_op_action(cls, controller_id: str):
        """Compute the action (in input-command space) required to maintain the current state.

        Ensures a no-op goal is set if none exists, then computes the corresponding
        command via ``_compute_no_op_command`` and maps it back to the normalised
        input range via ``reverse_preprocess_command``. The result is a PyTorch
        tensor that can be re-issued unchanged to produce no motion.

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.

        Returns:
            torch.Tensor: No-op action in input-command space.
        """
        if cls.goals[controller_id] is None:
            cls.goals[controller_id] = cls.compute_no_op_goal(controller_id=controller_id)
        command = cls._compute_no_op_command(controller_id=controller_id)
        return cb.to_torch(cls.reverse_preprocess_command(controller_id=controller_id, processed_command=command))

    @classmethod
    def _compute_no_op_command(cls, controller_id: str):
        """Compute the raw output-space command that corresponds to no motion.

        Called internally by ``compute_no_op_action``. Returns the command in the
        output (joint / task) space before the reverse-scaling step that maps it
        back to the input range.

        The command is read from the current joint or EEF state as appropriate for
        the controller type. For delta-command controllers, the no-op is zero delta.

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.

        Returns:
            Array: No-op command in output space (same units as ``command_output_limits``).
        """
        ctype = cls.type_by_id[controller_id]
        config = cls.configs[controller_id]

        if ctype == ControllerType.NullJointController:
            return cb.array([])
        elif ctype in (ControllerType.JointController,):
            arpath = cls.articulation_root_paths[controller_id]
            if config["motor_type"] == "position":
                if config["use_delta_commands"]:
                    return cb.zeros(cls.command_dim[controller_id])
                return ControllableObjectViewAPI.get_joint_positions(arpath)[cls.dof_idx[controller_id]]
            if config["motor_type"] == "velocity":
                if config["use_delta_commands"]:
                    return -ControllableObjectViewAPI.get_joint_velocities(arpath, estimate=True)[
                        cls.dof_idx[controller_id]
                    ]
                return cb.zeros(cls.command_dim[controller_id])
            raise ValueError("Cannot compute noop action for effort motor type.")
        elif ctype == ControllerType.HolonomicBaseJointController:
            return cb.zeros(cls.command_dim[controller_id])
        elif ctype == ControllerType.InverseKinematicsController:
            arpath = cls.articulation_root_paths[controller_id]
            pos_relative, quat_relative = ControllableObjectViewAPI.get_link_relative_position_orientation(
                arpath, config["_task_space_link_name"]
            )
            command = cb.zeros(6)
            mode = config["mode"]
            if mode == "absolute_pose":
                command[:3] = pos_relative
            if mode in ("pose_absolute_ori", "absolute_pose"):
                command[3:] = cb.T.quat2axisangle(quat_relative)
            return command
        elif ctype == ControllerType.OperationalSpaceController:
            arpath = cls.articulation_root_paths[controller_id]
            pos_relative, quat_relative = ControllableObjectViewAPI.get_link_relative_position_orientation(
                arpath, config["_task_space_link_name"]
            )
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
            arpath = cls.articulation_root_paths[controller_id]
            if config["motor_type"] == "position":
                command = ControllableObjectViewAPI.get_joint_positions(arpath)[cls.dof_idx[controller_id]]
            elif config["motor_type"] == "velocity":
                command = cb.zeros(cls.command_dim[controller_id])
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
    def dump_state(cls, controller_id: str):
        """Capture the current mutable state of a controller as a plain dict.

        The dict always contains ``goal_is_valid`` (bool) and ``goal`` (None or a
        dict of torch tensors). Type-specific additions:
        - IK: includes ``control_filter`` state if a smoothing filter is active.
        - MultiFingerGripper: includes ``vel_filter`` state.

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.

        Returns:
            dict: Serialisable state snapshot.
        """
        goal = cls.goals[controller_id]
        state = dict(
            goal_is_valid=goal is not None,
            goal=None if goal is None else {k: cb.to_torch(v) for k, v in goal.items()},
        )
        ctype = cls.type_by_id[controller_id]
        if ctype == ControllerType.InverseKinematicsController:
            if cls.state[controller_id]["control_filter"] is not None:
                state["control_filter"] = cls.state[controller_id]["control_filter"].dump_state(serialized=False)
        elif ctype == ControllerType.MultiFingerGripperController:
            state["vel_filter"] = cls.state[controller_id]["vel_filter"].dump_state(serialized=False)
        return state

    @classmethod
    def _load_state(cls, controller_id: str, state):
        """Restore the mutable state of a controller from a previously dumped dict.

        Converts goal tensor values from torch to the current compute backend format.
        For IK and OSC controllers, restores ``fixed_quat_target`` when the mode is
        ``position_fixed_ori``. Also restores the smoothing filter (IK) and velocity
        filter (gripper) if included in the state dict.

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.
            state (dict): State dict as returned by ``_dump_state``.
        """
        if state["goal"] is None:
            cls.goals[controller_id] = None
        else:
            goal = dict()
            for name, goal_state in state["goal"].items():
                if isinstance(goal_state, th.Tensor):
                    goal[name] = cb.from_torch(goal_state)
                else:
                    goal[name] = goal_state
            cls.goals[controller_id] = goal

        ctype = cls.type_by_id[controller_id]
        if ctype == ControllerType.InverseKinematicsController:
            if cls.goals[controller_id] is not None:
                if cls.configs[controller_id]["mode"] == "position_fixed_ori":
                    cls.state[controller_id]["fixed_quat_target"] = cls.goals[controller_id]["target_quat"]
                if cls.state[controller_id]["control_filter"] is not None:
                    cls.state[controller_id]["control_filter"].load_state(state["control_filter"], serialized=False)
        elif ctype == ControllerType.OperationalSpaceController:
            if cls.goals[controller_id] is not None and cls.configs[controller_id]["mode"] == "position_fixed_ori":
                cls.state[controller_id]["fixed_quat_target"] = cls.goals[controller_id]["target_quat"]
        elif ctype == ControllerType.MultiFingerGripperController:
            if cls.goals[controller_id] is not None:
                cls.state[controller_id]["vel_filter"].load_state(state["vel_filter"], serialized=False)

    @classmethod
    def load_state(cls, controller_id: str, state, serialized=False):
        """Public wrapper to restore controller state from a dict or flat tensor.

        When ``serialized=True``, first calls ``deserialize`` to reconstruct the
        state dict from a 1-D tensor, asserting that all values were consumed.
        Then delegates to ``_load_state``.

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.
            state (dict or torch.Tensor): State produced by a prior ``dump_state`` call.
            serialized (bool): Whether ``state`` is a flat tensor (True) or a dict (False).
        """
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
        """Flatten the state dict into a 1-D torch tensor for checkpoint storage.

        The layout is:
        ``[goal_is_valid (1)] + [goal_values (goal_dim)] + [filter_state (optional)]``

        For IK controllers, the smoothing filter state is appended if a filter exists.
        For gripper controllers, the velocity filter state is appended.
        When the goal is invalid, a zero tensor of ``goal_dim`` is used as placeholder.

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.
            state (dict): State dict as returned by ``_dump_state``.

        Returns:
            torch.Tensor: 1-D flat tensor suitable for concatenation with other state tensors.
        """
        goal_is_valid = state["goal_is_valid"]
        goal_state_flattened = (
            th.cat([goal_state.flatten() for goal_state in state["goal"].values()])
            if goal_is_valid
            else th.zeros(cls.configs[controller_id]["goal_dim"])
        )
        state_flat = th.cat([th.tensor([goal_is_valid]), goal_state_flattened])

        ctype = cls.type_by_id[controller_id]
        if ctype == ControllerType.InverseKinematicsController:
            state_flat = th.cat(
                [
                    state_flat,
                    (
                        th.tensor([])
                        if cls.state[controller_id]["control_filter"] is None
                        else cls.state[controller_id]["control_filter"].serialize(state=state["control_filter"])
                    ),
                ]
            )
        elif ctype == ControllerType.MultiFingerGripperController:
            state_flat = th.cat(
                [state_flat, cls.state[controller_id]["vel_filter"].serialize(state=state["vel_filter"])]
            )
        return state_flat

    @classmethod
    def deserialize(cls, controller_id: str, state):
        """Reconstruct the state dict from a flat 1-D tensor produced by ``serialize``.

        Reads the goal validity flag, then slices out each goal field according to
        the shapes in ``config["goal_shapes"]``. Appends optional filter state
        for IK and gripper controllers. Returns both the reconstructed dict and the
        total number of elements consumed from the input tensor.

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.
            state (torch.Tensor): Flat state tensor from a prior ``serialize`` call.

        Returns:
            tuple[dict, int]: ``(state_dict, elements_consumed)`` where
            ``elements_consumed`` equals ``state_size(controller_id)``.
        """
        goal_is_valid = bool(state[0])
        if goal_is_valid:
            idx = 1
            goal = dict()
            for key, shape in cls.configs[controller_id]["goal_shapes"].items():
                length = math.prod(shape)
                goal[key] = state[idx : idx + length].reshape(shape)
                idx += length
        else:
            goal = None
        state_dict = dict(goal_is_valid=goal_is_valid, goal=goal)
        idx = cls.configs[controller_id]["goal_dim"] + 1

        ctype = cls.type_by_id[controller_id]
        if ctype == ControllerType.InverseKinematicsController:
            if cls.state[controller_id]["control_filter"] is not None:
                state_dict["control_filter"], deserialized_items = cls.state[controller_id][
                    "control_filter"
                ].deserialize(state=state[idx:])
                idx += deserialized_items
        elif ctype == ControllerType.MultiFingerGripperController:
            state_dict["vel_filter"], deserialized_items = cls.state[controller_id]["vel_filter"].deserialize(
                state=state[idx:]
            )
            idx += deserialized_items
        return state_dict, idx

    # -------------------------------------------------------------------------
    # Property accessors
    # -------------------------------------------------------------------------

    @classmethod
    def _get_goal_shapes(cls, controller_id: str, command_dim: int, control_dim: int):
        """Return a dict mapping goal field names to their tensor shapes for serialisation.

        Shapes are used by ``serialize`` / ``deserialize`` to correctly flatten and
        slice the goal dict. The dict is also used to compute ``goal_dim`` stored in
        the config.

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.

        Returns:
            dict[str, tuple]: ``{field_name: shape_tuple}`` for each goal field.
        """
        ctype = cls.type_by_id[controller_id]
        if ctype in (
            ControllerType.JointController,
            ControllerType.NullJointController,
            ControllerType.HolonomicBaseJointController,
        ):
            return dict(target=(control_dim,))
        elif ctype == ControllerType.InverseKinematicsController:
            return dict(target_pos=(3,), target_ori_mat=(3, 3), target_quat=(4,))
        elif ctype == ControllerType.OperationalSpaceController:
            return dict(target_pos=(3,), target_ori_mat=(3, 3))
        elif ctype == ControllerType.DifferentialDriveController:
            return dict(vel=(2,))
        elif ctype == ControllerType.MultiFingerGripperController:
            return dict(target=(command_dim,))
        raise ValueError(f"Unknown controller type: {ctype}")

    @staticmethod
    def nums2array(nums, dim):
        """Convert a scalar, list, or array to a backend array of length ``dim``.

        - If ``nums`` is already a backend array, it is returned unchanged.
        - If ``nums`` is any other iterable, it is wrapped with ``cb.array``.
        - If ``nums`` is a scalar, ``cb.ones(dim) * nums`` is returned.
        - Strings are rejected (common misconfiguration).

        Args:
            nums: Input value — scalar, iterable, or backend array.
            dim (int): Target length when broadcasting a scalar.

        Returns:
            Array: Backend array of length ``dim``.

        Raises:
            TypeError: If ``nums`` is a string.
        """
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
    def get_state_size(cls, controller_id: str):
        """Return the total number of floats in the serialised state tensor.

        Equals ``goal_dim + 1`` (for the validity flag) plus optional filter sizes
        for IK (smoothing filter) and gripper (velocity filter) controllers.

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.

        Returns:
            int: Number of elements consumed by ``serialize`` / ``deserialize``.
        """
        size = cls.goal_dim[controller_id] + 1
        ctype = cls.type_by_id[controller_id]
        if ctype == ControllerType.InverseKinematicsController:
            control_filter = cls.state[controller_id]["control_filter"]
            size += 0 if control_filter is None else control_filter.state_size
        return size

    @classmethod
    def _compute_control_type(cls, ctype, config):
        """Compute the ControlType constant from a controller type and its processed config.

        This is the single source of truth for the POSITION / VELOCITY / EFFORT
        mapping. It is called once per controller during ``register()`` to populate
        ``control_type``, and also called directly from internal config-processing
        helpers (``_process_config_base``, ``_generate_default_command_output_limits``)
        where the dict has not yet been populated.

        Args:
            ctype (ControllerType): The controller's type enum value.
            config (dict): The (possibly in-progress) processed config dict for the controller.

        Returns:
            int: One of ``ControlType.POSITION``, ``ControlType.VELOCITY``, or
            ``ControlType.EFFORT``.
        """
        if ctype in _JOINT_TYPES:
            return (
                ControlType.EFFORT if config["use_impedances"] else ControlType.get_type(type_str=config["motor_type"])
            )
        elif ctype == ControllerType.OperationalSpaceController:
            return ControlType.EFFORT
        elif ctype == ControllerType.DifferentialDriveController:
            return ControlType.VELOCITY
        elif ctype == ControllerType.MultiFingerGripperController:
            return ControlType.get_type(type_str=config["motor_type"])
        raise ValueError(f"Unknown controller type: {ctype}")

    @classmethod
    def _command_dim_from_config(cls, controller_id: str, config: dict) -> int:
        """Compute command vector length from controller type and processed config.

        Used during registration to populate ``command_dim`` and inside
        ``_process_config_base`` before ``command_dim`` dict is populated.

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.
            config (dict): Processed (or in-progress) config dict.

        Returns:
            int: Expected command vector length.
        """
        ctype = cls.type_by_id[controller_id]
        if ctype == ControllerType.NullJointController:
            return 0
        elif ctype in (ControllerType.JointController, ControllerType.HolonomicBaseJointController):
            return len(config["dof_idx"])
        elif ctype == ControllerType.InverseKinematicsController:
            return IK_MODE_COMMAND_DIMS[config["mode"]]
        elif ctype == ControllerType.OperationalSpaceController:
            return config["command_dim"]
        elif ctype == ControllerType.DifferentialDriveController:
            return 2
        elif ctype == ControllerType.MultiFingerGripperController:
            return len(config["dof_idx"]) if config["mode"] == "independent" else 1
        raise ValueError(f"Unknown controller type: {ctype}")

    # -------------------------------------------------------------------------
    # Type-specific utilities
    # -------------------------------------------------------------------------

    @classmethod
    def is_grasping(cls, controller_id: str):
        """Return the current grasping state for gripper controllers.

        For non-gripper controllers, returns ``IsGraspingState.UNKNOWN`` (0)
        since grasping is not applicable.

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.

        Returns:
            IsGraspingState: TRUE (1), UNKNOWN (0), or FALSE (-1).
        """
        ctype = cls.type_by_id[controller_id]
        if ctype == ControllerType.MultiFingerGripperController:
            return cls.state[controller_id]["is_grasping"]
        return IsGraspingState.UNKNOWN

    @classmethod
    def update_default_goal(cls, controller_id: str, target):
        """Update the default hold-position goal for a NullJointController.

        Raises an AssertionError for any other controller type since only
        NullJointController uses a stored default goal.

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.
            target: New default goal array of length ``control_dim``.
        """
        assert cls.type_by_id[controller_id] == ControllerType.NullJointController
        assert (
            len(target) == cls.control_dim[controller_id]
        ), f"Default goal must be length: {cls.control_dim[controller_id]}, got length: {len(target)}"
        cls.state[controller_id]["default_goal"] = cb.array(target)

    @classmethod
    def _clear_variable_gains(cls, controller_id: str):
        """Reset variable OSC gains to None, clearing any gains set by a previous command.

        Called during ``reset`` for OSC controllers so that the next step will
        re-read gains from the command stream rather than using stale values.

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.
        """
        config = cls.configs[controller_id]
        if config["variable_kp"]:
            config["kp"] = None
        if config["variable_damping_ratio"]:
            config["damping_ratio"] = None
        if config["variable_kp_null"]:
            config["kp_null"] = None
            config["kd_null"] = None

    @classmethod
    def _update_variable_gains(cls, controller_id: str, gains):
        """Apply variable gains embedded in an OSC command to the controller config.

        Slices the ``gains`` array into ``kp`` (6 dims), ``damping_ratio`` (6 dims),
        and ``kp_null`` (control_dim dims) segments according to which are flagged as
        variable. Also recomputes ``kd_null = 2 * sqrt(kp_null)`` when ``kp_null``
        is variable.

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.
            gains: Gain array extracted from the trailing dimensions of the OSC command.
        """
        config = cls.configs[controller_id]
        idx = 0
        if config["variable_kp"]:
            config["kp"] = gains[:, idx : idx + 6]
            idx += 6
        if config["variable_damping_ratio"]:
            config["damping_ratio"] = gains[:, idx : idx + 6]
            idx += 6
        if config["variable_kp_null"]:
            config["kp_null"] = gains[:, idx : idx + cls.control_dim[controller_id]]
            config["kd_null"] = 2 * cb.sqrt(config["kp_null"])
            idx += cls.control_dim[controller_id]

    @classmethod
    def _update_grasping_state(cls, controller_id: str):
        """Update the ``is_grasping`` classification for a MultiFingerGripper controller.

        Uses a moving-average filter on finger joint velocities (``vel_filter``) to
        smooth out noise. The classification logic:

        - **UNKNOWN**: independent mode (per-finger control), or inconsistent finger
          commands, or position tolerance exceeds the limit tolerance threshold.
        - **UNKNOWN**: fingers are near their target (position motor: position error
          below threshold; velocity/torque motor: velocity command below threshold) —
          i.e. fingers are freely closing/opening without contact.
        - **TRUE**: fingers are away from both joint limits and not moving
          (velocity below VEL_TOLERANCE) — i.e. an object is blocking closure.
        - **FALSE**: all other cases (fingers at limit or moving freely).

        Args:
            controller_id (str): Unique identifier formatted as ``"robot_name:controller_name"``.
        """
        config = cls.configs[controller_id]
        arpath = cls.articulation_root_paths[controller_id]
        finger_vel = cls.state[controller_id]["vel_filter"].estimate(
            ControllableObjectViewAPI.get_joint_velocities(arpath, estimate=True)[cls.dof_idx[controller_id]]
        )

        if config["mode"] == "independent":
            is_grasping = IsGraspingState.UNKNOWN
        elif cls.controls[controller_id] is None:
            is_grasping = IsGraspingState.FALSE
        elif not cb.all(cls.controls[controller_id] == cls.controls[controller_id][0]):
            is_grasping = IsGraspingState.UNKNOWN
        elif not m.POS_TOLERANCE > config["limit_tolerance"]:
            is_grasping = IsGraspingState.UNKNOWN
        else:
            finger_pos = ControllableObjectViewAPI.get_joint_positions(arpath)[cls.dof_idx[controller_id]]

            if (
                config["motor_type"] == "position"
                and cb.abs(finger_pos - cls.controls[controller_id]).mean() < m.POS_TOLERANCE
            ):
                is_grasping = IsGraspingState.UNKNOWN
            elif (
                config["motor_type"] in {"velocity", "torque"}
                and cb.abs(cls.controls[controller_id]).mean() < m.VEL_TOLERANCE
            ):
                is_grasping = IsGraspingState.UNKNOWN
            else:
                min_pos = config["control_limits"][ControlType.POSITION][0][cls.dof_idx[controller_id]]
                max_pos = config["control_limits"][ControlType.POSITION][1][cls.dof_idx[controller_id]]
                finger_pos = finger_pos.clip(min_pos, max_pos)
                dist_from_lower_limit = finger_pos - min_pos
                dist_from_upper_limit = max_pos - finger_pos
                valid_grasp_pos = (
                    dist_from_lower_limit.mean() > m.POS_TOLERANCE or dist_from_upper_limit.mean() > m.POS_TOLERANCE
                )
                valid_grasp_vel = cb.all(cb.abs(finger_vel) < m.VEL_TOLERANCE)
                is_grasping = IsGraspingState.TRUE if valid_grasp_pos and valid_grasp_vel else IsGraspingState.FALSE

        cls.state[controller_id]["is_grasping"] = is_grasping


# =============================================================================
# JIT Functions: Joint Controller
# =============================================================================


@torch_compile
def _compute_joint_torques_torch(u: th.Tensor, mm: th.Tensor, dof_idx: th.Tensor):
    """Compute impedance joint torques via the mass matrix for PyTorch tensors.

    Selects the ``dof_idx × dof_idx`` sub-block from the full mass matrix and
    left-multiplies it by the PD error vector ``u`` to produce torque commands.

    Args:
        u (th.Tensor): PD error vector of shape ``(n_dof,)``.
        mm (th.Tensor): Full generalised mass matrix of shape ``(n_joints, n_joints)``,
            already sliced to exclude the virtual base DOFs.
        dof_idx (th.Tensor): Integer index tensor of length ``n_dof`` selecting the
            controlled joints from the full joint set.

    Returns:
        th.Tensor: Torque vector of shape ``(n_dof,)``.
    """
    dof_idxs_mat = th.meshgrid(dof_idx, dof_idx, indexing="xy")
    return mm[dof_idxs_mat] @ u


@jit(nopython=True)
def numba_ix(arr, rows, cols):
    """Numba-accelerated 2-D fancy indexing for NumPy arrays.

    Computes ``arr[rows][:, cols]`` without creating intermediate copies by
    building a flat 1-D index and using ``np.take`` — required because Numba's
    nopython mode does not support advanced NumPy indexing with two index arrays.

    Args:
        arr (np.ndarray): 2-D source array of shape ``(M, N)``.
        rows (np.ndarray): 1-D integer row indices.
        cols (np.ndarray): 1-D integer column indices.

    Returns:
        np.ndarray: Sub-matrix of shape ``(len(rows), len(cols))``.
    """
    one_d_index = np.zeros(len(rows) * len(cols), dtype=np.int32)
    for i, r in enumerate(rows):
        start = i * len(cols)
        one_d_index[start : start + len(cols)] = cols + arr.shape[1] * r
    arr_1d = arr.reshape((arr.shape[0] * arr.shape[1], 1))
    slice_1d = np.take(arr_1d, one_d_index)
    return slice_1d.reshape((len(rows), len(cols)))


@jit(nopython=True)
def _compute_joint_torques_numpy(u, mm, dof_idx):
    """Compute impedance joint torques via the mass matrix for NumPy arrays (Numba JIT).

    Equivalent to ``_compute_joint_torques_torch`` but operating on NumPy arrays
    using ``numba_ix`` for DOF sub-matrix extraction.

    Args:
        u (np.ndarray): PD error vector of shape ``(n_dof,)``.
        mm (np.ndarray): Generalised mass matrix of shape ``(n_joints, n_joints)``,
            already sliced to exclude virtual base DOFs.
        dof_idx (np.ndarray): Integer index array of length ``n_dof``.

    Returns:
        np.ndarray: Torque vector of shape ``(n_dof,)``.
    """
    return numba_ix(mm, dof_idx, dof_idx) @ u


add_compute_function(
    name="compute_joint_torques", np_function=_compute_joint_torques_numpy, th_function=_compute_joint_torques_torch
)


# =============================================================================
# JIT Functions: IK Controller (batched)
# =============================================================================


def _compute_ik_qpos_batch_torch(
    q: th.Tensor,
    j_eef: th.Tensor,
    ee_pos: th.Tensor,
    ee_mat: th.Tensor,
    goal_pos: th.Tensor,
    goal_ori_mat: th.Tensor,
    q_lower_limit: th.Tensor,
    q_upper_limit: th.Tensor,
):
    """Solve a batch of IK problems via Jacobian pseudo-inverse (PyTorch).

    Computes the Jacobian pseudo-inverse and applies one damped-least-squares step:
    ``q_target = q + J_pinv @ [pos_err; ori_err]``, then clips to joint limits.

    Args:
        q (th.Tensor): Current joint positions, shape ``(N, max_dim)``.
        j_eef (th.Tensor): EEF Jacobians, shape ``(N, 6, max_dim)``.
        ee_pos (th.Tensor): Current EEF positions in robot-root frame, shape ``(N, 3)``.
        ee_mat (th.Tensor): Current EEF rotation matrices, shape ``(N, 3, 3)``.
        goal_pos (th.Tensor): Target EEF positions, shape ``(N, 3)``.
        goal_ori_mat (th.Tensor): Target EEF rotation matrices, shape ``(N, 3, 3)``.
        q_lower_limit (th.Tensor): Lower joint limits, shape ``(N, max_dim)``.
        q_upper_limit (th.Tensor): Upper joint limits, shape ``(N, max_dim)``.

    Returns:
        th.Tensor: Target joint positions clipped to limits, shape ``(N, max_dim)``.
    """
    pos_err = goal_pos - ee_pos
    ori_err = TT.orientation_error(goal_ori_mat, ee_mat)
    err = th.cat([pos_err, ori_err], dim=-1)
    j_eef_pinv = th.linalg.pinv(j_eef)
    delta_j = (j_eef_pinv @ err.unsqueeze(-1)).squeeze(-1)
    target_joint_pos = q + delta_j
    return target_joint_pos.clip(min=q_lower_limit, max=q_upper_limit)


def _compute_ik_qpos_batch_numpy(
    q,
    j_eef,
    ee_pos,
    ee_mat,
    goal_pos,
    goal_ori_mat,
    q_lower_limit,
    q_upper_limit,
):
    """Solve a batch of IK problems via Jacobian pseudo-inverse (NumPy).

    NumPy equivalent of ``_compute_ik_qpos_batch_torch``. Identical algorithm
    using ``np.linalg.pinv`` and ``np.concatenate``.

    Args:
        q (np.ndarray): Current joint positions, shape ``(N, max_dim)``.
        j_eef (np.ndarray): EEF Jacobians, shape ``(N, 6, max_dim)``.
        ee_pos (np.ndarray): Current EEF positions, shape ``(N, 3)``.
        ee_mat (np.ndarray): Current EEF rotation matrices, shape ``(N, 3, 3)``.
        goal_pos (np.ndarray): Target EEF positions, shape ``(N, 3)``.
        goal_ori_mat (np.ndarray): Target EEF rotation matrices, shape ``(N, 3, 3)``.
        q_lower_limit (np.ndarray): Lower joint limits, shape ``(N, max_dim)``.
        q_upper_limit (np.ndarray): Upper joint limits, shape ``(N, max_dim)``.

    Returns:
        np.ndarray: Target joint positions clipped to limits, shape ``(N, max_dim)``.
    """
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
# JIT Functions: OSC Controller (batched)
# =============================================================================


def _compute_osc_torques_batch_torch(
    q,
    qd,
    mm,
    j_eef,
    ee_pos,
    ee_mat,
    ee_lin_vel,
    ee_ang_vel_err,
    goal_pos,
    goal_ori_mat,
    kp,
    kd,
    kp_null,
    kd_null,
    rest_qpos,
    max_dim,
    decouple_pos_ori,
    base_lin_vel,
    base_ang_vel,
):
    """Compute OSC joint torques for a batch of controllers (PyTorch).

    Implements the full Operational Space Control law with optional position/orientation
    decoupling and null-space posture control:

    1. Task-space PD error: ``task_err = kp * (goal - ee) + kd * vel_err``
    2. Operational-space inertia: ``M_ee = (J M^{-1} J^T)^{-1}``
    3. Task wrench: ``F = M_ee @ task_err`` (or decoupled pos/ori separately)
    4. Joint torques: ``u = J^T @ F``
    5. Null-space torques projected onto the null-space of J to pull joints toward
       ``rest_qpos`` without disturbing the EEF pose.

    Args:
        q: Joint positions ``(N, max_dim)``.
        qd: Joint velocities ``(N, max_dim)``.
        mm: Mass matrices ``(N, max_dim, max_dim)``.
        j_eef: EEF Jacobians ``(N, 6, max_dim)``.
        ee_pos: EEF positions ``(N, 3)``.
        ee_mat: EEF rotation matrices ``(N, 3, 3)``.
        ee_lin_vel: EEF linear velocities in robot frame ``(N, 3)``.
        ee_ang_vel_err: Angular velocity error (axis-angle) ``(N, 3)``.
        goal_pos: Target EEF positions ``(N, 3)``.
        goal_ori_mat: Target EEF rotation matrices ``(N, 3, 3)``.
        kp: Proportional gains ``(N, 6)``.
        kd: Derivative gains ``(N, 6)``.
        kp_null: Null-space proportional gains ``(N, max_dim)``.
        kd_null: Null-space derivative gains ``(N, max_dim)``.
        rest_qpos: Rest joint positions for null-space control ``(N, max_dim)``.
        max_dim (int): Padded joint dimension (for batching heterogeneous robots).
        decouple_pos_ori (bool): If True, compute separate inertia for position and orientation.
        base_lin_vel: Robot base linear velocity ``(N, 3)`` for velocity error computation.
        base_ang_vel: Robot base angular velocity ``(N, 3)`` for velocity error computation.

    Returns:
        Tensor: Joint torques ``(N, max_dim)``.
    """
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
    q,
    qd,
    mm,
    j_eef,
    ee_pos,
    ee_mat,
    ee_lin_vel,
    ee_ang_vel_err,
    goal_pos,
    goal_ori_mat,
    kp,
    kd,
    kp_null,
    kd_null,
    rest_qpos,
    max_dim,
    decouple_pos_ori,
    base_lin_vel,
    base_ang_vel,
):
    """Compute OSC joint torques for a batch of controllers (NumPy).

    NumPy equivalent of ``_compute_osc_torques_batch_torch``. Uses
    ``np.linalg.inv``, ``np.concatenate``, and ``np.expand_dims`` in place
    of their PyTorch counterparts. Identical OSC algorithm and argument layout.

    Args:
        q: Joint positions ``(N, max_dim)``.
        qd: Joint velocities ``(N, max_dim)``.
        mm: Mass matrices ``(N, max_dim, max_dim)``.
        j_eef: EEF Jacobians ``(N, 6, max_dim)``.
        ee_pos: EEF positions ``(N, 3)``.
        ee_mat: EEF rotation matrices ``(N, 3, 3)``.
        ee_lin_vel: EEF linear velocities in robot frame ``(N, 3)``.
        ee_ang_vel_err: Angular velocity error (axis-angle) ``(N, 3)``.
        goal_pos: Target EEF positions ``(N, 3)``.
        goal_ori_mat: Target EEF rotation matrices ``(N, 3, 3)``.
        kp: Proportional gains ``(N, 6)``.
        kd: Derivative gains ``(N, 6)``.
        kp_null: Null-space proportional gains ``(N, max_dim)``.
        kd_null: Null-space derivative gains ``(N, max_dim)``.
        rest_qpos: Rest joint positions ``(N, max_dim)``.
        max_dim (int): Padded joint dimension.
        decouple_pos_ori (bool): Separate position/orientation inertia if True.
        base_lin_vel: Robot base linear velocity ``(N, 3)``.
        base_ang_vel: Robot base angular velocity ``(N, 3)``.

    Returns:
        np.ndarray: Joint torques ``(N, max_dim)``.
    """
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
