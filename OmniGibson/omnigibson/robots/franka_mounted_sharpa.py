import os
import torch as th
from functools import cached_property

from omnigibson.robots.franka import FrankaPanda
from omnigibson.robots.manipulation_robot import GraspingPoint


# ============================================================================
# Base class for both left and right Franka+SharpaWave variants
# ============================================================================

class _FrankaMountedSharpaBase(FrankaPanda):
    """
    Base class for Franka Emika Panda mounted with a SharpaWave dexterous hand.
    Subclasses must set `_hand_side` ('right' or 'left') before calling __init__.
    """

    # Subclasses override this
    _hand_side = None  # 'right' or 'left'

    def __init__(
        self,
        # Shared kwargs in hierarchy
        name,
        relative_prim_path=None,
        scale=None,
        visible=True,
        fixed_base=False,
        visual_only=False,
        self_collisions=True,
        link_physics_materials=None,
        load_config=None,
        # Unique to USDObject hierarchy
        abilities=None,
        # Unique to ControllableObject hierarchy
        control_freq=None,
        controller_config=None,
        action_type="continuous",
        action_normalize=True,
        reset_joint_pos=None,
        # Unique to BaseRobot
        obs_modalities=("rgb", "proprio"),
        include_sensor_names=None,
        exclude_sensor_names=None,
        proprio_obs="default",
        sensor_config=None,
        # Unique to ManipulationRobot
        grasping_mode="physical",
        disable_grasp_handling=False,
        finger_static_friction=None,
        finger_dynamic_friction=None,
        **kwargs,
    ):
        side = self._hand_side
        assert side in ("right", "left"), f"_hand_side must be 'right' or 'left', got {side!r}"

        # SharpaWave hand configuration (parameterized by side)
        self.end_effector = f"sharpa_{side}"
        self._model_name = f"franka_mounted_sharpa_{side}"
        self._gripper_control_idx = th.arange(7, 29)  # 22 hand joints after 7 arm joints
        self._eef_link_names = f"{side}_hand_C_MC"
        self._finger_link_names = [
            f"{side}_thumb_PP", f"{side}_thumb_DP", f"{side}_thumb_elastomer",
            f"{side}_index_PP", f"{side}_index_MP", f"{side}_index_DP", f"{side}_index_elastomer",
            f"{side}_middle_PP", f"{side}_middle_MP", f"{side}_middle_DP", f"{side}_middle_elastomer",
            f"{side}_ring_PP", f"{side}_ring_MP", f"{side}_ring_DP", f"{side}_ring_elastomer",
            f"{side}_pinky_PP", f"{side}_pinky_MP", f"{side}_pinky_DP", f"{side}_pinky_elastomer",
        ]
        self._finger_joint_names = [
            f"{side}_thumb_CMC_FE", f"{side}_thumb_CMC_AA",
            f"{side}_thumb_MCP_FE", f"{side}_thumb_MCP_AA", f"{side}_thumb_IP",
            f"{side}_index_MCP_FE", f"{side}_index_MCP_AA", f"{side}_index_PIP", f"{side}_index_DIP",
            f"{side}_middle_MCP_FE", f"{side}_middle_MCP_AA", f"{side}_middle_PIP", f"{side}_middle_DIP",
            f"{side}_ring_MCP_FE", f"{side}_ring_MCP_AA", f"{side}_ring_PIP", f"{side}_ring_DIP",
            f"{side}_pinky_CMC", f"{side}_pinky_MCP_FE", f"{side}_pinky_MCP_AA",
            f"{side}_pinky_PIP", f"{side}_pinky_DIP",
        ]
        # 7 arm joints + 22 hand joints (all at 0 for hand)
        self._default_robot_model_joint_pos = th.cat((
            th.tensor([0.00, -1.3, 0.00, -2.87, 0.00, 2.00, 0.75]),
            th.zeros(22),
        ))
        self._teleop_rotation_offset = th.tensor([-1, 0, 0, 0])
        self._ag_start_points = [
            GraspingPoint(link_name=f"{side}_thumb_elastomer", position=th.tensor([0.0, 0.0, 0.0])),
        ]
        self._ag_end_points = [
            GraspingPoint(link_name=f"{side}_index_elastomer", position=th.tensor([0.0, 0.0, 0.0])),
            GraspingPoint(link_name=f"{side}_middle_elastomer", position=th.tensor([0.0, 0.0, 0.0])),
        ]

        # Skip FrankaPanda.__init__'s end_effector logic -- go directly to its parent
        super(FrankaPanda, self).__init__(
            name=name,
            relative_prim_path=relative_prim_path,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            self_collisions=self_collisions,
            link_physics_materials=link_physics_materials,
            load_config=load_config,
            abilities=abilities,
            control_freq=control_freq,
            controller_config=controller_config,
            action_type=action_type,
            action_normalize=action_normalize,
            reset_joint_pos=reset_joint_pos,
            obs_modalities=obs_modalities,
            include_sensor_names=include_sensor_names,
            exclude_sensor_names=exclude_sensor_names,
            proprio_obs=proprio_obs,
            sensor_config=sensor_config,
            grasping_mode=grasping_mode,
            grasping_direction="upper",
            disable_grasp_handling=disable_grasp_handling,
            finger_static_friction=finger_static_friction,
            finger_dynamic_friction=finger_dynamic_friction,
            **kwargs,
        )

        # Our converted USD has ghost xformOp attributes (empty typeName), so we must
        # force _set_xform_properties() to run during _post_load.  robot_base forces
        # xform_props_pre_loaded=True, which skips that step; override it here.
        self._xform_props_pre_loaded = False

    def _initialize(self):
        super()._initialize()
        # ManipulationRobot._initialize() hides the EEF link (visible=False) because
        # standard Franka's eef_link is a virtual link with no geometry.  For SharpaWave,
        # the EEF link IS the physical palm ({side}_hand_C_MC) and must be visible.
        for arm in self.arm_names:
            self._links[self.eef_link_names[arm]].visible = True

    @property
    def model_name(self):
        return self._model_name

    @property
    def discrete_action_list(self):
        raise NotImplementedError()

    def _create_discrete_action_space(self):
        raise ValueError(f"{self.__class__.__name__} does not support discrete actions!")

    @property
    def _raw_controller_order(self):
        return [f"arm_{self.default_arm}", f"gripper_{self.default_arm}"]

    @property
    def _default_controllers(self):
        controllers = super()._default_controllers
        controllers[f"arm_{self.default_arm}"] = "InverseKinematicsController"
        controllers[f"gripper_{self.default_arm}"] = "MultiFingerGripperController"
        return controllers

    @property
    def _default_joint_pos(self):
        return self._default_robot_model_joint_pos

    @cached_property
    def arm_link_names(self):
        return {self.default_arm: [f"panda_link{i}" for i in range(8)]}

    @cached_property
    def arm_joint_names(self):
        return {self.default_arm: [f"panda_joint{i + 1}" for i in range(7)]}

    @cached_property
    def eef_link_names(self):
        return {self.default_arm: self._eef_link_names}

    @cached_property
    def finger_link_names(self):
        return {self.default_arm: self._finger_link_names}

    @cached_property
    def finger_joint_names(self):
        return {self.default_arm: self._finger_joint_names}

    @property
    def curobo_path(self):
        return None

    @property
    def _assisted_grasp_start_points(self):
        return None

    @property
    def _assisted_grasp_end_points(self):
        return None


# ============================================================================
# Concrete classes
# ============================================================================

# Resolve paths relative to the BEHAVIOR-1K root
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


class FrankaMountedSharpaRight(_FrankaMountedSharpaBase):
    """Franka Emika Panda mounted on a custom chassis with a SharpaWave right hand."""

    _hand_side = "right"

    @property
    def usd_path(self):
        return os.path.join(
            _REPO_ROOT, "datasets/objects/robot/franka_mounted_sharpa_right/usd/franka_mounted_sharpa_right.usda"
        )

    @property
    def urdf_path(self):
        return os.path.join(
            _REPO_ROOT, "datasets/omnigibson-robot-assets/models/franka/franka_mounted_sharpa_right/urdf/franka_mounted_sharpa_right.urdf"
        )


class FrankaMountedSharpaLeft(_FrankaMountedSharpaBase):
    """Franka Emika Panda mounted on a custom chassis with a SharpaWave left hand."""

    _hand_side = "left"

    @property
    def usd_path(self):
        return os.path.join(
            _REPO_ROOT, "datasets/objects/robot/franka_mounted_sharpa_left/usd/franka_mounted_sharpa_left.usda"
        )

    @property
    def urdf_path(self):
        return os.path.join(
            _REPO_ROOT, "datasets/omnigibson-robot-assets/models/franka/franka_mounted_sharpa_left/urdf/franka_mounted_sharpa_left.urdf"
        )
