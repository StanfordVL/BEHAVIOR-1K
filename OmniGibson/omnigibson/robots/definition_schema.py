"""
Robot definition dataclasses for schema-validated definitions.

Uses OmegaConf + dataclasses to define, parse, and enforce schema for robot YAML files.
The presence of optional capability sub-definitions (e.g., manipulation, two_wheel) indicates
the robot has that capability, replacing the old "capabilities" list approach.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


# === Capability-specific definitions ===


@dataclass
class EndEffectorDefinition:
    """Definition for a specific end effector (gripper, robotiq, allegro, etc.)"""

    model_name: Optional[str] = None
    eef_link_names: Optional[Dict[str, str]] = None
    finger_link_names: Optional[Dict[str, List[str]]] = None
    finger_joint_names: Optional[Dict[str, List[str]]] = None
    default_joint_pos: Optional[List[Union[float, str]]] = None
    teleop_rotation_offset: Optional[Dict[str, List[Union[float, str]]]] = None
    usd_path: Optional[str] = None
    urdf_path: Optional[str] = None
    curobo_path: Optional[str] = None
    disabled_collision_pairs: Optional[List[List[str]]] = None
    ag_start_points: Optional[List[List[Any]]] = None
    ag_end_points: Optional[List[List[Any]]] = None
    gripper_control_idx: Optional[List[int]] = None
    not_support_urdf: bool = False


@dataclass
class ManipulationDefinition:
    """Fields for manipulation robots (replaces is_manipulation capability)"""

    n_arms: int = 1
    arm_names: Optional[List[str]] = None  # defaults to ["0"], ["1"], etc. if None
    arm_link_names: Dict[str, List[str]] = field(default_factory=dict)
    arm_joint_names: Dict[str, List[str]] = field(default_factory=dict)
    eef_link_names: Optional[Dict[str, str]] = None
    finger_link_names: Optional[Dict[str, List[str]]] = None
    finger_joint_names: Optional[Dict[str, List[str]]] = None
    gripper_link_names: Optional[Dict[str, List[str]]] = None
    arm_workspace_range: Optional[Dict[str, List[float]]] = None
    teleop_rotation_offset: Optional[Dict[str, List[Union[float, str]]]] = None
    # For robots with swappable end effectors
    supported_end_effector: Optional[List[str]] = None
    eef_support_curobo_attached_object_link_names: Optional[List[str]] = None
    end_effectors: Optional[Dict[str, EndEffectorDefinition]] = None
    # Assisted grasp definition (when not using end_effector-specific)
    assisted_grasp_start_points: Optional[Dict[str, List[List[Any]]]] = None
    assisted_grasp_end_points: Optional[Dict[str, List[List[Any]]]] = None
    # Other manipulation flags
    add_combined_arm_control_idx: bool = False


@dataclass
class TwoWheelDefinition:
    """Fields for two-wheel differential drive robots (wheel-specific params only)"""

    wheel_radius: float = 0.0
    wheel_axle_length: float = 0.0


@dataclass
class HolonomicBaseDefinition:
    """Fields for holonomic base robots (e.g., R1, Tiago)"""

    base_footprint_link_name: str = ""
    floor_touching_base_link_names: List[str] = field(default_factory=list)
    force_sphere_wheel_approximation: bool = False


@dataclass
class ArticulatedTrunkDefinition:
    """Fields for robots with articulated trunk"""

    trunk_link_names: List[str] = field(default_factory=list)
    trunk_joint_names: List[str] = field(default_factory=list)


@dataclass
class ActiveCameraDefinition:
    """Fields for robots with controllable camera"""

    camera_joint_names: List[str] = field(default_factory=list)


@dataclass
class LocomotionDefinition:
    """Fields for generic locomotion robots (not two_wheel or holonomic)"""

    base_joint_names: List[str] = field(default_factory=list)


@dataclass
class MobileManipulationDefinition:
    """Fields for mobile manipulation robots"""

    tucked_default_joint_pos: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UntuckedArmPoseDefinition:
    """Fields for robots with multiple arm poses"""

    default_arm_poses: Dict[str, List[float]] = field(default_factory=dict)


# === Main Robot Definition ===


@dataclass
class RobotDefinition:
    """
    Root configuration for a robot, with optional capability sub-configs.

    The presence of a capability sub-config (e.g., manipulation is not None)
    indicates the robot has that capability.
    """

    # Common fields (always present)
    raw_controller_order: List[str] = field(default_factory=list)
    default_controllers: Dict[str, str] = field(default_factory=dict)
    default_joint_pos: Optional[List[Union[float, str]]] = None
    usd_path: Optional[str] = None
    urdf_path: Optional[str] = None
    curobo_path: Optional[str] = None
    disabled_collision_pairs: List[List[str]] = field(default_factory=list)
    disabled_collision_link_names: List[str] = field(default_factory=list)

    # Tiago-specific
    support_variant: bool = False
    manipulation_link_names: Optional[List[str]] = None
    visual_only_eef_links: bool = False

    # Fetch-specific (tucked joint pos at top level instead of in mobile_manipulation)
    tucked_default_joint_pos: Optional[List[Union[float, str]]] = None

    # Capability sub-definitions (presence indicates capability)
    manipulation: Optional[ManipulationDefinition] = None
    two_wheel: Optional[TwoWheelDefinition] = None
    holonomic_base: Optional[HolonomicBaseDefinition] = None
    locomotion: Optional[LocomotionDefinition] = None
    articulated_trunk: Optional[ArticulatedTrunkDefinition] = None
    active_camera: Optional[ActiveCameraDefinition] = None
    mobile_manipulation: Optional[MobileManipulationDefinition] = None
    untucked_arm_pose: Optional[UntuckedArmPoseDefinition] = None
