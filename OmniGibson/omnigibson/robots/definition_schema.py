from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


# === Capability-specific definitions ===


@dataclass
class EndEffectorDefinition:
    """Definition for a specific end effector (gripper, robotiq, allegro, etc.)"""

    model: Optional[str] = None
    eef_link_names: Optional[Dict[str, str]] = None
    finger_link_names: Optional[Dict[str, List[str]]] = None
    finger_joint_names: Optional[Dict[str, List[str]]] = None
    default_joint_pos: Optional[List[Any]] = None
    teleop_rotation_offset: Optional[Dict[str, List[Any]]] = None
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
    """Fields for manipulation robots"""

    n_arms: int = 1
    arm_names: Optional[List[str]] = None
    arm_link_names: Dict[str, List[str]] = field(default_factory=dict)
    arm_joint_names: Dict[str, List[str]] = field(default_factory=dict)
    eef_link_names: Optional[Dict[str, str]] = None
    finger_link_names: Optional[Dict[str, List[str]]] = None
    finger_joint_names: Optional[Dict[str, List[str]]] = None
    gripper_link_names: Optional[Dict[str, List[str]]] = None
    arm_workspace_range: Optional[Dict[str, List[Any]]] = None
    teleop_rotation_offset: Optional[Dict[str, List[Any]]] = None
    supported_end_effector: Optional[List[str]] = None
    eef_support_curobo_attached_object_link_names: Optional[List[str]] = None
    end_effectors: Optional[Dict[str, EndEffectorDefinition]] = None
    assisted_grasp_start_points: Optional[Dict[str, List[List[Any]]]] = None
    assisted_grasp_end_points: Optional[Dict[str, List[List[Any]]]] = None
    add_combined_arm_control_idx: bool = False
    manipulation_link_names: Optional[List[str]] = None


@dataclass
class TwoWheelDefinition:
    """Fields for two-wheel differential drive robots (wheel-specific params only)"""

    wheel_radius: float = 0.0
    wheel_axle_length: float = 0.0


@dataclass
class HolonomicBaseDefinition:
    """Fields for holonomic base robots (e.g., R1, Tiago)"""
    
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
    floor_touching_base_link_names: List[str] = field(default_factory=list)


@dataclass
class MobileManipulationDefinition:
    """Fields for mobile manipulation robots"""

    untucked_default_joint_pos: Optional[List[Any]] = None
    tucked_default_joint_pos: Optional[List[Any]] = None


@dataclass
class UntuckedArmPoseDefinition:
    """Fields for robots with multiple arm poses"""

    default_arm_pose_key: str = "vertical"
    default_arm_poses: Dict[str, List[Any]] = field(default_factory=dict)


# === Main Robot Definition ===


@dataclass
class RobotDefinition:
    """
    Root configuration for a robot, with optional capability sub-configs.
    """

    raw_controller_order: List[str] = field(default_factory=list)
    default_controllers: Dict[str, str] = field(default_factory=dict)
    default_joint_pos: Optional[List[Any]] = None
    usd_path: Optional[str] = None
    urdf_path: Optional[str] = None
    curobo_path: Optional[str] = None
    disabled_collision_pairs: List[List[str]] = field(default_factory=list)
    disabled_collision_link_names: List[str] = field(default_factory=list)
    base_footprint_link_name: Optional[str] = None
    visual_only_eef_links: bool = False

    manipulation: Optional[ManipulationDefinition] = None
    two_wheel: Optional[TwoWheelDefinition] = None
    holonomic_base: Optional[HolonomicBaseDefinition] = None
    locomotion: Optional[LocomotionDefinition] = None
    articulated_trunk: Optional[ArticulatedTrunkDefinition] = None
    active_camera: Optional[ActiveCameraDefinition] = None
    mobile_manipulation: Optional[MobileManipulationDefinition] = None
    untucked_arm_pose: Optional[UntuckedArmPoseDefinition] = None
