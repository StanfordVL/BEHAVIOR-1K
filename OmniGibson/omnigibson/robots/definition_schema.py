from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# === Capability-specific definitions ===
@dataclass
class EndEffectorDefinition:
    """Definition for a specific end effector (gripper, robotiq, allegro, etc.)"""

    model: str
    eef_link_names: Dict[str, str]
    finger_link_names: Dict[str, List[str]]
    finger_joint_names: Dict[str, List[str]]
    default_joint_pos: List[Any]
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
    arm_link_names: Optional[Dict[str, List[str]]] = None
    arm_joint_names: Optional[Dict[str, List[str]]] = None
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
    """Fields for holonomic base robots"""

    force_sphere_wheel_approximation: bool = False


@dataclass
class ArticulatedTrunkDefinition:
    """Fields for robots with articulated trunk"""

    trunk_joint_names: List[str]
    trunk_link_names: Optional[List[str]] = None


@dataclass
class ActiveCameraDefinition:
    """Fields for robots with controllable camera"""

    camera_joint_names: List[str]


@dataclass
class LocomotionDefinition:
    """Fields for locomotion robots"""

    base_joint_names: Optional[List[str]] = None
    floor_touching_base_link_names: Optional[List[str]] = None


@dataclass
class MobileManipulationDefinition:
    """Fields for mobile manipulation robots (tuck/untuck and multiple arm poses)"""

    untucked_default_joint_pos: List[Any]
    tucked_default_joint_pos: List[Any]
    # Multiple arm pose support (optional)
    default_arm_pose_key: Optional[str] = None
    default_arm_poses: Optional[Dict[str, List[Any]]] = None


# === Main Robot Definition ===
@dataclass
class RobotDefinition:
    """
    Root configuration for a robot, with optional capability sub-configs.
    """

    raw_controller_order: Optional[List[str]] = None
    default_controllers: Optional[Dict[str, str]] = None
    default_joint_pos: Optional[List[Any]] = None
    disabled_collision_pairs: Optional[List[List[str]]] = None
    disabled_collision_link_names: Optional[List[Any]] = None
    usd_path: Optional[str] = None
    urdf_path: Optional[str] = None
    curobo_path: Optional[str] = None
    base_footprint_link_name: Optional[str] = None
    visual_only_eef_links: bool = False
    linear_velocity_gain_for_primitives: Optional[float] = None
    angular_velocity_gain_for_primitives: Optional[float] = None

    manipulation: Optional[ManipulationDefinition] = None
    two_wheel: Optional[TwoWheelDefinition] = None
    holonomic_base: Optional[HolonomicBaseDefinition] = None
    locomotion: Optional[LocomotionDefinition] = None
    articulated_trunk: Optional[ArticulatedTrunkDefinition] = None
    active_camera: Optional[ActiveCameraDefinition] = None
    mobile_manipulation: Optional[MobileManipulationDefinition] = None
