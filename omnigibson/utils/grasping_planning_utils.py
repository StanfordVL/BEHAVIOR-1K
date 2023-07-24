import numpy as np
import random
from scipy.spatial.transform import Rotation

import omnigibson.utils.transform_utils as T
from omnigibson.object_states.open import _get_relevant_joints
from omnigibson.utils.constants import JointType, JointAxis

p = None
grasp_position_for_open_on_revolute_joint = None


PRISMATIC_JOINT_FRACTION_ACROSS_SURFACE_AXIS_BOUNDS = (0.2, 0.8)
GRASP_OFFSET = np.array([0, 0.05, -0.08])
OPEN_GRASP_OFFSET = np.array([0, 0.05, -0.12])  # 5cm back and 12cm up.

# def get_grasp_poses_for_object_sticky(target_obj, force_allow_any_extent=True):
#     bbox_center_in_world, bbox_quat_in_world, bbox_extent_in_base_frame, _ = target_obj.get_base_aligned_bbox(
#         visual=False
#     )

#     grasp_center_pos = bbox_center_in_world + np.array([0, 0, np.max(bbox_extent_in_base_frame) + 0.05])
#     towards_object_in_world_frame = bbox_center_in_world - grasp_center_pos
#     towards_object_in_world_frame /= np.linalg.norm(towards_object_in_world_frame)

#     grasp_quat = T.euler2quat([0, np.pi/2, 0])

#     grasp_pose = (grasp_center_pos, grasp_quat)
#     grasp_candidate = [(grasp_pose, towards_object_in_world_frame)]

#     return grasp_candidate

def get_grasp_poses_for_object_sticky(target_obj):
    """
    Target object to get a grasp pose for

    Args:
        target_object (StatefulObject): Object to get a grasp pose for
    
    Returns:
        Array of arrays: Array of possible grasp poses
    """
    bbox_center_in_world, bbox_quat_in_world, bbox_extent_in_base_frame, _ = target_obj.get_base_aligned_bbox(
        visual=False
    )

    # Pick an axis and a direction.
    approach_axis = random.choice([0, 1, 2])
    approach_direction = random.choice([-1, 1]) if approach_axis != 2 else 1
    constant_dimension_in_base_frame = approach_direction * bbox_extent_in_base_frame * np.eye(3)[approach_axis]
    randomizable_dimensions_in_base_frame = bbox_extent_in_base_frame - np.abs(constant_dimension_in_base_frame)
    random_dimensions_in_base_frame = np.random.uniform([-1, -1, 0], [1, 1, 1]) # note that we don't allow going below center
    grasp_center_in_base_frame = random_dimensions_in_base_frame * randomizable_dimensions_in_base_frame + constant_dimension_in_base_frame

    grasp_center_pos = T.mat2pose(
        T.pose2mat((bbox_center_in_world, bbox_quat_in_world)) @  # base frame to world frame
        T.pose2mat((grasp_center_in_base_frame, [0, 0, 0, 1]))    # grasp pose in base frame
    )[0] + np.array([0, 0, 0.02])
    towards_object_in_world_frame = bbox_center_in_world - grasp_center_pos
    towards_object_in_world_frame /= np.linalg.norm(towards_object_in_world_frame)

    # For the grasp, we want the X+ direction to be the direction of the object's surface.
    # The other two directions can be randomized.
    rand_vec = np.random.rand(3)
    rand_vec /= np.linalg.norm(rand_vec)
    grasp_x = towards_object_in_world_frame
    grasp_y = np.cross(rand_vec, grasp_x)
    grasp_y /= np.linalg.norm(grasp_y)
    grasp_z = np.cross(grasp_x, grasp_y)
    grasp_z /= np.linalg.norm(grasp_z)
    grasp_mat = np.array([grasp_x, grasp_y, grasp_z]).T
    grasp_quat = Rotation.from_matrix(grasp_mat).as_quat()

    grasp_pose = (grasp_center_pos, grasp_quat)
    grasp_candidate = [(grasp_pose, towards_object_in_world_frame)]

    return grasp_candidate

def get_grasp_position_for_open(robot, target_obj, should_open, link_id=None):
    # Pick a moving link of the object.
    relevant_joints_full = _get_relevant_joints(target_obj)
    relevant_joints = relevant_joints_full[1]

    # If a particular link ID was specified, filter our candidates down to that one.
    # if link_id is not None:
    #     relevant_joints = [ji for ji in relevant_joints if ji.jointIndex == link_id]

    if len(relevant_joints) == 0:
        raise ValueError("Cannot open/close object without relevant joints.")

    # Make sure what we got is an appropriately open/close joint.
    random.shuffle(relevant_joints)
    selected_joint = None
    for joint in relevant_joints:
        current_position = joint.get_state()[0][0]
        joint_range = joint.upper_limit - joint.lower_limit
        openness_fraction = (current_position - joint.lower_limit) / joint_range
        if (should_open and openness_fraction < 0.8) or (not should_open and openness_fraction > 0.05):
            selected_joint = joint

    if selected_joint is None:
        return None

    if selected_joint.joint_type == JointType.JOINT_REVOLUTE:
        return grasp_position_for_open_on_revolute_joint(robot, target_obj, selected_joint, should_open)
    elif selected_joint.joint_type == JointType.JOINT_PRISMATIC:
        return grasp_position_for_open_on_prismatic_joint(robot, target_obj, selected_joint, should_open)
    else:
        raise ValueError("Unknown joint type encountered while generating joint position.")
    

def grasp_position_for_open_on_prismatic_joint(robot, target_obj, relevant_joint, should_open):
    link_name = relevant_joint.body1.split("/")[-1]
    
    # Get the bounding box of the child link.
    (
        bbox_center_in_world,
        bbox_quat_in_world,
        bbox_extent_in_link_frame,
        _,
    ) = target_obj.get_base_aligned_bbox(link_name=link_name, visual=False)

    # Match the push axis to one of the bb axes.
    push_axis_idx = JointAxis.index(relevant_joint.axis)
    canonical_push_axis = np.eye(3)[push_axis_idx]
    # TODO: Need to figure out how to get the correct push direction.
    # canonical_push_direction = canonical_push_axis * np.sign(push_axis[push_axis_idx])
    canonical_push_direction = canonical_push_axis * 1

    # Pick the closer of the two faces along the push axis as our favorite.
    points_along_push_axis = (
        np.array([canonical_push_axis, -canonical_push_axis]) * bbox_extent_in_link_frame[push_axis_idx] / 2
    )
    (
        push_axis_closer_side_idx,
        center_of_selected_surface_along_push_axis,
        _,
    ) = get_closest_point_to_point_in_world_frame(
        points_along_push_axis, (bbox_center_in_world, bbox_quat_in_world), robot.get_position()
    )
    push_axis_closer_side_sign = 1 if push_axis_closer_side_idx == 0 else -1

    # Pick the other axes.
    all_axes = list(set(range(3)) - {push_axis_idx})
    x_axis_idx, y_axis_idx = tuple(sorted(all_axes))
    canonical_x_axis = np.eye(3)[x_axis_idx]
    canonical_y_axis = np.eye(3)[y_axis_idx]

    # Find the correct side of the lateral axis & go some distance along that direction.
    min_lateral_pos_wrt_surface_center = (canonical_x_axis + canonical_y_axis) * -bbox_extent_in_link_frame / 2
    max_lateral_pos_wrt_surface_center = (canonical_x_axis + canonical_y_axis) * bbox_extent_in_link_frame / 2
    diff_lateral_pos_wrt_surface_center = max_lateral_pos_wrt_surface_center - min_lateral_pos_wrt_surface_center
    sampled_lateral_pos_wrt_min = np.random.uniform(
        PRISMATIC_JOINT_FRACTION_ACROSS_SURFACE_AXIS_BOUNDS[0] * diff_lateral_pos_wrt_surface_center,
        PRISMATIC_JOINT_FRACTION_ACROSS_SURFACE_AXIS_BOUNDS[1] * diff_lateral_pos_wrt_surface_center,
    )
    lateral_pos_wrt_surface_center = min_lateral_pos_wrt_surface_center + sampled_lateral_pos_wrt_min
    grasp_position_in_bbox_frame = center_of_selected_surface_along_push_axis + lateral_pos_wrt_surface_center

    # Get the appropriate rotation
    # palm = canonical_push_axis * -push_axis_closer_side_sign
    # wrist = canonical_y_axis
    # lateral = np.cross(wrist, palm)
    # hand_orn_in_bbox_frame = get_hand_rotation_from_axes(lateral, wrist, palm)
    hand_orn_in_bbox_frame = T.euler2quat([0, 0, -np.pi])

    # Apply an additional random rotation along the face plane
    # random_rot = random.choice([np.pi, np.pi / 2, 0, -np.pi / 2])
    # hand_orn_in_bbox_frame = hand_orn_in_bbox_frame * Rotation.from_rotvec([0, 0, random_rot])

    # Finally apply our predetermined rotation around the X axis.
    # grasp_orn_in_bbox_frame = hand_orn_in_bbox_frame * Rotation.from_euler("X", -GRASP_ANGLE)
    # grasp_quat_in_bbox_frame = grasp_orn_in_bbox_frame.as_quat()
    grasp_quat_in_bbox_frame = hand_orn_in_bbox_frame

    # Now apply the grasp offset.
    # offset_in_bbox_frame = hand_orn_in_bbox_frame.apply(OPEN_GRASP_OFFSET)
    offset_grasp_pose_in_bbox_frame = (grasp_position_in_bbox_frame + OPEN_GRASP_OFFSET, grasp_quat_in_bbox_frame)
    offset_grasp_pose_in_world_frame = T.pose_transform(
        bbox_center_in_world, bbox_quat_in_world, *offset_grasp_pose_in_bbox_frame
    )

    # To compute the rotation position, we want to decide how far along the rotation axis we'll go.
    # target_joint_pos = relevant_joint.upper_limit if should_open else relevant_joint.lower_limit
    # [target_bid] = target_obj.get_body_ids()
    # current_joint_pos = get_joint_position(target_bid, link_id)
    # required_pos_change = target_joint_pos - current_joint_pos
    # push_vector_in_bbox_frame = canonical_push_direction * required_pos_change
    # target_hand_pos_in_bbox_frame = grasp_position_in_bbox_frame + push_vector_in_bbox_frame
    # target_hand_pos_in_world_frame = p.multiplyTransforms(
    #     bbox_center_in_world, bbox_quat_in_world, target_hand_pos_in_bbox_frame, grasp_quat_in_bbox_frame
    # )

    # Compute the approach direction.
    # approach_direction_in_world_frame = Rotation.from_quat(bbox_quat_in_world).apply(palm)

    # Decide whether a grasp is required. If approach direction and displacement are similar, no need to grasp.
    # grasp_required = np.dot(push_vector_in_bbox_frame, palm) < 0

    target_hand_pos_in_world_frame = None
    approach_direction_in_world_frame = None
    grasp_required = None

    return (
        offset_grasp_pose_in_world_frame,
        [target_hand_pos_in_world_frame],
        approach_direction_in_world_frame,
        relevant_joint,
        grasp_required,
    )


def get_closest_point_to_point_in_world_frame(
    vectors_in_arbitrary_frame, arbitrary_frame_to_world_frame, point_in_world
):
    vectors_in_world = np.array(
        [
            T.pose_transform(*arbitrary_frame_to_world_frame, vector, [0, 0, 0, 1])[0]
            for vector in vectors_in_arbitrary_frame
        ]
    )

    vector_distances_to_point = np.linalg.norm(vectors_in_world - np.array(point_in_world)[None, :], axis=1)
    closer_option_idx = np.argmin(vector_distances_to_point)
    vector_in_arbitrary_frame = vectors_in_arbitrary_frame[closer_option_idx]
    vector_in_world_frame = vectors_in_world[closer_option_idx]

    return closer_option_idx, vector_in_arbitrary_frame, vector_in_world_frame


