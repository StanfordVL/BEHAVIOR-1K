# import numpy as np
# from scipy.spatial.transform import Rotation

# from igibson.objects.articulated_object import URDFObject
# from igibson.robots import behavior_robot

# GRASP_ANGLE = 0.35  # 20 degrees

# def get_grasp_poses_for_object_sticky(robot, target_obj: URDFObject, force_allow_any_extent=True):
#     bbox_center_in_world, bbox_quat_in_world, bbox_extent_in_base_frame, _ = target_obj.get_base_aligned_bounding_box(
#         visual=False
#     )

#     grasp_center_pos = bbox_center_in_world + np.array([0, 0, np.max(bbox_extent_in_base_frame)])
#     towards_object_in_world_frame = bbox_center_in_world - grasp_center_pos
#     towards_object_in_world_frame /= np.linalg.norm(towards_object_in_world_frame)

#     # The default angle is hand-downwards. We can rotate this later if needed.
#     uncorrected_grasp_quat = behavior_robot.RIGHT_HAND_LOC_POSE_TRACKED[1]
#     corrected_grasp_quat = (
#         Rotation.from_quat(uncorrected_grasp_quat) * Rotation.from_euler("X", -GRASP_ANGLE)
#     ).as_quat()

#     grasp_pose = (grasp_center_pos, corrected_grasp_quat)
#     grasp_candidate = [(grasp_pose, towards_object_in_world_frame)]

#     return grasp_candidate