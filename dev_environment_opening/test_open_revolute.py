import yaml
import numpy as np
import argparse

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, UndoableContext
from omnigibson.objects.primitive_object import PrimitiveObject
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.utils.grasping_planning_utils import get_grasp_position_for_open

import cProfile, pstats, io
import time
import os
import argparse
    

def pause(time):
    for _ in range(int(time*100)):
        og.sim.step()

def replay_controller(env, filename):
    actions = yaml.load(open(filename, "r"), Loader=yaml.FullLoader)
    for action in actions:
        env.step(action)

def execute_controller(ctrl_gen, env, filename=None):
    actions = []
    for action in ctrl_gen:
        env.step(action)
        actions.append(action.tolist())
    if filename is not None:
        with open(filename, "w") as f:
            yaml.dump(actions, f)

def main():
    # Load the config
    config_filename = "test_tiago.yaml"
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["load_object_categories"] = ["floors"]

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    controller = StarterSemanticActionPrimitives(None, scene, robot)

    # open_obj = DatasetObject(
    #     name="fridge",
    #     category="fridge",
    #     model="dszchb",
    #     scale=0.7
    # )

    open_obj = DatasetObject(
        name="bottom_cabinet",
        category="bottom_cabinet",
        model="bamfsz",
        scale=0.7
    )

    og.sim.import_object(open_obj)
    open_obj.set_position_orientation([-1.2, -0.4, 0.5], T.euler2quat([0, 0, np.pi/2]))
    og.sim.step()

    def set_start_pose():
        reset_pose_tiago = np.array([
            -1.78029833e-04,  3.20231302e-05, -1.85759447e-07, -1.16488536e-07,
            4.55182843e-08,  2.36128806e-04,  1.50000000e-01,  9.40000000e-01,
            -1.10000000e+00,  0.00000000e+00, -0.90000000e+00,  1.47000000e+00,
            0.00000000e+00,  2.10000000e+00,  2.71000000e+00,  1.50000000e+00,
            1.71000000e+00,  1.30000000e+00, -1.57000000e+00, -1.40000000e+00,
            1.39000000e+00,  0.00000000e+00,  0.00000000e+00,  4.50000000e-02,
            4.50000000e-02,  4.50000000e-02,  4.50000000e-02,
        ])
        robot.set_joint_positions(reset_pose_tiago)
        og.sim.step()

    def test_open_no_navigation():
        # Need to set start pose to reset_hand because default tuck pose for Tiago collides with itself
        set_start_pose()
        pose = controller._get_robot_pose_from_2d_pose([-1.0, -0.5, np.pi/2])
        robot.set_position_orientation(*pose)
        og.sim.step()
        get_grasp_position_for_open(robot, open_obj, True)
        # execute_controller(controller._open_or_close(cabinet), env)

    def test_open():
        set_start_pose()
        # pose_2d = [-0.231071, -0.272773, 2.55196]

        # # pose_2d = [-0.282843, 0.297682, -3.07804]
        # pose = controller._get_robot_pose_from_2d_pose(pose_2d)
        # robot.set_position_orientation(*pose)
        # og.sim.step()

        # joint_pos = [0.0133727 ,0.216775 ,0.683931 ,2.04371 ,1.88204 ,0.720747 ,1.23276 ,1.72251]
        # control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx["left"]])
        # robot.set_joint_positions(joint_pos, control_idx)
        # og.sim.step()
        # pause(100)
        execute_controller(controller._open_or_close(open_obj, True), env)

    markers = []
    for i in range(20):
        marker = PrimitiveObject(
            prim_path=f"/World/test_{i}",
            name=f"test_{i}",
            primitive_type="Cube",
            size=0.07,
            visual_only=True,
            rgba=[1.0, 0, 0, 1.0])
        markers.append(marker)
        og.sim.import_object(marker)
    
    from omnigibson.object_states.open import _get_relevant_joints
    controller.markers = markers
    # j = _get_relevant_joints(open_obj)[1][0]
    # j.set_pos(0.5)
    # pause(2)
    test_open()
    return

    # markers = []
    # for i in range(20):
    #     marker = PrimitiveObject(
    #         prim_path=f"/World/test_{i}",
    #         name=f"test_{i}",
    #         primitive_type="Cube",
    #         size=0.07,
    #         visual_only=True,
    #         rgba=[1.0, 0, 0, 1.0])
    #     markers.append(marker)
    #     og.sim.import_object(marker)

    def set_marker(position, idx):
        markers[idx].set_position(position)
        og.sim.step()

    def get_closest_point_to_point_in_world_frame(vectors_in_arbitrary_frame, arbitrary_frame_to_world_frame, point_in_world):
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
    
    def get_quaternion_between_vectors(v1, v2):
        """
        Get the quaternion between two vectors.

        Args:
            v1: The first vector.
            v2: The second vector.

        Returns:
            The quaternion between the two vectors.
        """
        q = np.cross(v1, v2).tolist()
        w = np.sqrt((np.linalg.norm(v1) ** 2) * (np.linalg.norm(v2) ** 2)) + np.dot(v1, v2)
        # np.sqrt((np.linalg.norm(a) ^ 2) * (np.linalg.norm(b) ** 2)) + np.dot(a, b)
        q.append(w)
        q = np.array(q) / np.linalg.norm(q)
        return q
    
    def rotate_point_around_axis(point_wrt_arbitrary_frame, arbitrary_frame_wrt_origin, joint_axis, yaw_change):
        # grasp_pose_in_bbox_frame, bbox_wrt_origin, joint_axis, partial_yaw_change
        # rotation_to_joint = get_quaternion_between_vectors([1, 0, 0], joint_axis)
        # rotation = T.quat_multiply(rotation_to_joint, T.euler2quat([yaw_change, 0, 0]))
        # rotation = T.quat_multiply(rotation, T.quat_inverse(rotation_to_joint))
        rotation = R.from_rotvec(joint_axis * yaw_change).as_quat()
        origin_wrt_arbitrary_frame = T.invert_pose_transform(*arbitrary_frame_wrt_origin)

        pose_in_origin_frame = T.pose_transform(*arbitrary_frame_wrt_origin, *point_wrt_arbitrary_frame)
        rotated_pose_in_origin_frame = T.pose_transform([0, 0, 0], rotation, *pose_in_origin_frame)
        rotated_pose_in_arbitrary_frame = T.pose_transform(*origin_wrt_arbitrary_frame, *rotated_pose_in_origin_frame)
        return rotated_pose_in_arbitrary_frame
    
    def get_orientation_facing_vector_with_random_yaw(vector):
        forward = vector / np.linalg.norm(vector)
        rand_vec = np.random.rand(3)
        rand_vec /= np.linalg.norm(3)
        side = np.cross(rand_vec, forward)
        side /= np.linalg.norm(3)
        up = np.cross(forward, side)
        assert np.isclose(np.linalg.norm(up), 1)
        rotmat = np.array([forward, side, up]).T
        return R.from_matrix(rotmat).as_quat()

    ########################################################

    # from omnigibson.objects.primitive_object import PrimitiveObject
    # import numpy as np
    import random
    from scipy.spatial.transform import Rotation

    # import omnigibson.utils.transform_utils as T
    from omnigibson.object_states.open import _get_relevant_joints
    from omnigibson.utils.constants import JointType, JointAxis
    from omni.isaac.core.utils.rotations import gf_quat_to_np_array
    from scipy.spatial.transform import Rotation as R
    from math import ceil

    p = None
    grasp_position_for_open_on_revolute_joint = None
    should_open = True
    PRISMATIC_JOINT_FRACTION_ACROSS_SURFACE_AXIS_BOUNDS = (0.2, 0.8)
    GRASP_OFFSET = np.array([0, 0.05, -0.08])
    OPEN_GRASP_OFFSET = np.array([0, 0.05, -0.12])  # 5cm back and 12cm up.
    ROTATION_ARC_SEGMENT_LENGTHS = 0.1

    # Pick a moving link of the object.
    relevant_joints_full = _get_relevant_joints(open_obj)
    relevant_joints = relevant_joints_full[1]

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
    
    robot = robot
    target_obj = open_obj
    relevant_joint = selected_joint
    should_open = True
    REVOLUTE_JOINT_FRACTION_ACROSS_SURFACE_AXIS_BOUNDS = (0.4, 0.6)
    
    link_name = relevant_joint.body1.split("/")[-1]
    link = target_obj.links[link_name]

    # Get the bounding box of the child link.
    (
        bbox_center_in_world,
        bbox_quat_in_world,
        bbox_extent_in_link_frame,
        bbox_center_in_obj_frame
    ) = target_obj.get_base_aligned_bbox(link_name=link_name, visual=False)

    # Get the part of the object away from the joint position/axis.
    # The link origin is where the joint is. Let's get the position of the origin w.r.t the CoM.
    # from IPython import embed; embed()
    # [target_bid] = target_obj.get_body_ids()
    # dynamics_info = p.getDynamicsInfo(target_bid, link_id)
    # com_wrt_origin = (dynamics_info[3], dynamics_info[4])

    # bbox_wrt_origin = T.pose_transform(link.get_position(), [0, 0, 0, 1], bbox_center_in_link_frame, [0, 0, 0, 1])
    bbox_center_in_world_frame = T.pose_transform(*target_obj.get_position_orientation(), bbox_center_in_obj_frame, [0, 0, 0, 1])[0]
    bbox_wrt_origin = T.relative_pose_transform(bbox_center_in_world_frame, bbox_quat_in_world, *link.get_position_orientation())
    origin_wrt_bbox = T.invert_pose_transform(*bbox_wrt_origin)
    # from IPython import embed; embed()

    joint_orientation = gf_quat_to_np_array(relevant_joint.get_attribute("physics:localRot0"))[[1, 2, 3, 0]]
    joint_axis = R.from_quat(joint_orientation).apply([1, 0, 0])
    joint_axis /= np.linalg.norm(joint_axis)
    origin_towards_bbox = np.array(bbox_wrt_origin[0])
    open_direction = np.cross(joint_axis, origin_towards_bbox)
    open_direction /= np.linalg.norm(open_direction)
    lateral_axis = np.cross(open_direction, joint_axis)

    # Match the axes to the canonical axes of the link bb.
    lateral_axis_idx = np.argmax(np.abs(lateral_axis))
    open_axis_idx = np.argmax(np.abs(open_direction))
    joint_axis_idx = np.argmax(np.abs(joint_axis))
    assert lateral_axis_idx != open_axis_idx
    assert lateral_axis_idx != joint_axis_idx
    assert open_axis_idx != joint_axis_idx

    # Find the correct side of the push/pull axis to grasp from. To do this, imagine the closed position of the object.
    # In that position, which side is the robot on?
    canonical_open_direction = np.eye(3)[open_axis_idx]
    points_along_open_axis = (
        np.array([canonical_open_direction, -canonical_open_direction]) * bbox_extent_in_link_frame[open_axis_idx] / 2
    )
    current_yaw = relevant_joint.get_state()[0][0]
    closed_yaw = relevant_joint.lower_limit
    points_along_open_axis_after_rotation = [
        rotate_point_around_axis((point, [0, 0, 0, 1]), bbox_wrt_origin, joint_axis, closed_yaw - current_yaw)[0]
        for point in points_along_open_axis
    ]
    open_axis_closer_side_idx, _, _ = get_closest_point_to_point_in_world_frame(
        points_along_open_axis_after_rotation, (bbox_center_in_world, bbox_quat_in_world), robot.get_position()
    )
    open_axis_closer_side_sign = 1 if open_axis_closer_side_idx == 0 else -1
    center_of_selected_surface_along_push_axis = points_along_open_axis[open_axis_closer_side_idx]

    # Find the correct side of the lateral axis & go some distance along that direction.
    canonical_joint_axis = np.eye(3)[joint_axis_idx]
    lateral_away_from_origin = np.eye(3)[lateral_axis_idx] * np.sign(origin_towards_bbox[lateral_axis_idx])
    min_lateral_pos_wrt_surface_center = (
        lateral_away_from_origin * -np.array(origin_wrt_bbox[0])
        - canonical_joint_axis * bbox_extent_in_link_frame[lateral_axis_idx] / 2
    )
    max_lateral_pos_wrt_surface_center = (
        lateral_away_from_origin * bbox_extent_in_link_frame[lateral_axis_idx] / 2
        + canonical_joint_axis * bbox_extent_in_link_frame[lateral_axis_idx] / 2
    )
    diff_lateral_pos_wrt_surface_center = max_lateral_pos_wrt_surface_center - min_lateral_pos_wrt_surface_center
    sampled_lateral_pos_wrt_min = np.random.uniform(
        REVOLUTE_JOINT_FRACTION_ACROSS_SURFACE_AXIS_BOUNDS[0] * diff_lateral_pos_wrt_surface_center,
        REVOLUTE_JOINT_FRACTION_ACROSS_SURFACE_AXIS_BOUNDS[1] * diff_lateral_pos_wrt_surface_center,
    )
    lateral_pos_wrt_surface_center = min_lateral_pos_wrt_surface_center + sampled_lateral_pos_wrt_min
    grasp_position = center_of_selected_surface_along_push_axis + lateral_pos_wrt_surface_center
    # Get the appropriate rotation

    grasp_quat_in_bbox_frame = get_quaternion_between_vectors([1, 0, 0], canonical_open_direction * open_axis_closer_side_sign * -1)

    # grasp_quat_in_bbox_frame = get_orientation_facing_vector_with_random_yaw(canonical_open_direction * open_axis_closer_side_sign * -1)
    # Now apply the grasp offset.
    offset_in_bbox_frame = canonical_open_direction * open_axis_closer_side_sign * 0.2
    offset_grasp_pose_in_bbox_frame = (grasp_position + offset_in_bbox_frame, grasp_quat_in_bbox_frame)
    offset_grasp_pose_in_world_frame = T.pose_transform(
        bbox_center_in_world, bbox_quat_in_world, *offset_grasp_pose_in_bbox_frame
    )
    grasp_pose_in_world_frame = T.pose_transform(bbox_center_in_world, bbox_quat_in_world, grasp_position, grasp_quat_in_bbox_frame)

    # To compute the rotation position, we want to decide how far along the rotation axis we'll go.
    desired_yaw = relevant_joint.upper_limit if should_open else relevant_joint.lower_limit
    required_yaw_change = desired_yaw - current_yaw

    # Now we'll rotate the grasp position around the origin by the desired rotation.
    # Note that we use the non-offset position here since the joint can't be pulled all the way to the offset.
    grasp_pose_in_bbox_frame = grasp_position, grasp_quat_in_bbox_frame
    grasp_pose_in_origin_frame = T.pose_transform(*bbox_wrt_origin, *grasp_pose_in_bbox_frame)

    # Get the arc length and divide it up to 10cm segments
    arc_length = abs(required_yaw_change) * np.linalg.norm(grasp_pose_in_origin_frame[0])
    turn_steps = int(ceil(arc_length / ROTATION_ARC_SEGMENT_LENGTHS))
    targets = []
    for i in range(turn_steps):
        partial_yaw_change = (i + 1) / turn_steps * required_yaw_change
        rotated_grasp_pose_in_bbox_frame = rotate_point_around_axis(
            grasp_pose_in_bbox_frame, bbox_wrt_origin, joint_axis, partial_yaw_change
        )
        rotated_grasp_pose_in_world_frame = T.pose_transform(
            bbox_center_in_world, bbox_quat_in_world, *rotated_grasp_pose_in_bbox_frame
        )
        targets.append(rotated_grasp_pose_in_world_frame)

    # Compute the approach direction.
    approach_direction_in_world_frame = Rotation.from_quat(bbox_quat_in_world).apply(canonical_open_direction * -open_axis_closer_side_sign)

    # Decide whether a grasp is required. If approach direction and displacement are similar, no need to grasp.
    movement_in_world_frame = np.array(targets[-1][0]) - np.array(offset_grasp_pose_in_world_frame[0])
    grasp_required = np.dot(movement_in_world_frame, approach_direction_in_world_frame) < 0

    from IPython import embed; embed()
if __name__ == "__main__":
    main()



