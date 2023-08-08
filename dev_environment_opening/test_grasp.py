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

    config["scene"]["load_object_categories"] = ["floors", "bottom_cabinet"]

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    controller = StarterSemanticActionPrimitives(None, scene, robot)

    for o in scene.objects:
        if o.prim_path == "/World/bottom_cabinet_bamfsz_0":
            cabinet = o

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
        get_grasp_position_for_open(robot, cabinet, True)
        # execute_controller(controller._open_or_close(cabinet), env)

    def test_open():
        set_start_pose()
        pose_2d = [-0.762831, -0.377231, 2.72892]
        pose = controller._get_robot_pose_from_2d_pose(pose_2d)
        robot.set_position_orientation(*pose)
        og.sim.step()

        # joint_pos = [0.0133727 ,0.216775 ,0.683931 ,2.04371 ,1.88204 ,0.720747 ,1.23276 ,1.72251]
        # control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx["left"]])
        # robot.set_joint_positions(joint_pos, control_idx)
        # og.sim.step()
        # pause(100)
        execute_controller(controller._open_or_close(cabinet, True), env)
        # replay_controller(env, "./replays/test_tiago_open.yaml")
    
    test_open()

    marker = None
    def set_marker(position):
        marker = PrimitiveObject(
            prim_path=f"/World/marker",
            name="marker",
            primitive_type="Cube",
            size=0.07,
            visual_only=True,
            rgba=[1.0, 0, 0, 1.0],
        )
        og.sim.import_object(marker)
        marker.set_position(position)
        og.sim.step()

    def remove_marker():
        marker.remove()
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

    ########################################################

    # from omnigibson.objects.primitive_object import PrimitiveObject
    # import numpy as np
    # import random
    # from scipy.spatial.transform import Rotation

    # import omnigibson.utils.transform_utils as T
    # from omnigibson.object_states.open import _get_relevant_joints
    # from omnigibson.utils.constants import JointType, JointAxis
    # from omni.isaac.core.utils.rotations import gf_quat_to_np_array
    # from scipy.spatial.transform import Rotation as R

    # p = None
    # grasp_position_for_open_on_revolute_joint = None
    # should_open = True
    # PRISMATIC_JOINT_FRACTION_ACROSS_SURFACE_AXIS_BOUNDS = (0.2, 0.8)
    # GRASP_OFFSET = np.array([0, 0.05, -0.08])
    # OPEN_GRASP_OFFSET = np.array([0, 0.05, -0.12])  # 5cm back and 12cm up.

    # # Pick a moving link of the object.
    # relevant_joints_full = _get_relevant_joints(cabinet)
    # relevant_joints = relevant_joints_full[1]

    # # If a particular link ID was specified, filter our candidates down to that one.
    # # if link_id is not None:
    # #     relevant_joints = [ji for ji in relevant_joints if ji.jointIndex == link_id]

    # if len(relevant_joints) == 0:
    #     raise ValueError("Cannot open/close object without relevant joints.")

    # # Make sure what we got is an appropriately open/close joint.
    # random.shuffle(relevant_joints)
    # selected_joint = None
    # for joint in relevant_joints:
    #     current_position = joint.get_state()[0][0]
    #     joint_range = joint.upper_limit - joint.lower_limit
    #     openness_fraction = (current_position - joint.lower_limit) / joint_range
    #     if (should_open and openness_fraction < 0.8) or (not should_open and openness_fraction > 0.05):
    #         selected_joint = joint


    # target_obj = cabinet
    # relevant_joint = relevant_joints_full[1][2]

    # link_name = relevant_joint.body1.split("/")[-1]
    
    # # Get the bounding box of the child link.
    # (
    #     bbox_center_in_world,
    #     bbox_quat_in_world,
    #     bbox_extent_in_link_frame,
    #     _,
    # ) = target_obj.get_base_aligned_bbox(link_name=link_name, visual=False)
    # from IPython import embed; embed()
    # # Match the push axis to one of the bb axes.
    # push_axis_idx = JointAxis.index(relevant_joint.axis)
    # canonical_push_axis = np.eye(3)[push_axis_idx]
    # joint_orientation = gf_quat_to_np_array(relevant_joint.get_attribute("physics:localRot0"))[[1, 2, 3, 0]]
    # push_axis = R.from_quat(joint_orientation).apply([1, 0, 0])
    # assert np.isclose(np.max(np.abs(push_axis)), 1.0)  # Make sure we're aligned with a bb axis.
    # push_axis_idx = np.argmax(np.abs(push_axis))
    # canonical_push_axis = np.eye(3)[push_axis_idx]


    # # TODO: Need to figure out how to get the correct push direction.
    # canonical_push_direction = canonical_push_axis * np.sign(push_axis[push_axis_idx])

    # # Pick the closer of the two faces along the push axis as our favorite.
    # points_along_push_axis = (
    #     np.array([canonical_push_axis, -canonical_push_axis]) * bbox_extent_in_link_frame[push_axis_idx] / 2
    # )
    # (
    #     push_axis_closer_side_idx,
    #     center_of_selected_surface_along_push_axis,
    #     _,
    # ) = get_closest_point_to_point_in_world_frame(
    #     points_along_push_axis, (bbox_center_in_world, bbox_quat_in_world), robot.get_position()
    # )
    # push_axis_closer_side_sign = 1 if push_axis_closer_side_idx == 0 else -1

    # # Pick the other axes.
    # all_axes = list(set(range(3)) - {push_axis_idx})
    # x_axis_idx, y_axis_idx = tuple(sorted(all_axes))
    # canonical_x_axis = np.eye(3)[x_axis_idx]
    # canonical_y_axis = np.eye(3)[y_axis_idx]

    # # Find the correct side of the lateral axis & go some distance along that direction.
    # min_lateral_pos_wrt_surface_center = (canonical_x_axis + canonical_y_axis) * -bbox_extent_in_link_frame / 2
    # max_lateral_pos_wrt_surface_center = (canonical_x_axis + canonical_y_axis) * bbox_extent_in_link_frame / 2
    # diff_lateral_pos_wrt_surface_center = max_lateral_pos_wrt_surface_center - min_lateral_pos_wrt_surface_center
    # sampled_lateral_pos_wrt_min = np.random.uniform(
    #     PRISMATIC_JOINT_FRACTION_ACROSS_SURFACE_AXIS_BOUNDS[0] * diff_lateral_pos_wrt_surface_center,
    #     PRISMATIC_JOINT_FRACTION_ACROSS_SURFACE_AXIS_BOUNDS[1] * diff_lateral_pos_wrt_surface_center,
    # )
    # lateral_pos_wrt_surface_center = min_lateral_pos_wrt_surface_center + sampled_lateral_pos_wrt_min
    # grasp_position_in_bbox_frame = center_of_selected_surface_along_push_axis + lateral_pos_wrt_surface_center
    # grasp_quat_in_bbox_frame = T.quat_inverse(joint_orientation)

    # # Now apply the grasp offset.
    # offset_grasp_pose_in_bbox_frame = (grasp_position_in_bbox_frame, grasp_quat_in_bbox_frame)
    # offset_grasp_pose_in_world_frame = T.pose_transform(
    #     bbox_center_in_world, bbox_quat_in_world, *offset_grasp_pose_in_bbox_frame
    # )

    # # To compute the rotation position, we want to decide how far along the rotation axis we'll go.
    # target_joint_pos = relevant_joint.upper_limit if should_open else relevant_joint.lower_limit
    # current_joint_pos = relevant_joint.get_state()[0][0]
    
    # required_pos_change = target_joint_pos - current_joint_pos
    # push_vector_in_bbox_frame = canonical_push_direction * required_pos_change
    # target_hand_pos_in_bbox_frame = grasp_position_in_bbox_frame + push_vector_in_bbox_frame
    # target_hand_pos_in_world_frame = T.pose_transform(
    #     bbox_center_in_world, bbox_quat_in_world, target_hand_pos_in_bbox_frame, grasp_quat_in_bbox_frame
    # )

    # # Compute the approach direction.
    # approach_direction_in_world_frame = R.from_quat(bbox_quat_in_world).apply(canonical_push_axis * -push_axis_closer_side_sign)

    # # Decide whether a grasp is required. If approach direction and displacement are similar, no need to grasp.
    # grasp_required = np.dot(push_vector_in_bbox_frame, canonical_push_axis * -push_axis_closer_side_sign) < 0

    # # return (
    # #     offset_grasp_pose_in_world_frame,
    # #     [target_hand_pos_in_world_frame],
    # #     approach_direction_in_world_frame,
    # #     relevant_joint,
    # #     grasp_required,
    # # )
    # from IPython import embed; embed()


if __name__ == "__main__":
    main()



