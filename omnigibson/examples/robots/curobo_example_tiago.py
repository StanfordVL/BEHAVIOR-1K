import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.curobo import CuroboEmbodimentSelection, CuRoboMotionGenerator
from omnigibson.macros import gm, macros
from omnigibson.object_states import Touching
from omnigibson.robots.holonomic_base_robot import HolonomicBaseRobot

def correct_gripper_friction(robot):
    state = og.sim.dump_state()
    og.sim.stop()
    # Set friction
    from omni.isaac.core.materials import PhysicsMaterial
    gripper_mat = PhysicsMaterial(
        prim_path=f"{robot.prim_path}/gripper_mat",
        name="gripper_material",
        static_friction=50.0,
        dynamic_friction=50.0,
        restitution=None,
    )
    for arm, links in robot.finger_links.items():
        for link in links:
            for msh in link.collision_meshes.values():
                msh.apply_physics_material(gripper_mat)

    og.sim.play()
    og.sim.load_state(state)

def plan_trajectory(cmg, target_pos, target_quat, emb_sel=CuroboEmbodimentSelection.DEFAULT, attached_obj=None):
    # Generate collision-free trajectories to the sampled eef poses (including self-collisions)
    successes, traj_paths = cmg.compute_trajectories(
        target_pos=target_pos,
        target_quat=target_quat,
        is_local=False,
        max_attempts=50,
        timeout=60.0,
        ik_fail_return=5,
        enable_finetune_trajopt=True,
        finetune_attempts=1,
        return_full_result=False,
        success_ratio=1.0,
        attached_obj=attached_obj,
        emb_sel=emb_sel,
    )
    return successes, traj_paths


def execute_trajectory(q_traj, env, robot, attached_obj):
    for i, q in enumerate(q_traj):
        q = q.cpu()
        q = set_gripper_joint_positions(robot, q, attached_obj)
        command = q_to_command(q, robot)

        num_repeat = 5
        print(f"Executing waypoint {i}/{len(q_traj)}")
        for _ in range(num_repeat):
            env.step(command)


def plan_and_execute_trajectory(
    cmg, target_pos, target_quat, emb_sel, attached_obj, eef_markers, env, robot, dry_run=False
):
    successes, traj_paths = plan_trajectory(cmg, target_pos, target_quat, emb_sel, attached_obj)
    success, traj_path = successes[0], traj_paths[0]
    # Move the markers to the desired eef positions
    for marker in eef_markers:
        marker.set_position_orientation(position=th.tensor([100, 100, 100]))
    for target_link, marker in zip(target_pos.keys(), eef_markers):
        marker.set_position_orientation(position=target_pos[target_link])
    if success:
        print("Successfully planned trajectory")
        if not dry_run:
            q_traj = cmg.path_to_joint_trajectory(traj_path, emb_sel)
            execute_trajectory(q_traj, env, robot, attached_obj)
    else:
        print("Failed to plan trajectory")


def set_gripper_joint_positions(robot, q, attached_obj):
    # Overwrite the gripper joint positions based on attached_obj
    joint_names = list(robot.joints.keys())
    # print("attached_obj: ", attached_obj)
    for arm, finger_joints in robot.finger_joints.items():
        close_gripper = attached_obj is not None and robot.eef_link_names[arm] in attached_obj
        for finger_joint in finger_joints:
            idx = joint_names.index(finger_joint.joint_name)
            q[idx] = finger_joint.lower_limit if close_gripper else finger_joint.upper_limit
            # print(f"finger_joint: {finger_joint.joint_name}, close_gripper: {close_gripper}, q[idx]: {q[idx]}")
    return q


def control_gripper(env, robot, attached_obj):
    # Control the gripper to open or close, while keeping the rest of the robot still
    q = robot.get_joint_positions()
    q = set_gripper_joint_positions(robot, q, attached_obj)
    command = q_to_command(q, robot)
    num_repeat = 100
    print(f"Gripper (attached_obj={attached_obj})")
    for _ in range(num_repeat):
        env.step(command)


def q_to_command(q, robot):
    # Convert target joint positions to command
    command = []
    for controller in robot.controllers.values():
        command.append(q[controller.dof_idx])
    command = th.cat(command, dim=0)
    assert command.shape[0] == robot.action_dim
    return command


def set_all_seeds(seed):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True

def test_curobo():
    set_all_seeds(seed=1)
    ROBOT_TYPE = "Tiago"
    robot_cfg = {
        "Tiago": {
            "type": "Tiago",
            "obs_modalities": "rgb",
            "position": [0, 0, 0],
            "orientation": [0, 0, 0, 1],
            "self_collisions": True,
            "action_normalize": False,
            "grasping_mode": "assisted",
            "rigid_trunk": False,
            "controller_config": {
                "base": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": None,
                    "use_delta_commands": False,
                    "use_impedances": True,
                },
                "camera": {
                    "name": "JointController",
                },
                "arm_left": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": None,
                    "use_delta_commands": False,
                    "use_impedances": True,
                    "pos_kp": 200.0,
                },
                "arm_right": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": None,
                    "use_delta_commands": False,
                    "use_impedances": True,
                    "pos_kp": 200.0,
                },
                "gripper_left": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": [-1, 1],
                    "use_delta_commands": False,
                    "use_impedances": True,
                    "pos_kp": 1500.0,
                },
                "gripper_right": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": [-1, 1],
                    "use_delta_commands": False,
                    "use_impedances": True,
                    "pos_kp": 1500.0,
                },
            },
        },
    }
    robots = []
    robots.append(robot_cfg[ROBOT_TYPE])

    # Create env
    cfg = {
        "env": {
            "action_frequency": 30,
            "physics_frequency": 300,
        },
        "scene": {
            "type": "Scene",
        },
        "objects": [
            {
                "type": "PrimitiveObject",
                "name": "eef_marker_0",
                "primitive_type": "Sphere",
                "radius": 0.05,
                "visual_only": True,
                "position": [0, 0, 0],
                "orientation": [0, 0, 0, 1],
                "rgba": [1, 0, 0, 1],
            },
            {
                "type": "PrimitiveObject",
                "name": "eef_marker_1",
                "primitive_type": "Sphere",
                "radius": 0.05,
                "visual_only": True,
                "position": [0, 0, 0],
                "orientation": [0, 0, 0, 1],
                "rgba": [0, 1, 0, 1],
            },
            {
                "type": "DatasetObject",
                "name": "table",
                "category": "breakfast_table",
                "model": "rjgmmy",
                "position": [2, 0, 0.41],
                "orientation": [0, 0, 0, 1],
            },
            {
                "type": "DatasetObject",
                "name": "cologne",
                "category": "bottle_of_cologne",
                "model": "lyipur",
                "position": [1.6, 0.15, 0.65],
                "orientation": [0, 0, 0, 1],
            },
            {
                "type": "DatasetObject",
                "name": "fridge",
                "category": "fridge",
                "model": "dszchb",
                "position": [2, 1, 0.86],
                "orientation": T.euler2quat(th.tensor([0, 0, -math.pi / 2])),
            },
        ],
        "robots": robots,
    }

    env = og.Environment(configs=cfg)
    robot = env.robots[0]
    eef_markers = [env.scene.object_registry("name", f"eef_marker_{i}") for i in range(2)]

    correct_gripper_friction(robot)

    og.sim.viewer_camera.set_position_orientation(position=[0.59, 2.31, 2.07], orientation=[-0.086, 0.434, 0.879, -0.175])

    ee_link_left = "gripper_left_grasping_frame"
    ee_link_right = "gripper_right_grasping_frame"

    # Stablize the robot and update the initial state
    robot.reset()

    # Open the gripper(s) to match cuRobo's default state
    for arm_name in robot.gripper_control_idx.keys():
        grpiper_control_idx = robot.gripper_control_idx[arm_name]
        robot.set_joint_positions(th.ones_like(grpiper_control_idx), indices=grpiper_control_idx, normalized=True)
    robot.keep_still()

    for _ in range(5):
        og.sim.step()

    env.scene.update_initial_state()
    env.scene.reset()

    # Create CuRobo instance
    cmg = CuRoboMotionGenerator(
        robot=robot,
        batch_size=1,
        use_cuda_graph=True,
    )

    table = env.scene.object_registry("name", "table")
    cologne = env.scene.object_registry("name", "cologne")
    fridge = env.scene.object_registry("name", "fridge")

    table_local_pose = (th.tensor([-1.1, 0.0, -0.402]), th.tensor([0.0, 0.0, 0.0, 1.0]))

    cologne_local_pose = (th.tensor([-0.03, 0.0, 0.102]), T.euler2quat(th.tensor([math.pi, math.pi / 2.0, 0.0])))

    fridge_local_pose = (th.tensor([-0.55, -1.25, -0.8576]), T.euler2quat(th.tensor([0.0, 0.0, math.pi / 2.0])))

    fridge_door_local_pose = (
        th.tensor([-0.28, -0.42, 0.15]),
        T.euler2quat(th.tensor([math.pi, 0.0, math.pi / 2.0])),
    )

    fridge_door_open_local_pose = (th.tensor([0.35, -0.97, 0.15]), T.euler2quat(th.tensor([math.pi, 0.0, math.pi])))
    
    fridge_place_local_pose = (
        th.tensor([-0.10, -0.15, 0.5]),
        T.euler2quat(th.tensor([math.pi, 0.0, math.pi / 2.0])),
    )

    print("start")
    breakpoint()

    # Navigate to table (base)
    table_nav_pos, table_nav_quat = T.pose_transform(*table.get_position_orientation(), *table_local_pose)
    target_pos = {robot.base_footprint_link_name: table_nav_pos}
    target_quat = {robot.base_footprint_link_name: table_nav_quat}
    emb_sel = CuroboEmbodimentSelection.BASE
    attached_obj = None
    plan_and_execute_trajectory(cmg, target_pos, target_quat, emb_sel, attached_obj, eef_markers, env, robot)

    # Record reset pose
    left_hand_reset_pos, left_hand_reset_quat = robot.get_eef_pose(arm="left")
    right_hand_reset_pos, right_hand_reset_quat = robot.get_eef_pose(arm="right")

    # Grasp cologne (left hand)
    left_hand_pos, left_hand_quat = T.pose_transform(*cologne.get_position_orientation(), *cologne_local_pose)
    right_hand_pos, right_hand_quat = robot.get_eef_pose(arm="right")
    # target_pos = {robot.eef_link_names["left"]: left_hand_pos, robot.eef_link_names["right"]: right_hand_pos}
    # target_quat = {robot.eef_link_names["left"]: left_hand_quat, robot.eef_link_names["right"]: right_hand_quat}
    target_pos = {robot.eef_link_names["left"]: left_hand_pos}
    target_quat = {robot.eef_link_names["left"]: left_hand_quat}
    emb_sel = CuroboEmbodimentSelection.ARM
    attached_obj = None
    plan_and_execute_trajectory(cmg, target_pos, target_quat, emb_sel, attached_obj, eef_markers, env, robot)
    attached_obj = {ee_link_left: cologne.root_link}
    control_gripper(env, robot, attached_obj)
    # assert robot._ag_obj_in_hand["left"] == cologne and robot._ag_obj_in_hand["right"] == None

    # Move left hand up
    target_pos = {robot.eef_link_names["left"]: left_hand_pos + th.tensor([0, 0, 0.1])}
    target_quat = {robot.eef_link_names["left"]: left_hand_quat}
    emb_sel = CuroboEmbodimentSelection.ARM
    attached_obj = {ee_link_left: cologne.root_link}
    # breakpoint()
    plan_and_execute_trajectory(cmg, target_pos, target_quat, emb_sel, attached_obj, eef_markers, env, robot)

    # Reset to reset pose (both hands)
    target_pos = {
        robot.eef_link_names["left"]: left_hand_reset_pos,
        robot.eef_link_names["right"]: right_hand_reset_pos,
    }
    target_quat = {
        robot.eef_link_names["left"]: left_hand_reset_quat,
        robot.eef_link_names["right"]: right_hand_reset_quat,
    }
    emb_sel = CuroboEmbodimentSelection.ARM
    attached_obj = {ee_link_left: cologne.root_link}
    plan_and_execute_trajectory(cmg, target_pos, target_quat, emb_sel, attached_obj, eef_markers, env, robot)

    # Navigate to fridge (base)
    print("Navigate to fridge (base)")
    fridge_nav_pos, fridge_nav_quat = T.pose_transform(*fridge.get_position_orientation(), *fridge_local_pose)
    target_pos = {robot.base_footprint_link_name: fridge_nav_pos}
    target_quat = {robot.base_footprint_link_name: fridge_nav_quat}
    emb_sel = CuroboEmbodimentSelection.BASE
    attached_obj = {ee_link_left: cologne.root_link}
    plan_and_execute_trajectory(cmg, target_pos, target_quat, emb_sel, attached_obj, eef_markers, env, robot)

    # Grasp fridge door (right hand)
    print("Grasp fridge door (right hand)")
    # left_hand_pos, left_hand_quat = robot.get_eef_pose(arm="left")
    right_hand_pos, right_hand_quat = T.pose_transform(*fridge.get_position_orientation(), *fridge_door_local_pose)
    # target_pos = {robot.eef_link_names["left"]: left_hand_pos, robot.eef_link_names["right"]: right_hand_pos}
    # target_quat = {robot.eef_link_names["left"]: left_hand_quat, robot.eef_link_names["right"]: right_hand_quat}
    target_pos = {robot.eef_link_names["right"]: right_hand_pos}
    target_quat = {robot.eef_link_names["right"]: right_hand_quat}
    emb_sel = CuroboEmbodimentSelection.ARM
    attached_obj = {ee_link_left: cologne.root_link}
    plan_and_execute_trajectory(cmg, target_pos, target_quat, emb_sel, attached_obj, eef_markers, env, robot)
    attached_obj = {ee_link_left: cologne.root_link, ee_link_right: fridge.links["link_0"]}
    control_gripper(env, robot, attached_obj)
    # assert robot._ag_obj_in_hand["left"] == cologne and robot._ag_obj_in_hand["right"] == fridge

    # Pull fridge door (right hand)
    print("Pull fridge door (right hand)")
    right_hand_pos, right_hand_quat = T.pose_transform(*fridge.get_position_orientation(), *fridge_door_open_local_pose)
    # rel_pos, rel_quat = T.relative_pose_transform(
    #     left_hand_reset_pos, left_hand_reset_quat, right_hand_reset_pos, right_hand_reset_quat
    # )
    # left_hand_pos, left_hand_quat = T.pose_transform(right_hand_pos, right_hand_quat, rel_pos, rel_quat)
    # target_pos = {robot.eef_link_names["left"]: left_hand_pos, robot.eef_link_names["right"]: right_hand_pos}
    # target_quat = {robot.eef_link_names["left"]: left_hand_quat, robot.eef_link_names["right"]: right_hand_quat}
    target_pos = {robot.eef_link_names["right"]: right_hand_pos}
    target_quat = {robot.eef_link_names["right"]: right_hand_quat}
    emb_sel = CuroboEmbodimentSelection.DEFAULT
    attached_obj = {ee_link_left: cologne.root_link, ee_link_right: fridge.links["link_0"]}
    plan_and_execute_trajectory(cmg, target_pos, target_quat, emb_sel, attached_obj, eef_markers, env, robot)
    attached_obj = {ee_link_left: cologne.root_link}
    control_gripper(env, robot, attached_obj)
    # assert robot._ag_obj_in_hand["left"] == cologne and robot._ag_obj_in_hand["right"] == None

    # Place the cologne (left hand)
    left_hand_pos, left_hand_quat = T.pose_transform(*fridge.get_position_orientation(), *fridge_place_local_pose)
    # Unclear how to find this pose directly, I just run the two steps below and record the resulting robot.get_eef_pose("right")
    # right_hand_pos, right_hand_quat = th.tensor([0.7825, 1.3466, 0.9568]), th.tensor(
    #     [-0.7083, -0.0102, -0.7058, 0.0070]
    # )
    # target_pos = {robot.eef_link_names["left"]: left_hand_pos, robot.eef_link_names["right"]: right_hand_pos}
    # target_quat = {robot.eef_link_names["left"]: left_hand_quat, robot.eef_link_names["right"]: right_hand_quat}
    target_pos = {robot.eef_link_names["left"]: left_hand_pos}
    target_quat = {robot.eef_link_names["left"]: left_hand_quat}
    emb_sel = CuroboEmbodimentSelection.DEFAULT
    attached_obj = {ee_link_left: cologne.root_link}
    plan_and_execute_trajectory(cmg, target_pos, target_quat, emb_sel, attached_obj, eef_markers, env, robot)
    attached_obj = None
    control_gripper(env, robot, attached_obj)
    # assert robot._ag_obj_in_hand["left"] == None and robot._ag_obj_in_hand["right"] == None

    # # Navigate to fridge (step 1)
    # left_hand_pos, left_hand_quat = T.pose_transform(*fridge.get_position_orientation(), *fridge_place_local_pose_prepare)
    # rel_pos, rel_quat = th.tensor([0.0, 0.8, 0.0]), T.euler2quat(th.tensor([0.0, 0.0, 0.0]))
    # right_hand_pos, right_hand_quat = T.pose_transform(left_hand_pos, left_hand_quat, rel_pos, rel_quat)
    # target_pos = {robot.eef_link_names["left"]: left_hand_pos, robot.eef_link_names["right"]: right_hand_pos}
    # target_quat = {robot.eef_link_names["left"]: left_hand_quat, robot.eef_link_names["right"]: right_hand_quat}
    # emb_sel = CuroboEmbodimentSelection.DEFAULT
    # attached_obj = {"left_hand": cologne.root_link}
    # plan_and_execute_trajectory(cmg, target_pos, target_quat, emb_sel, attached_obj, eef_markers, env, robot)

    # # Place the cologne (step 2)
    # left_hand_pos, left_hand_quat = T.pose_transform(*fridge.get_position_orientation(), *fridge_place_local_pose)
    # target_pos = {robot.eef_link_names["left"]: left_hand_pos}
    # target_quat = {robot.eef_link_names["left"]: left_hand_quat}
    # emb_sel = CuroboEmbodimentSelection.DEFAULT
    # attached_obj = {"left_hand": cologne.root_link}
    # plan_and_execute_trajectory(cmg, target_pos, target_quat, emb_sel, attached_obj, eef_markers, env, robot)
    # attached_obj = None
    # control_gripper(env, robot, attached_obj)

    print("done")
    breakpoint()

    og.shutdown()


if __name__ == "__main__":
    test_curobo()
