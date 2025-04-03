import math

import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection, CuRoboMotionGenerator
from omnigibson.utils.ui_utils import choose_from_options


def plan_trajectory(
    cmg, target_pos, target_quat, emb_sel=CuRoboEmbodimentSelection.DEFAULT, attached_obj=None, attached_obj_scale=None
):
    # Generate collision-free trajectories to the sampled eef poses (including self-collisions)
    successes, traj_paths = cmg.compute_trajectories(
        target_pos=target_pos,
        target_quat=target_quat,
        is_local=False,
        max_attempts=100,
        timeout=60.0,
        ik_fail_return=50,
        enable_finetune_trajopt=True,
        finetune_attempts=1,
        return_full_result=False,
        success_ratio=1.0,
        attached_obj=attached_obj,
        attached_obj_scale=attached_obj_scale,
        emb_sel=emb_sel,
    )
    return successes, traj_paths


def execute_trajectory(q_traj, env, robot, attached_obj):
    for i, q in enumerate(q_traj):
        q = set_gripper_joint_positions(robot, q, attached_obj)
        action = robot.q_to_action(q)
        print(f"Executing waypoint {i}/{len(q_traj)}")
        env.step(action)


def plan_and_execute_trajectory(
    cmg, target_pos, target_quat, emb_sel, attached_obj, attached_obj_scale, eef_markers, env, robot, dry_run=False
):
    successes, traj_paths = plan_trajectory(cmg, target_pos, target_quat, emb_sel, attached_obj, attached_obj_scale)
    success, traj_path = successes[0], traj_paths[0]
    # Move the markers to the desired eef positions
    for marker in eef_markers:
        marker.set_position_orientation(position=th.tensor([100, 100, 100]))
    for target_link, marker in zip(target_pos.keys(), eef_markers):
        marker.set_position_orientation(position=target_pos[target_link])
    if success:
        print("Successfully planned trajectory")
        if not dry_run:
            q_traj = cmg.path_to_joint_trajectory(traj_path, get_full_js=True, emb_sel=emb_sel).cpu().float()
            q_traj = cmg.add_linearly_interpolated_waypoints(traj=q_traj, max_inter_dist=0.01)
            execute_trajectory(q_traj, env, robot, attached_obj)
    else:
        print("Failed to plan trajectory")


def set_gripper_joint_positions(robot, q, attached_obj):
    # Overwrite the gripper joint positions based on attached_obj
    joint_names = list(robot.joints.keys())
    for arm, finger_joints in robot.finger_joints.items():
        close_gripper = attached_obj is not None and robot.eef_link_names[arm] in attached_obj
        for finger_joint in finger_joints:
            idx = joint_names.index(finger_joint.joint_name)
            q[idx] = finger_joint.lower_limit if close_gripper else finger_joint.upper_limit
    return q


def control_gripper(env, robot, attached_obj):
    # Control the gripper to open or close, while keeping the rest of the robot still
    q = robot.get_joint_positions()
    q = set_gripper_joint_positions(robot, q, attached_obj)
    action = robot.q_to_action(q)
    num_repeat = 30
    print(f"Gripper (attached_obj={attached_obj})")
    for _ in range(num_repeat):
        env.step(action)


def test_curobo():
    # Ask the user whether they want online object sampling or not
    robot_options = ["R1", "Tiago"]
    robot_type = choose_from_options(options=robot_options, name="robot options", random_selection=False)

    robot_cfg = {
        "type": robot_type,
        "obs_modalities": "rgb",
        "position": [0, 0, 0],
        "orientation": [0, 0, 0, 1],
        "self_collisions": True,
        "action_normalize": False,
        "grasping_mode": "sticky",
        "controller_config": {
            "base": {
                "name": "HolonomicBaseJointController",
                "motor_type": "position",
                "command_input_limits": None,
                "use_impedances": False,
            },
            "trunk": {
                "name": "JointController",
                "motor_type": "position",
                "command_input_limits": None,
                "use_delta_commands": False,
                "use_impedances": False,
            },
            "arm_left": {
                "name": "JointController",
                "motor_type": "position",
                "command_input_limits": None,
                "use_delta_commands": False,
                "use_impedances": False,
            },
            "arm_right": {
                "name": "JointController",
                "motor_type": "position",
                "command_input_limits": None,
                "use_delta_commands": False,
                "use_impedances": False,
            },
            "gripper_left": {
                "name": "JointController",
                "motor_type": "position",
                "command_input_limits": None,
                "use_delta_commands": False,
                "use_impedances": False,
            },
            "gripper_right": {
                "name": "JointController",
                "motor_type": "position",
                "command_input_limits": None,
                "use_delta_commands": False,
                "use_impedances": False,
            },
        },
    }
    if robot_type == "Tiago":
        robot_cfg["controller_config"]["camera"] = {
            "name": "JointController",
            "motor_type": "position",
            "command_input_limits": None,
            "use_delta_commands": False,
            "use_impedances": False,
        }

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
                "orientation": T.euler2quat(th.tensor([0, 0, math.pi / 2])),
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
        "robots": [robot_cfg],
    }

    env = og.Environment(configs=cfg)
    robot = env.robots[0]
    eef_markers = [env.scene.object_registry("name", f"eef_marker_{i}") for i in range(2)]

    # Create CuRobo instance
    cmg = CuRoboMotionGenerator(
        robot=robot,
        batch_size=1,
        use_cuda_graph=True,
    )

    table = env.scene.object_registry("name", "table")
    cologne = env.scene.object_registry("name", "cologne")
    fridge = env.scene.object_registry("name", "fridge")
    cologne_scale = 0.95
    fridge_door_scale = 0.6
    if robot_type == "R1":
        table_local_pose = (th.tensor([-1.1, 0.0, -0.402]), th.tensor([0.0, 0.0, 0.0, 1.0]))
        cologne_local_pose = (th.tensor([0.0, 0.0, 0.04]), th.tensor([0.5, -0.5, 0.5, 0.5]))
        fridge_local_pose = (th.tensor([-0.45, -1.2, -0.8576]), T.euler2quat(th.tensor([0.0, 0.0, math.pi / 2.0])))
        fridge_door_local_pose = (
            th.tensor([-0.28, -0.37, 0.15]),
            T.euler2quat(th.tensor([-math.pi / 2, -math.pi / 2, 0.0])),
        )
        fridge_door_open_local_pose = (
            th.tensor([0.35, -0.90, 0.15]),
            T.euler2quat(th.tensor([0.0, -math.pi / 2, 0.0])),
        )
        fridge_place_local_pose = (
            th.tensor([-0.15, -0.20, 0.5]),
            T.euler2quat(th.tensor([-math.pi / 2, -math.pi / 2, 0.0])),
        )
    elif robot_type == "Tiago":
        table_local_pose = (th.tensor([-1.1, 0.0, -0.402]), th.tensor([0.0, 0.0, 0.0, 1.0]))
        cologne_local_pose = (th.tensor([0.0, 0.0, 0.04]), th.tensor([0.5, -0.5, 0.5, 0.5]))
        fridge_local_pose = (th.tensor([-0.55, -1.25, -0.8576]), T.euler2quat(th.tensor([0.0, 0.0, math.pi / 2.0])))
        fridge_door_local_pose = (
            th.tensor([-0.28, -0.37, 0.15]),
            T.euler2quat(th.tensor([-math.pi / 2, -math.pi / 2, 0.0])),
        )
        fridge_door_open_local_pose = (
            th.tensor([0.35, -0.97, 0.15]),
            T.euler2quat(th.tensor([0.0, -math.pi / 2, 0.0])),
        )
        fridge_place_local_pose = (
            th.tensor([-0.10, -0.15, 0.5]),
            T.euler2quat(th.tensor([-math.pi / 2, -math.pi / 2, 0.0])),
        )
    else:
        raise ValueError(f"Unknown robot type: {robot_type}")

    # Set random seed for reproducibility
    th.manual_seed(1)

    # Navigate to table (base)
    table_nav_pos, table_nav_quat = T.pose_transform(*table.get_position_orientation(), *table_local_pose)
    target_pos = {robot.base_footprint_link_name: table_nav_pos}
    target_quat = {robot.base_footprint_link_name: table_nav_quat}
    emb_sel = CuRoboEmbodimentSelection.BASE
    attached_obj = None
    attached_obj_scale = None
    plan_and_execute_trajectory(
        cmg, target_pos, target_quat, emb_sel, attached_obj, attached_obj_scale, eef_markers, env, robot
    )

    # Record reset pose
    left_hand_reset_pos, left_hand_reset_quat = robot.get_eef_pose(arm="left")
    right_hand_reset_pos, right_hand_reset_quat = robot.get_eef_pose(arm="right")

    # Grasp cologne (left hand)
    left_hand_pos, left_hand_quat = T.pose_transform(*cologne.get_position_orientation(), *cologne_local_pose)
    right_hand_pos, right_hand_quat = robot.get_eef_pose(arm="right")
    target_pos = {robot.eef_link_names["left"]: left_hand_pos}
    target_quat = {robot.eef_link_names["left"]: left_hand_quat}
    emb_sel = CuRoboEmbodimentSelection.ARM
    attached_obj = None
    attached_obj_scale = None
    plan_and_execute_trajectory(
        cmg, target_pos, target_quat, emb_sel, attached_obj, attached_obj_scale, eef_markers, env, robot
    )
    attached_obj = {robot.eef_link_names["left"]: cologne.root_link}
    control_gripper(env, robot, attached_obj)
    assert robot._ag_obj_in_hand["left"] == cologne and robot._ag_obj_in_hand["right"] is None

    # Reset to reset pose (both hands)
    target_pos = {
        robot.eef_link_names["left"]: left_hand_reset_pos,
        robot.eef_link_names["right"]: right_hand_reset_pos,
    }
    target_quat = {
        robot.eef_link_names["left"]: left_hand_reset_quat,
        robot.eef_link_names["right"]: right_hand_reset_quat,
    }
    emb_sel = CuRoboEmbodimentSelection.ARM
    attached_obj = {robot.eef_link_names["left"]: cologne.root_link}
    attached_obj_scale = {robot.eef_link_names["left"]: cologne_scale}
    plan_and_execute_trajectory(
        cmg, target_pos, target_quat, emb_sel, attached_obj, attached_obj_scale, eef_markers, env, robot
    )

    # Navigate to fridge (base)
    fridge_nav_pos, fridge_nav_quat = T.pose_transform(*fridge.get_position_orientation(), *fridge_local_pose)
    target_pos = {robot.base_footprint_link_name: fridge_nav_pos}
    target_quat = {robot.base_footprint_link_name: fridge_nav_quat}
    emb_sel = CuRoboEmbodimentSelection.BASE
    attached_obj = {robot.eef_link_names["left"]: cologne.root_link}
    attached_obj_scale = {robot.eef_link_names["left"]: cologne_scale}
    plan_and_execute_trajectory(
        cmg, target_pos, target_quat, emb_sel, attached_obj, attached_obj_scale, eef_markers, env, robot
    )

    # Grasp fridge door (right hand)
    right_hand_pos, right_hand_quat = T.pose_transform(*fridge.get_position_orientation(), *fridge_door_local_pose)
    target_pos = {robot.eef_link_names["right"]: right_hand_pos}
    target_quat = {robot.eef_link_names["right"]: right_hand_quat}
    emb_sel = CuRoboEmbodimentSelection.ARM
    attached_obj = {robot.eef_link_names["left"]: cologne.root_link}
    attached_obj_scale = {robot.eef_link_names["left"]: cologne_scale}
    plan_and_execute_trajectory(
        cmg, target_pos, target_quat, emb_sel, attached_obj, attached_obj_scale, eef_markers, env, robot
    )
    attached_obj = {
        robot.eef_link_names["left"]: cologne.root_link,
        robot.eef_link_names["right"]: fridge.links["link_0"],
    }
    control_gripper(env, robot, attached_obj)
    assert robot._ag_obj_in_hand["left"] == cologne and robot._ag_obj_in_hand["right"] == fridge

    # Pull fridge door (right hand)
    right_hand_pos, right_hand_quat = T.pose_transform(*fridge.get_position_orientation(), *fridge_door_open_local_pose)
    target_pos = {robot.eef_link_names["right"]: right_hand_pos}
    target_quat = {robot.eef_link_names["right"]: right_hand_quat}
    emb_sel = CuRoboEmbodimentSelection.DEFAULT
    attached_obj = {
        robot.eef_link_names["left"]: cologne.root_link,
        robot.eef_link_names["right"]: fridge.links["link_0"],
    }
    attached_obj_scale = {robot.eef_link_names["left"]: cologne_scale, robot.eef_link_names["right"]: fridge_door_scale}
    plan_and_execute_trajectory(
        cmg, target_pos, target_quat, emb_sel, attached_obj, attached_obj_scale, eef_markers, env, robot
    )
    attached_obj = {robot.eef_link_names["left"]: cologne.root_link}
    control_gripper(env, robot, attached_obj)
    assert robot._ag_obj_in_hand["left"] == cologne and robot._ag_obj_in_hand["right"] is None

    # Place the cologne (left hand)
    left_hand_pos, left_hand_quat = T.pose_transform(*fridge.get_position_orientation(), *fridge_place_local_pose)
    target_pos = {robot.eef_link_names["left"]: left_hand_pos}
    target_quat = {robot.eef_link_names["left"]: left_hand_quat}
    emb_sel = CuRoboEmbodimentSelection.DEFAULT
    attached_obj = {robot.eef_link_names["left"]: cologne.root_link}
    attached_obj_scale = {robot.eef_link_names["left"]: cologne_scale}
    plan_and_execute_trajectory(
        cmg, target_pos, target_quat, emb_sel, attached_obj, attached_obj_scale, eef_markers, env, robot
    )
    attached_obj = None
    control_gripper(env, robot, attached_obj)
    assert robot._ag_obj_in_hand["left"] is None and robot._ag_obj_in_hand["right"] is None

    og.shutdown()


if __name__ == "__main__":
    test_curobo()
