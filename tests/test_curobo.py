import gc
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.curobo import CuRoboMotionGenerator
from omnigibson.macros import gm, macros
from omnigibson.object_states import Touching
from omnigibson.robots.holonomic_base_robot import HolonomicBaseRobot


def test_curobo():
    # Make sure object states are enabled
    assert gm.ENABLE_OBJECT_STATES

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
                "name": "obj0",
                "primitive_type": "Cube",
                "scale": [0.4, 0.4, 0.4],
                "fixed_base": True,
                "position": [0.5, -0.1, 0.2],
                "orientation": [0, 0, 0, 1],
            },
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
        ],
        "robots": [],
    }

    robot_cfgs = [
        {
            "type": "FrankaPanda",
            "obs_modalities": "rgb",
            "position": [0.7, -0.55, 0.0],
            "orientation": [0, 0, 0.707, 0.707],
            "self_collisions": True,
            "action_normalize": False,
            "controller_config": {
                "arm_0": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": None,
                    "use_delta_commands": False,
                    "use_impedances": True,
                },
                "gripper_0": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": None,
                    "use_delta_commands": False,
                    "use_impedances": True,
                },
            },
        },
        {
            "type": "R1",
            "obs_modalities": "rgb",
            "position": [0.7, -0.55, 0.0],
            "orientation": [0, 0, 0.707, 0.707],
            "self_collisions": True,
            "action_normalize": False,
            "rigid_trunk": False,
            "controller_config": {
                "base": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": None,
                    "use_delta_commands": False,
                    "use_impedances": True,
                },
                "arm_left": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": None,
                    "use_delta_commands": False,
                    "use_impedances": True,
                },
                "arm_right": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": None,
                    "use_delta_commands": False,
                    "use_impedances": True,
                },
                "gripper_left": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": None,
                    "use_delta_commands": False,
                    "use_impedances": True,
                },
                "gripper_right": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": None,
                    "use_delta_commands": False,
                    "use_impedances": True,
                },
            },
        },
        {
            "type": "Tiago",
            "obs_modalities": "rgb",
            "position": [0.7, -0.85, 0],
            "orientation": [0, 0, 0.707, 0.707],
            "self_collisions": True,
            "action_normalize": False,
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
                },
                "arm_right": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": None,
                    "use_delta_commands": False,
                    "use_impedances": True,
                },
                "gripper_left": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": [-1, 1],
                    "use_delta_commands": False,
                    "use_impedances": True,
                },
                "gripper_right": {
                    "name": "JointController",
                    "motor_type": "position",
                    "command_input_limits": [-1, 1],
                    "use_delta_commands": False,
                    "use_impedances": True,
                },
            },
        },
    ]

    for robot_cfg in robot_cfgs:
        cfg["robots"] = [robot_cfg]

        env = og.Environment(configs=cfg)
        robot = env.robots[0]
        obj = env.scene.object_registry("name", "obj0")

        eef_markers = [env.scene.object_registry("name", f"eef_marker_{i}") for i in range(2)]

        if robot.model_name == "R1":
            bottom_links = [
                os.path.join(robot.prim_path, bottom_link)
                for bottom_link in ["wheel_link1", "wheel_link2", "wheel_link3"]
            ]
        elif robot.model_name == "Tiago":
            bottom_links = [
                os.path.join(robot.prim_path, bottom_link)
                for bottom_link in [
                    "base_link",
                    "wheel_front_left_link",
                    "wheel_front_right_link",
                    "wheel_rear_left_link",
                    "wheel_rear_right_link",
                ]
            ]
        else:
            bottom_links = []

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
        batch_size = 10
        n_samples = 10

        cmg = CuRoboMotionGenerator(
            robot=robot,
            batch_size=batch_size,
            debug=False,
            use_cuda_graph=True,
            use_default_embodiment_only=True,
        )

        # Sample values for robot
        th.manual_seed(1)
        lo, hi = robot.joint_lower_limits.clone().view(1, -1), robot.joint_upper_limits.clone().view(1, -1)

        if isinstance(robot, HolonomicBaseRobot):
            lo[0, :2] = -0.1
            lo[0, 2:5] = 0.0
            lo[0, 5] = -math.pi
            hi[0, :2] = 0.1
            hi[0, 2:5] = 0.0
            hi[0, 5] = math.pi

        random_qs = lo + th.rand((n_samples, robot.n_dof)) * (hi - lo)

        # Test collision with the environment (not including self-collisions)
        collision_results = cmg.check_collisions(q=random_qs)

        target_pos, target_quat = defaultdict(list), defaultdict(list)

        floor_plane_prim_paths = {child.GetPath().pathString for child in og.sim.floor_plane._prim.GetChildren()}

        # View results
        false_positive = 0
        false_negative = 0

        target_pos_in_world_frame = defaultdict(list)
        for i, (q, curobo_has_contact) in enumerate(zip(random_qs, collision_results)):
            # Set robot to desired qpos
            robot.set_joint_positions(q)
            robot.keep_still()
            og.sim.step_physics()

            # To debug
            # cmg.save_visualization(robot.get_joint_positions(), "/home/arpit/Downloads/test.obj", emb_sel=emb_sel)

            # Sanity check in the GUI that the robot pose makes sense
            for _ in range(10):
                og.sim.render()

            # Validate that expected collision result is correct
            touching_object = robot.states[Touching].get_value(obj)
            touching_floor = False

            self_collision_pairs = set()
            floor_contact_pairs = set()
            wheel_contact_pairs = set()
            obj_contact_pairs = set()

            for contact in robot.contact_list():
                assert contact.body0 in robot.link_prim_paths
                if contact.body1 in robot.link_prim_paths:
                    self_collision_pairs.add((contact.body0, contact.body1))
                elif contact.body1 in floor_plane_prim_paths:
                    if contact.body0 not in bottom_links:
                        floor_contact_pairs.add((contact.body0, contact.body1))
                    else:
                        wheel_contact_pairs.add((contact.body0, contact.body1))
                elif contact.body1 in obj.link_prim_paths:
                    obj_contact_pairs.add((contact.body0, contact.body1))
                else:
                    assert False, f"Unexpected contact pair: {contact.body0}, {contact.body1}"

            touching_itself = len(self_collision_pairs) > 0
            touching_floor = len(floor_contact_pairs) > 0
            touching_object = len(obj_contact_pairs) > 0

            curobo_has_contact = curobo_has_contact.item()
            physx_has_contact = touching_itself or touching_floor or touching_object

            # cuRobo reports contact, but physx reports no contact
            if curobo_has_contact and not physx_has_contact:
                false_positive += 1
                print(
                    f"False positive {i}: {curobo_has_contact} vs. {physx_has_contact} (touching_itself/obj/floor: {touching_itself}/{touching_object}/{touching_floor})"
                )

            # physx reports contact, but cuRobo reports no contact (this should not happen!)
            elif not curobo_has_contact and physx_has_contact:
                false_negative += 1
                print(
                    f"False negative {i}: {curobo_has_contact} vs. {physx_has_contact} (touching_itself/obj/floor: {touching_itself}/{touching_object}/{touching_floor})"
                )

            if not curobo_has_contact and not physx_has_contact:
                for arm_name in robot.arm_names:
                    # For holonomic base robots, we need to be in the frame of @robot.root_link, not @robot.base_footprint_link
                    if isinstance(robot, HolonomicBaseRobot):
                        base_link_pose = robot.root_link.get_position_orientation()
                        eef_link_pose = robot.eef_links[arm_name].get_position_orientation()
                        eef_pos, eef_quat = T.relative_pose_transform(*eef_link_pose, *base_link_pose)
                    else:
                        eef_pos, eef_quat = robot.get_relative_eef_pose(arm_name)

                    target_pos[robot.eef_link_names[arm_name]].append(eef_pos)
                    target_quat[robot.eef_link_names[arm_name]].append(eef_quat)

                    target_pos_in_world_frame[robot.eef_link_names[arm_name]].append(robot.get_eef_position(arm_name))

        print(
            f"Collision checking false positive: {false_positive / n_samples}, false negative: {false_negative / n_samples}."
        )
        assert (
            false_positive / n_samples == 0.0
        ), f"Collision checking false positive rate: {false_positive / n_samples}, should be == 0.0."
        assert (
            false_negative / n_samples == 0.0
        ), f"Collision checking false positive rate: {false_positive / n_samples}, should be == 0.0."

        env.scene.reset()

        for arm_name in robot.arm_names:
            target_pos[robot.eef_link_names[arm_name]] = th.stack(target_pos[robot.eef_link_names[arm_name]], dim=0)
            target_quat[robot.eef_link_names[arm_name]] = th.stack(target_quat[robot.eef_link_names[arm_name]], dim=0)
            target_pos_in_world_frame[robot.eef_link_names[arm_name]] = th.stack(
                target_pos_in_world_frame[robot.eef_link_names[arm_name]], dim=0
            )

        # Cast defaultdict to dict
        target_pos = dict(target_pos)
        target_quat = dict(target_quat)
        target_pos_in_world_frame = dict(target_pos_in_world_frame)

        print(f"Planning for {len(target_pos[robot.eef_link_names[robot.default_arm]])} eef targets...")

        # Generate collision-free trajectories to the sampled eef poses (including self-collisions)
        successes, traj_paths = cmg.compute_trajectories(
            target_pos=target_pos,
            target_quat=target_quat,
            is_local=True,
            max_attempts=1,
            timeout=60.0,
            ik_fail_return=5,
            enable_finetune_trajopt=True,
            finetune_attempts=1,
            return_full_result=False,
            success_ratio=1.0,
            attached_obj=None,
        )

        # Make sure collision-free trajectories are generated
        success_rate = successes.double().mean().item()
        print(f"Collision-free trajectory generation success rate: {success_rate}")
        assert success_rate == 1.0, f"Collision-free trajectory generation success rate: {success_rate}"

        # 1cm and 3 degrees error tolerance for prismatic and revolute joints, respectively
        error_tol = th.tensor(
            [0.01 if joint.joint_type == "PrismaticJoint" else 3.0 / 180.0 * math.pi for joint in robot.joints.values()]
        )

        for bypass_physics in [True, False]:
            for traj_idx, (success, traj_path) in enumerate(zip(successes, traj_paths)):
                if not success:
                    continue

                # Reset the environment
                env.scene.reset()

                # Move the markers to the desired eef positions
                for marker, arm_name in zip(eef_markers, robot.arm_names):
                    eef_link_name = robot.eef_link_names[arm_name]
                    marker.set_position_orientation(position=target_pos_in_world_frame[eef_link_name][traj_idx])

                q_traj = cmg.path_to_joint_trajectory(traj_path)
                # joint_positions_set_point = []
                # joint_positions_response = []
                for i, q in enumerate(q_traj):
                    if bypass_physics:
                        print(f"Teleporting waypoint {i}/{len(q_traj)}")
                        robot.set_joint_positions(q)
                        robot.keep_still()
                        og.sim.step()
                        for contact in robot.contact_list():
                            assert contact.body0 in robot.link_prim_paths
                            if contact.body1 in floor_plane_prim_paths and contact.body0 in bottom_links:
                                continue
                            print(f"Unexpected contact pair during traj rollout: {contact.body0}, {contact.body1}")
                            assert (
                                False
                            ), f"Unexpected contact pair during traj rollout: {contact.body0}, {contact.body1}"
                    else:
                        # Convert target joint positions to command
                        q = q.cpu()
                        command = []
                        for controller in robot.controllers.values():
                            command.append(q[controller.dof_idx])
                        command = th.cat(command, dim=0)
                        assert command.shape[0] == robot.action_dim

                        num_repeat = 3
                        for j in range(num_repeat):
                            print(f"Executing waypoint {i}/{len(q_traj)}, step {j}")
                            env.step(command)

                            for contact in robot.contact_list():
                                assert contact.body0 in robot.link_prim_paths
                                if contact.body1 in floor_plane_prim_paths and contact.body0 in bottom_links:
                                    continue

                                print(f"Unexpected contact pair during traj rollout: {contact.body0}, {contact.body1}")
                                # Controller is not perfect, so collisions might happen
                                # assert False, f"Unexpected contact pair during traj rollout: {contact.body0}, {contact.body1}"

                            cur_joint_positions = robot.get_joint_positions()
                            if ((cur_joint_positions - q).abs() < error_tol).all():
                                break

        og.clear()

        del cmg
        gc.collect()
        th.cuda.empty_cache()

    og.shutdown()
