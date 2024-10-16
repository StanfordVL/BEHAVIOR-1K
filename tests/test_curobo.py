import math
import os
from collections import defaultdict

import pytest
import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.action_primitives.curobo import CuRoboMotionGenerator
from omnigibson.macros import gm
from omnigibson.object_states import Touching
from omnigibson.robots.holonomic_base_robot import HolonomicBaseRobot


def test_curobo():
    # Make sure object states are enabled
    assert gm.ENABLE_OBJECT_STATES

    # Create env
    cfg = {
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
        # {
        #     "type": "FrankaPanda",
        #     "obs_modalities": "rgb",
        #     "position": [0.7, -0.55, 0.0],
        #     "orientation": [0, 0, 0.707, 0.707],
        #     "self_collisions": True,
        # },
        {
            "type": "R1",
            "obs_modalities": "rgb",
            "position": [0.7, -0.55, 0.0],
            "orientation": [0, 0, 0.707, 0.707],
            "self_collisions": True,
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
        else:
            bottom_links = []

        robot.reset()
        robot.keep_still()

        for _ in range(5):
            og.sim.step()

        env.scene.update_initial_state()
        env.scene.reset()

        # Create CuRobo instance
        batch_size = 25
        n_samples = 50

        cmg = CuRoboMotionGenerator(
            robot=robot,
            batch_size=batch_size,
            debug=False,
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
        collision_results = cmg.check_collisions(q=random_qs, activation_distance=0.0)

        eef_positions, eef_quats = [], []
        additional_eef_positions, additional_eef_quats = defaultdict(list), defaultdict(list)

        floor_plane_prim_paths = {child.GetPath().pathString for child in og.sim.floor_plane._prim.GetChildren()}

        # View results
        false_positive = 0
        false_negative = 0

        absolute_eef_positions = []
        for i, (q, curobo_has_contact) in enumerate(zip(random_qs, collision_results)):
            # Set robot to desired qpos
            robot.set_joint_positions(q)
            robot.keep_still()
            og.sim.step_physics()

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
                absolute_eef_position = []
                for arm_name in robot.arm_names:
                    # For holonomic base robots, we need to be in the frame of @robot.root_link, not @robot.base_footprint_link
                    if isinstance(robot, HolonomicBaseRobot):
                        base_link_pose = robot.root_link.get_position_orientation()
                        eef_link_pose = robot.eef_links[arm_name].get_position_orientation()
                        eef_pos, eef_quat = T.relative_pose_transform(*eef_link_pose, *base_link_pose)
                    else:
                        eef_pos, eef_quat = robot.get_relative_eef_pose(arm_name)

                    if arm_name == robot.default_arm:
                        eef_positions.append(eef_pos)
                        eef_quats.append(eef_quat)
                    else:
                        additional_eef_positions[arm_name].append(eef_pos)
                        additional_eef_quats[arm_name].append(eef_quat)

                    absolute_eef_position.append(robot.get_eef_position(arm_name))

                absolute_eef_positions.append(absolute_eef_position)

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

        print(f"Planning for {len(eef_positions)} eef targets...")

        # TODO: BoundCost needs to remove .clone() call for joint limits
        # cmg.mg.kinematics.kinematics_config.joint_limits.position[:, 0] = 0.0

        # Generate collision-free trajectories to the sampled eef poses (including self-collisions)
        successes, traj_paths = cmg.compute_trajectories(
            target_pos=th.stack(eef_positions, dim=0),
            target_quat=th.stack(eef_quats, dim=0),
            right_target_pos=(
                th.stack(additional_eef_positions["right"], dim=0) if "right" in additional_eef_positions else None
            ),
            right_target_quat=(
                th.stack(additional_eef_quats["right"], dim=0) if "right" in additional_eef_quats else None
            ),
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

        for success, traj_path, absolute_eef_pos in zip(successes, traj_paths, absolute_eef_positions):
            if not success:
                continue
            for pos, marker in zip(absolute_eef_pos, eef_markers):
                marker.set_position_orientation(position=pos)

            q_traj = cmg.path_to_joint_trajectory(traj_path)

            for q in q_traj:
                robot.set_joint_positions(q)
                robot.keep_still()
                og.sim.step()

                for contact in robot.contact_list():
                    assert contact.body0 in robot.link_prim_paths
                    if contact.body1 in floor_plane_prim_paths and contact.body0 in bottom_links:
                        continue

                    assert False, f"Unexpected contact pair during traj rollout: {contact.body0}, {contact.body1}"

        og.clear()

    og.shutdown()


if __name__ == "__main__":
    test_curobo()
