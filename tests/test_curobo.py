import pytest
import torch as th

import omnigibson as og
from omnigibson.action_primitives.curobo import CuRoboMotionGenerator
from omnigibson.macros import gm
from omnigibson.object_states import Touching


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
        ],
        "robots": [
            {
                "type": "FrankaPanda",
                "obs_modalities": "rgb",
                "position": [0.7, -0.55, 0.0],
                "orientation": [0, 0, 0.707, 0.707],
                "self_collisions": True,
            },
        ],
    }
    env = og.Environment(configs=cfg)
    robot = env.robots[0]
    obj = env.scene.object_registry("name", "obj0")

    robot.reset()
    robot.keep_still()

    for _ in range(5):
        og.sim.step()

    # Create CuRobo instance
    batch_size = 25
    n_samples = 50
    cmg = CuRoboMotionGenerator(
        robot=robot,
        batch_size=batch_size,
    )

    # Sample values for robot
    th.manual_seed(1)
    lo, hi = robot.joint_lower_limits.view(1, -1), robot.joint_upper_limits.view(1, -1)
    random_qs = lo + th.rand((n_samples, robot.n_dof)) * (hi - lo)

    # Test collision with the environment (not including self-collisions)
    collision_results = cmg.check_collisions(q=random_qs, activation_distance=0.0)
    eef_positions, eef_quats = [], []

    floor_plane_prim_paths = {child.GetPath().pathString for child in og.sim.floor_plane._prim.GetChildren()}

    # View results
    n_mismatch = 0
    for i, (q, result) in enumerate(zip(random_qs, collision_results)):
        # Set robot to desired qpos
        robot.set_joint_positions(q)
        robot.keep_still()
        og.sim.step()

        # Validate that expected collision result is correct
        touching_objcet = robot.states[Touching].get_value(obj)
        contact_bodies = set()
        for contact in robot.contact_list():
            contact_bodies.update({contact.body0, contact.body1})
        touching_floor = len(contact_bodies & floor_plane_prim_paths) > 0

        true_result = touching_objcet or touching_floor

        if result.item() != true_result:
            n_mismatch += 1
            print(
                f"Mismatch {i}: {result.item()} vs. {true_result} (touching_objcet: {touching_objcet}, touching_floor: {touching_floor})"
            )

        # If we're collision-free, record this pose so that we can test trajectory planning afterwards
        if not result and len(robot.contact_list()) == 0:
            eef_pos, eef_quat = robot.get_relative_eef_pose()
            eef_positions.append(eef_pos)
            eef_quats.append(eef_quat)

    assert n_mismatch / n_samples == 0.0, f"Check collision mismatch rate: {n_mismatch / n_samples}"

    # Test trajectories
    robot.reset()
    robot.keep_still()
    og.sim.step()

    # Generate collision-free trajectories to the sampled eef poses (including self-collisions)
    successes, traj_paths = cmg.compute_trajectories(
        target_pos=th.stack(eef_positions, dim=0),
        target_quat=th.stack(eef_quats, dim=0),
        is_local=True,
        max_attempts=1,
        enable_finetune_trajopt=True,
        return_full_result=False,
        success_ratio=1.0,
        attached_obj=None,
    )

    # Make sure collision-free trajectories are generated
    success_rate = th.mean(successes)
    assert success_rate == 1.0, f"Collision-free trajectory generation success rate: {success_rate}"

    print(f"Collision-free trajectory generation success rate: {success_rate}")

    for success, traj_path in zip(successes, traj_paths):
        if not success:
            continue
        q_traj = cmg.path_to_joint_trajectory(traj_path)
        for q in q_traj:
            robot.set_joint_positions(q)
            robot.keep_still()
            og.sim.step()
            assert len(robot.contact_list()) == 0
