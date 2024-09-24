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
    n_samples = 55
    cmg = CuRoboMotionGenerator(
        robot=robot,
        batch_size=batch_size,
    )

    # Sample values for robot
    th.manual_seed(1)
    lo, hi = robot.joint_lower_limits.view(1, -1), robot.joint_upper_limits.view(1, -1)
    random_qs = lo + th.rand((n_samples, robot.n_dof)) * (hi - lo)

    # Test collision
    collision_results = cmg.check_collisions(q=random_qs, activation_distance=0.0)
    eef_positions, eef_quats = [], []

    # View results
    n_mismatch = 0
    for i, (q, result) in enumerate(zip(random_qs, collision_results)):
        # Set robot to desired qpos
        robot.set_joint_positions(q)
        robot.keep_still()
        og.sim.step()

        # Validate that expected collision result is correct
        true_result = robot.states[Touching].get_value(obj)

        if result.item() != true_result:
            n_mismatch += 1

        # If we're collision-free, record this pose so that we can test trajectory planning afterwards
        if not result and len(robot.contact_list()) == 0:
            eef_pos, eef_quat = robot.get_relative_eef_pose()
            eef_positions.append(eef_pos)
            eef_quats.append(eef_quat)

    # Make sure mismatched results are small
    # Slight mismatch may occur because sphere approximation is not quite equal to the collision sim representation
    assert n_mismatch / n_samples < 0.1, f"Proportion mismatched results: {n_mismatch / n_samples}"

    # Test trajectories
    robot.reset()
    robot.keep_still()
    og.sim.step()

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

    # Execute the trajectory and make sure there's rarely any collisions
    assert th.sum(successes) > 0.95, f"Failed to find > 95% collision-free trajectories: {successes}"
    print(f"Total successes: {th.sum(successes)} / {len(successes)}")
    for success, traj_path in zip(successes, traj_paths):
        if not success:
            continue
        q_traj = cmg.path_to_joint_trajectory(traj_path)
        for q in q_traj:
            robot.set_joint_positions(q)
            robot.keep_still()
            og.sim.step()
            assert len(robot.contact_list()) == 0
