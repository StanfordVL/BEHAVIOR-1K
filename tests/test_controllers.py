import numpy as np
import pytest
import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.robots import LocomotionRobot


def test_arm_control():
    # Create env
    cfg = {
        "scene": {
            "type": "Scene",
        },
        "objects": [],
        "robots": [
            {
                "type": "FrankaPanda",
                "obs_modalities": "rgb",
                "position": [150, 150, 100],
                "orientation": [0, 0, 0, 1],
            },
            {
                "type": "Fetch",
                "obs_modalities": "rgb",
                "position": [150, 150, 105],
                "orientation": [0, 0, 0, 1],
            },
            {
                "type": "Tiago",
                "obs_modalities": "rgb",
                "position": [150, 150, 110],
                "orientation": [0, 0, 0, 1],
            },
        ],
    }
    env = og.Environment(configs=cfg)

    # Define error functions to use
    def check_zero_error(curr_position, init_position, tol=1e-2):
        return th.norm(curr_pos - init_pos).item() < tol

    def check_forward_error(curr_position, init_position, tol=1e-2, forward_tol=1e-2):
        # x should be positive
        return (curr_position[0] - init_position[0]).item() > forward_tol and th.norm(
            curr_position[[1, 2]] - init_position[[1, 2]]
        ).item() < tol

    def check_side_error(curr_position, init_position, tol=1e-2, side_tol=1e-2):
        # y should be positive
        return (curr_position[1] - init_position[1]).item() > side_tol and th.norm(
            curr_position[[0, 2]] - init_position[[0, 2]]
        ).item() < tol

    def check_up_error(curr_position, init_position, tol=1e-2, up_tol=1e-2):
        # z should be positive
        return (curr_position[2] - init_position[2]).item() > up_tol and th.norm(
            curr_position[[0, 1]] - init_position[[0, 1]]
        ).item() < tol

    pos_err_checks = {
        "zero": check_zero_error,
        "forward": check_forward_error,
        "side": check_side_error,
        "up": check_up_error,
        "base_move": lambda x, y: check_zero_error(x, y, tol=0.02),  # Slightly bigger tolerance with base moving
    }

    n_steps = {
        "zero": 10,
        "forward": 10,
        "side": 10,
        "up": 10,
        "base_move": 30,  # Slightly bigger tolerance with base moving
    }

    for controller in ["InverseKinematicsController", "OperationalSpaceController"]:
        actions = {
            "zero": dict(),
            "forward": dict(),
            "side": dict(),
            "up": dict(),
            "base_move": dict(),
        }
        for i, robot in enumerate(env.robots):
            controller_config = {f"arm_{arm}": {"name": controller} for arm in robot.arm_names}
            robot.set_position_orientation(
                th.tensor([0.0, i * 5.0, 0.0]), T.euler2quat(th.tensor([0.0, 0.0, np.pi / 3]))
            )
            robot.reset()
            robot.keep_still()
            robot.reload_controllers(controller_config)

            # Define actions to use
            zero_action = th.zeros(robot.action_dim)
            forward_action = th.zeros(robot.action_dim)
            side_action = th.zeros(robot.action_dim)
            up_action = th.zeros(robot.action_dim)
            for arm in robot.arm_names:
                c_name = f"arm_{arm}"
                start_idx = 0
                for c in robot.controller_order:
                    if c == c_name:
                        break
                    start_idx += robot.controllers[c].command_dim
                forward_action[start_idx] = 0.1
                side_action[start_idx + 1] = 0.1
                up_action[start_idx + 2] = 0.1
            actions["zero"][robot.name] = zero_action
            actions["forward"][robot.name] = forward_action
            actions["side"][robot.name] = side_action
            actions["up"][robot.name] = up_action

            # Add base movement action if locomotion robot
            base_move_action = th.zeros(robot.action_dim)
            if isinstance(robot, LocomotionRobot):
                c_name = "base"
                start_idx = 0
                for c in robot.controller_order:
                    if c == c_name:
                        break
                    start_idx += robot.controllers[c].command_dim
                base_move_action[start_idx] = 0.1
            actions["base_move"][robot.name] = base_move_action

        # Take 5 steps
        for _ in range(5):
            env.step(actions["zero"])

        # Update initial state
        env.scene.update_initial_state()

        # For each action set, reset all robots, then run actions and see if arm moves in expected way
        for action_name, action in actions.items():

            # Reset env
            env.reset()

            # Record initial poses
            initial_eef_pose = dict()
            for i, robot in enumerate(env.robots):
                initial_eef_pose[robot.name] = {arm: robot.get_relative_eef_pose(arm=arm) for arm in robot.arm_names}

            # Take 10 steps with given action and check for error
            for _ in range(n_steps[action_name]):
                env.step(action)

            for i, robot in enumerate(env.robots):
                for arm in robot.arm_names:
                    init_pos, init_quat = initial_eef_pose[robot.name][arm]
                    curr_pos, curr_quat = robot.get_relative_eef_pose(arm=arm)
                    is_valid_pos = pos_err_checks[action_name](curr_pos, init_pos)
                    assert is_valid_pos, (
                        f"Got mismatch for controller [{controller}], robot [{robot.model_name}], action [{action_name}]\n"
                        f"curr_pos: {curr_pos}, init_pos: {init_pos}"
                    )
                    ori_err_normalized = th.norm(
                        T.quat2axisangle(T.mat2quat(T.quat2mat(init_quat).T @ T.quat2mat(curr_quat)))
                    ).item() / (np.pi * 2)
                    ori_err = np.abs(np.pi * 2 * (np.round(ori_err_normalized) - ori_err_normalized))
                    assert ori_err < 0.1, (
                        f"Got mismatch for controller [{controller}], robot [{robot.model_name}], action [{action_name}]\n"
                        f"curr_quat: {curr_quat}, init_quat: {init_quat}, err: {ori_err}"
                    )
