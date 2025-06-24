import numpy as np
import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.robots import LocomotionRobot
from omnigibson.utils.backend_utils import _compute_backend as cb


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
                "name": "robot0",
                "obs_modalities": [],
                "position": [150, 150, 100],
                "orientation": [0, 0, 0, 1],
                "action_normalize": False,
            },
            {
                "type": "Fetch",
                "name": "robot1",
                "obs_modalities": [],
                "position": [150, 150, 105],
                "orientation": [0, 0, 0, 1],
                "action_normalize": False,
            },
            {
                "type": "Tiago",
                "name": "robot2",
                "obs_modalities": [],
                "position": [150, 150, 110],
                "orientation": [0, 0, 0, 1],
                "action_normalize": False,
            },
            {
                "type": "A1",
                "name": "robot3",
                "obs_modalities": [],
                "position": [150, 150, 115],
                "orientation": [0, 0, 0, 1],
                "action_normalize": False,
            },
            {
                "type": "R1",
                "name": "robot4",
                "obs_modalities": [],
                "position": [150, 150, 120],
                "orientation": [0, 0, 0, 1],
                "action_normalize": False,
            },
        ],
    }

    env = og.Environment(configs=cfg)

    # Define error functions to use
    def check_zero_error(curr_position, init_position, tol=1e-2):
        return th.norm(curr_position - init_position).item() < tol

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

    def check_ori_error(curr_orientation, init_orientation, tol=0.1):
        ori_err_normalized = th.norm(
            T.quat2axisangle(T.mat2quat(T.quat2mat(init_orientation).T @ T.quat2mat(curr_orientation)))
        ).item() / (np.pi * 2)
        ori_err = np.abs(np.pi * 2 * (np.round(ori_err_normalized) - ori_err_normalized))
        return ori_err < tol

    # All functions take in (target, curr, init) tuple
    err_checks = {
        "pose_delta_ori": {
            "zero": {
                "pos": lambda target, curr, init: check_zero_error(curr, init),
                "ori": lambda target, curr, init: check_ori_error(curr, init),
            },
            "forward": {
                "pos": lambda target, curr, init: check_forward_error(curr, init),
                "ori": lambda target, curr, init: check_ori_error(curr, init),
            },
            "side": {
                "pos": lambda target, curr, init: check_side_error(curr, init),
                "ori": lambda target, curr, init: check_ori_error(curr, init),
            },
            "up": {
                "pos": lambda target, curr, init: check_up_error(curr, init),
                "ori": lambda target, curr, init: check_ori_error(curr, init),
            },
            "rotate": {
                "pos": lambda target, curr, init: check_zero_error(curr, init),
                "ori": None,
            },
            "base_move": {
                "pos": lambda target, curr, init: check_zero_error(
                    curr, init, tol=0.02
                ),  # Slightly bigger tolerance with base moving
                "ori": lambda target, curr, init: check_ori_error(curr, init),
            },
        },
        "absolute_pose": {
            "zero": {
                "pos": lambda target, curr, init: check_zero_error(target, curr, tol=5e-3),
                "ori": lambda target, curr, init: check_ori_error(target, curr),
            },
            "forward": {
                "pos": lambda target, curr, init: check_zero_error(target, curr, tol=5e-3),
                "ori": lambda target, curr, init: check_ori_error(target, curr),
            },
            "side": {
                "pos": lambda target, curr, init: check_zero_error(target, curr, tol=5e-3),
                "ori": lambda target, curr, init: check_ori_error(target, curr),
            },
            "up": {
                "pos": lambda target, curr, init: check_zero_error(target, curr, tol=5e-3),
                "ori": lambda target, curr, init: check_ori_error(target, curr),
            },
            "rotate": {
                "pos": lambda target, curr, init: check_zero_error(target, curr, tol=5e-3),
                "ori": lambda target, curr, init: check_ori_error(target, curr, tol=0.05),
            },
            "base_move": {
                "pos": lambda target, curr, init: check_zero_error(target, curr, tol=5e-3),
                "ori": lambda target, curr, init: check_ori_error(target, curr),
            },
        },
    }

    n_steps = {
        "pose_delta_ori": {
            "zero": 40,
            "forward": 40,
            "side": 40,
            "up": 40,
            "rotate": 20,
            "base_move": 30,
        },
        "absolute_pose": {
            "zero": 50,
            "forward": 50,
            "side": 50,
            "up": 50,
            "rotate": 50,
            "base_move": 50,
        },
    }

    # Position the robots, reset them, and keep them still
    for i, robot in enumerate(env.robots):
        robot.set_position_orientation(
            position=th.tensor([0.0, i * 5.0, 0.0]), orientation=T.euler2quat(th.tensor([0.0, 0.0, np.pi / 3]))
        )
        robot.reset()

    # Take 10 steps to stabilize
    for _ in range(10):
        og.sim.step()

    # Keep all robots still
    for robot in env.robots:
        robot.keep_still()

        # We need to explicitly reset the controllers to unify the initial state that will be seen
        # during downstream action executions -- i.e.: the state seen after robot.reload_controllers()
        # is called each time
        for controller in robot.controllers.values():
            controller.reset()

    # Update initial state (robot should be stable and still)
    env.scene.update_initial_file()

    env.scene.reset()

    # Record initial eef pose of all robots
    initial_eef_pose = dict()
    for i, robot in enumerate(env.robots):
        initial_eef_pose[robot.name] = {arm: robot.get_relative_eef_pose(arm=arm) for arm in robot.arm_names}

    for controller in ["InverseKinematicsController", "OperationalSpaceController"]:
        for controller_mode in ["pose_delta_ori", "absolute_pose"]:
            controller_kwargs = {
                "mode": controller_mode,
            }
            if controller_mode == "absolute_pose":
                controller_kwargs["command_input_limits"] = None
                controller_kwargs["command_output_limits"] = None
            actions = {
                "zero": dict(),
                "forward": dict(),
                "side": dict(),
                "up": dict(),
                "rotate": dict(),
                "base_move": dict(),
            }

            # Load the initial state without stepping physics
            env.scene.load_state(env.scene._initial_file["state"])

            for i, robot in enumerate(env.robots):
                controller_config = {f"arm_{arm}": {"name": controller, **controller_kwargs} for arm in robot.arm_names}
                robot.reload_controllers(controller_config)

                # Define actions to use
                zero_action = th.zeros(robot.action_dim)
                forward_action = th.zeros(robot.action_dim)
                side_action = th.zeros(robot.action_dim)
                up_action = th.zeros(robot.action_dim)
                rot_action = th.zeros(robot.action_dim)

                # Populate actions and targets
                for arm in robot.arm_names:
                    c_name = f"arm_{arm}"
                    start_idx = 0
                    init_eef_pos, init_eef_quat = initial_eef_pose[robot.name][arm]
                    for c in robot.controller_order:
                        if c == c_name:
                            break
                        start_idx += robot.controllers[c].command_dim
                    if controller_mode == "pose_delta_ori":
                        forward_action[start_idx] = 0.02
                        side_action[start_idx + 1] = 0.02
                        up_action[start_idx + 2] = 0.02
                        rot_action[start_idx + 3] = 0.02
                    elif controller_mode == "absolute_pose":
                        for act in [zero_action, forward_action, side_action, up_action, rot_action]:
                            act[start_idx : start_idx + 3] = init_eef_pos.clone()
                            act[start_idx + 3 : start_idx + 6] = T.quat2axisangle(init_eef_quat.clone())
                        forward_action[start_idx] += 0.1
                        side_action[start_idx + 1] += 0.1
                        up_action[start_idx + 2] += 0.1
                        rot_action[start_idx + 3 : start_idx + 6] = T.quat2axisangle(
                            T.quat_multiply(T.euler2quat(th.tensor([th.pi / 10, 0, 0])), init_eef_quat.clone())
                        )

                    else:
                        raise ValueError(f"Got invalid controller mode: {controller}")
                actions["zero"][robot.name] = zero_action
                actions["forward"][robot.name] = forward_action
                actions["side"][robot.name] = side_action
                actions["up"][robot.name] = up_action
                actions["rotate"][robot.name] = rot_action

                # Add base movement action if locomotion robot
                base_move_action = zero_action.clone()
                if isinstance(robot, LocomotionRobot):
                    c_name = "base"
                    start_idx = 0
                    for c in robot.controller_order:
                        if c == c_name:
                            break
                        start_idx += robot.controllers[c].command_dim
                    base_move_action[start_idx] = 0.1
                actions["base_move"][robot.name] = base_move_action

            # Update the state (e.g. goal, which is None) of the new controllers to the initial state
            # This step is crucial because if env.reset() is called directly, we will load the state of the old controllers and step physics,
            # which causes can cause errors because the goal is obsolete.
            env.scene.update_initial_file()

            # For each action set, reset the scene, then run actions and see if arm moves in expected way
            for action_name, action in actions.items():
                env.scene.reset()

                # Take N steps with given action and check for error
                for _ in range(n_steps[controller_mode][action_name]):
                    env.step(action)

                for i, robot in enumerate(env.robots):
                    for arm in robot.arm_names:
                        # Make sure no arm joints are at their limit
                        normalized_qpos = robot.get_joint_positions(normalized=True)[robot.arm_control_idx[arm]]
                        assert not th.any(th.abs(normalized_qpos) == 1.0), (
                            f"controller [{controller}], mode [{controller_mode}], robot [{robot.model_name}], arm [{arm}], action [{action_name}]:\n"
                            f"Some joints are at their limit (normalized values): {normalized_qpos}"
                        )

                        init_pos, init_quat = initial_eef_pose[robot.name][arm]
                        curr_pos, curr_quat = robot.get_relative_eef_pose(arm=arm)
                        arm_controller = robot.controllers[f"arm_{arm}"]
                        arm_goal = arm_controller.goal
                        target_pos = cb.to_torch(arm_goal["target_pos"])
                        target_quat = T.mat2quat(cb.to_torch(arm_goal["target_ori_mat"]))
                        pos_check = err_checks[controller_mode][action_name]["pos"]
                        if pos_check is not None:
                            is_valid_pos = pos_check(target_pos, curr_pos, init_pos)
                            assert is_valid_pos, (
                                f"Got mismatch for controller [{controller}], mode [{controller_mode}], robot [{robot.model_name}], action [{action_name}]\n"
                                f"target_pos: {target_pos}, curr_pos: {curr_pos}, init_pos: {init_pos}"
                            )
                        ori_check = err_checks[controller_mode][action_name]["ori"]
                        if ori_check is not None:
                            is_valid_ori = ori_check(target_quat, curr_quat, init_quat)
                            assert is_valid_ori, (
                                f"Got mismatch for controller [{controller}], mode [{controller_mode}], robot [{robot.model_name}], action [{action_name}]\n"
                                f"target_quat: {target_quat}, curr_quat: {curr_quat}, init_quat: {init_quat}"
                            )
