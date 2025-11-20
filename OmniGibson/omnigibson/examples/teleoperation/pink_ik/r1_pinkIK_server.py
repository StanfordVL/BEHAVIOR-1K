#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author:    Ji Yingwei
Created:   2025-11-20
Description:
    pink ik server for r1
"""

import multiprocessing
import sys
import meshcat_shapes
import numpy as np
import pinocchio as pin
import qpsolvers

import pink
from pink import solve_ik
from pink.tasks import FrameTask, JointCouplingTask, PostureTask, LowAccelerationTask

from network_ipc_v2 import NetworkIPC

try:
    from loop_rate_limiters import RateLimiter
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Examples use loop rate limiters, " "try `[conda|pip] install loop-rate-limiters`"
    ) from exc


from pinocchio.visualize import MeshcatVisualizer


def writer_process(config, host):
    # acts like your original "client" process
    # time.sleep(1.0)  # small delay so server binds first
    ipc = NetworkIPC("test", config, is_server=False)

    # --- Config ---
    input_file = "/home/j/BEHAVIOR-1K/t.log"  # path to your file

    # --- Regex pattern to capture left/right tensors ---
    teleop_pattern = re.compile(r"TeleopAction\(left=tensor\((.*?)\),\s*right=tensor\((.*?)\)", re.S)

    # --- Read and parse ---
    teleop_actions = []
    with open(input_file, "r") as f:
        content = f.read()

    for left_str, right_str in teleop_pattern.findall(content):
        # Convert each tensor string to list of floats
        left = np.fromstring(left_str.replace("\n", " ").replace("[", "").replace("]", ""), sep=",")
        right = np.fromstring(right_str.replace("\n", " ").replace("[", "").replace("]", ""), sep=",")
        combined = np.concatenate([left, right])
        teleop_actions.append(combined)

    # --- Stack all into one array ---
    teleop_actions = np.stack(teleop_actions)

    for i in range(len(teleop_actions)):
        # send cam_high to server for next round
        ipc.set("teleop_actions", np.array([teleop_actions[i]]))

        ipc.talk()  # fetch server snapshot (contains 'state')

    # ipc.commit()
    ipc.close()
    print("client close")


if __name__ == "__main__":
    urdf_path = "/home/j/BEHAVIOR-1K/datasets/omnigibson-robot-assets/models/r1/urdf/r1.urdf"

    mesh_path = "/home/j/BEHAVIOR-1K/datasets/omnigibson-robot-assets/models/r1/urdf"

    # 1. Build the kinematic/dynamic model
    model = pin.buildModelFromUrdf(urdf_path)

    # 2. Build collision and visual geometry models
    collision_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.COLLISION, package_dirs=[mesh_path])
    visual_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.VISUAL, package_dirs=[mesh_path])

    try:
        viz = MeshcatVisualizer(model, collision_model, visual_model)
        viz.initViewer(open=True)
    except ImportError as err:
        print("Error while initializing the viewer. " "It seems you should install Python meshcat")
        print(err)
        sys.exit(0)

    # Load the robot in the viewer.
    viz.loadViewerModel()

    # Create data required by the algorithms
    data = model.createData()

    # Sample a random configuration
    q0 = pin.neutral(model)

    # Initialize visualization
    # viz = start_meshcat_visualizer(robot)
    # show both real and target for each hand

    # show both real and target for each hand
    r_target = viz.viewer["right_hand_target"]
    r_current = viz.viewer["right_hand_current"]
    l_target = viz.viewer["left_hand_target"]
    l_current = viz.viewer["left_hand_current"]
    torso_link1 = viz.viewer["torso_link4"]
    torso_link1_target = viz.viewer["torso_link4_t"]

    # 用不同颜色或大小区分
    meshcat_shapes.frame(r_target, axis_length=0.15)  # 目标姿态
    meshcat_shapes.frame(r_current, axis_length=0.10)  # 实际姿态
    meshcat_shapes.frame(l_target, axis_length=0.15)
    meshcat_shapes.frame(l_current, axis_length=0.10)
    meshcat_shapes.frame(torso_link1, axis_length=0.10)
    meshcat_shapes.frame(torso_link1_target, axis_length=0.10)

    # Set initial robot configuration
    configuration = pink.Configuration(model, data, q0)
    viz.display(configuration.q)

    # Tasks initialization for IK

    right_wrist_task = FrameTask(
        "right_arm_link6",
        position_cost=1.0,
        orientation_cost=0.3,
    )
    left_wrist_task = FrameTask(
        "left_arm_link6",
        position_cost=1.0,
        orientation_cost=0.3,
    )
    posture_task = PostureTask(
        cost=1e-1,  # [cost] / [rad]
    )

    smooth_task = LowAccelerationTask(cost=0.5)

    pelvis_task = FrameTask(
        "torso_link4",
        position_cost=[2, 2, 1.6],
        orientation_cost=1.0,
    )

    tasks = [
        # left_foot_task,
        pelvis_task,
        # right_foot_task,
        right_wrist_task,
        left_wrist_task,
        posture_task,
        smooth_task,
        # l_knee_holonomic_task,
        # r_knee_holonomic_task,
    ]

    posture_task.set_target_from_configuration(configuration)
    smooth_task.set_target_from_configuration(configuration)

    torso_link4_pose_raw = configuration.get_transform_frame_to_world("torso_link4")
    torso_link4_pose_raw.translation[2] = 0.9

    pelvis_task.set_target(torso_link4_pose_raw)

    torso_link1_target.set_transform(torso_link4_pose_raw.np)

    print('configuration.get_transform_frame_to_world("torso_link4")', torso_link4_pose_raw.translation)

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "daqp" in qpsolvers.available_solvers:
        solver = "daqp"

    rate = RateLimiter(frequency=30.0, warn=False)
    dt = rate.period
    t = 0.0  # [s]

    def xyz_axisangle1dof_to_SE3_and_grip(v: np.ndarray):
        """
        v: shape (7,) numpy array
        [x, y, z, wx, wy, wz, grip], where w* is axis-angle (radians)
        returns: (pin.SE3, float)
        """
        v = np.asarray(v, dtype=float)
        p = v[:3]  # translation
        # p[2] +=0.7
        w = v[3:6]  # axis-angle vector (so3 exponential coordinates)
        grip = float(v[6])  # keep gripper separate from SE3

        R = pin.exp3(w)  # so3 -> SO3

        T = pin.SE3(R, p)
        return T, grip

    # Example with your data
    left = np.array([0.6868, 0.1752, 1.2041, -0.2615, 0.8972, 0.0685, -0.0000])
    right = np.array([0.6850, -0.2317, 1.1989, -0.6491, -5.1229, 1.3787, -0.0000])

    T_left, grip_left = xyz_axisangle1dof_to_SE3_and_grip(left)
    T_right, grip_right = xyz_axisangle1dof_to_SE3_and_grip(right)

    import re
    import numpy as np

    i = 0

    config = {"state": ((1, 16), np.float32), "teleop_actions": ((1, 14), np.float32)}

    ipc = NetworkIPC("test", config, is_server=True)

    # writer = multiprocessing.Process(target=writer_process, args=(config,0))

    # start server then client (order matters for bind/connect)

    # writer.start()

    while True:
        # --- 更新目标 ---

        ipc.talk()  # wait until client sent UPDATE (cam_high)
        teleop_actions = ipc.get("teleop_actions")

        left = teleop_actions[0][:7]
        right = teleop_actions[0][7:14]

        def rotate_point_x90(v):
            # rotation matrix around X by +90 degrees
            R_x90 = pin.exp3(np.array([np.pi / 2, 0, 0]))  # shape (3,3)
            return R_x90 @ v

        def rotate_point_z_neg90(v):
            R_z_neg90 = pin.exp3(np.array([0, 0, -np.pi / 2]))  # −90° about Z
            return R_z_neg90 @ v

        # original position
        p_left = left[:3]
        p_right = right[:3]

        # rotated positions
        p_left_rot = rotate_point_z_neg90(rotate_point_x90(p_left))
        p_right_rot = rotate_point_z_neg90(rotate_point_x90(p_right))

        p_left_rot[0] += 0.2
        p_right_rot[0] += 0.2

        p_left_rot[2] += 0.3
        p_right_rot[2] += 0.3

        left[3:6] = rotate_point_z_neg90(rotate_point_x90(left[3:6]))
        right[3:6] = rotate_point_z_neg90(rotate_point_x90(right[3:6]))

        # re-assemble into 7-D vector (keep same orientation & grip)
        left_rot = np.concatenate([p_left_rot, left[3:]])
        right_rot = np.concatenate([p_right_rot, right[3:]])

        T_left, grip_left = xyz_axisangle1dof_to_SE3_and_grip(left_rot)
        T_right, grip_right = xyz_axisangle1dof_to_SE3_and_grip(right_rot)

        # 定义绕X轴旋转90度的旋转矩阵
        R_x_90 = pin.exp3(np.array([0, np.pi / 2 + 30.0 / 180 * np.pi, 0]))

        # 在原姿态基础上右乘该旋转
        T_left.rotation = T_left.rotation @ R_x_90
        T_right.rotation = T_right.rotation @ R_x_90

        right_wrist_task.set_target(T_right)
        left_wrist_task.set_target(T_left)

        # --- 实际姿态 (FK from current q) ---
        T_r_real = configuration.get_transform_frame_to_world("right_arm_link6")
        T_l_real = configuration.get_transform_frame_to_world("left_arm_link6")

        # --- MeshCat 可视化 ---
        r_target.set_transform(T_right.np)
        l_target.set_transform(T_left.np)
        r_current.set_transform(T_r_real.np)
        l_current.set_transform(T_l_real.np)

        torso_link1_real = configuration.get_transform_frame_to_world("torso_link4")
        torso_link1.set_transform(torso_link1_real.np)

        # Compute velocity and integrate it into next configuration
        velocity = solve_ik(configuration, tasks, dt, solver=solver)
        configuration.integrate_inplace(velocity, dt)

        print(configuration.q)

        q_result = np.concatenate([configuration.q[9:19], configuration.q[21:27]])
        q_result = np.array([q_result])

        ipc.set("state", q_result)

        # Visualize result at fixed FPS
        viz.display(configuration.q)
        rate.sleep()
        t += dt
        i += 1

    writer.join()
