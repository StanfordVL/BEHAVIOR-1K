import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from dm_control import mjcf
from dm_control.utils.inverse_kinematics import qpos_from_site_pose

from gello.agents.agent import Agent
from gello.dm_control_tasks.arms.ur5e import UR5e

# mujoco has a slightly different coordinate system than UR control box
mj2ur = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
ur2mj = np.linalg.inv(mj2ur)

# cartensian space control, controller <> robot relative pose matters. This extrinsics is based on
# our setup, for details please checkout the project page.
spacemouse2ur = np.array(
    [
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)
ur2spacemouse = np.linalg.inv(spacemouse2ur)


def apply_transfer(mat: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    # xyz can be 3dim or 4dim (homogeneous) or can be a rotation matrix
    if len(xyz) == 3:
        xyz = np.append(xyz, 1)
    return np.matmul(mat, xyz)[:3]


@dataclass
class SpacemouseConfig:
    angle_scale: float = 0.24
    translation_scale: float = 0.06
    # only control the xyz, rotation direction, not the gripper
    invert_control: np.ndarray = np.ones(6)
    rotation_mode: str = "euler"


class SpacemouseAgent(Agent):
    def __init__(
        self,
        robot_type: str,
        config: SpacemouseConfig = SpacemouseConfig(),
        device_path: Optional[str] = None,
        verbose: bool = True,
        invert_button: bool = False,
    ) -> None:
        self.config = config
        self.last_state_lock = threading.Lock()
        self._invert_button = invert_button
        # example state:SpaceNavigator(t=3.581528532, x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, buttons=[0, 0])
        #  all continuous inputs range from 0-1, buttons are 0 or 1
        self.spacemouse_latest_state = None
        self._device_path = device_path
        spacemouse_thread = threading.Thread(target=self._read_from_spacemouse)
        spacemouse_thread.start()
        self._verbose = verbose
        if self._verbose:
            print(f"robot_type: {robot_type}")
        if robot_type == "ur5":
            _robot = UR5e()
        else:
            raise ValueError(f"Unknown robot type: {robot_type}")
        self.physics = mjcf.Physics.from_mjcf_model(_robot.mjcf_model)

    def _read_from_spacemouse(self):
        import pyspacemouse

        if self._device_path is None:
            mouse = pyspacemouse.open()
        else:
            mouse = pyspacemouse.open(path=self._device_path)
        if mouse:
            while 1:
                state = mouse.read()
                with self.last_state_lock:
                    self.spacemouse_latest_state = state
                time.sleep(0.001)
        else:
            raise ValueError("Failed to open spacemouse")

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        import quaternion

        # obs: the folllow robot's current state
        # in rad, 6/7 dof depends on the robot type
        num_dof = 6
        if self._verbose:
            print("act invoked")
        current_qpos = obs["joint_positions"][:num_dof]  # last one dim is the gripper
        current_gripper_angle = obs["joint_positions"][-1]
        self.physics.data.qpos[:num_dof] = current_qpos

        self.physics.step()
        ee_rot = np.array(self.physics.named.data.site_xmat["attachment_site"]).reshape(
            3, 3
        )
        ee_pos = np.array(self.physics.named.data.site_xpos["attachment_site"])

        ee_rot = mj2ur[:3, :3] @ ee_rot
        ee_pos = apply_transfer(mj2ur, ee_pos)
        # ^ mujoco coordinate to UR

        with self.last_state_lock:
            spacemouse_read = self.spacemouse_latest_state
            if self._verbose:
                print(f"spacemouse_read: {spacemouse_read}")

        assert spacemouse_read is not None
        spacemouse_xyz_rot_np = np.array(
            [
                spacemouse_read.x,
                spacemouse_read.y,
                spacemouse_read.z,
                spacemouse_read.roll,
                spacemouse_read.pitch,
                spacemouse_read.yaw,
            ]
        )
        spacemouse_button = (
            spacemouse_read.buttons
        )  # size 2 list binary indicating left/right button pressing
        spacemouse_xyz_rot_np = spacemouse_xyz_rot_np * self.config.invert_control
        if np.max(np.abs(spacemouse_xyz_rot_np)) > 0.9:
            spacemouse_xyz_rot_np[np.abs(spacemouse_xyz_rot_np) < 0.6] = 0
        tx, ty, tz, r, p, y = spacemouse_xyz_rot_np
        # convert roll pick yaw to rotation matrix (rpy)

        trans_transform = np.eye(4)
        # delta xyz from the spacemouse reading
        trans_transform[:3, 3] = apply_transfer(
            spacemouse2ur, np.array([tx, ty, tz]) * self.config.translation_scale
        )

        # break rot_transform into each axis
        rot_transform_x = np.eye(4)
        rot_transform_x[:3, :3] = quaternion.as_rotation_matrix(
            quaternion.from_rotation_vector(
                np.array([-p, 0, 0]) * self.config.angle_scale
            )
        )

        rot_transform_y = np.eye(4)
        rot_transform_y[:3, :3] = quaternion.as_rotation_matrix(
            quaternion.from_rotation_vector(
                np.array([0, r, 0]) * self.config.angle_scale
            )
        )

        rot_transform_z = np.eye(4)
        rot_transform_z[:3, :3] = quaternion.as_rotation_matrix(
            quaternion.from_rotation_vector(
                np.array([0, 0, -y]) * self.config.angle_scale
            )
        )

        # in ur space
        rot_transform = (
            spacemouse2ur
            @ rot_transform_z
            @ rot_transform_y
            @ rot_transform_x
            @ ur2spacemouse
        )

        if self._verbose:
            print(f"rot_transform: {rot_transform}")
            print(f"new spacemounse cmd in ur space = {trans_transform[:3, 3]}")

        # import pdb; pdb.set_trace()
        new_ee_pos = trans_transform[:3, 3] + ee_pos
        if self.config.rotation_mode == "rpy":
            new_ee_rot = ee_rot @ rot_transform[:3, :3]
        elif self.config.rotation_mode == "euler":
            new_ee_rot = rot_transform[:3, :3] @ ee_rot
        else:
            raise NotImplementedError(
                f"Unknown rotation mode: {self.config.rotation_mode}"
            )

        target_quat = quaternion.as_float_array(
            quaternion.from_rotation_matrix(ur2mj[:3, :3] @ new_ee_rot)
        )
        ik_result = qpos_from_site_pose(
            self.physics,
            "attachment_site",
            target_pos=apply_transfer(ur2mj, new_ee_pos),
            target_quat=target_quat,
            tol=1e-14,
            max_steps=400,
        )
        self.physics.reset()
        if ik_result.success:
            new_qpos = ik_result.qpos[:num_dof]
        else:
            print("ik failed, using the original qpos")
            return np.concatenate([current_qpos, [current_gripper_angle]])
        new_gripper_angle = current_gripper_angle
        if self._invert_button:
            if spacemouse_button[1]:
                new_gripper_angle = 1
            if spacemouse_button[0]:
                new_gripper_angle = 0
        else:
            if spacemouse_button[1]:
                new_gripper_angle = 0
            if spacemouse_button[0]:
                new_gripper_angle = 1
        command = np.concatenate([new_qpos, [new_gripper_angle]])
        return command


if __name__ == "__main__":
    import pyspacemouse

    success = pyspacemouse.open("/dev/hidraw4")
    success = pyspacemouse.open("/dev/hidraw5")
