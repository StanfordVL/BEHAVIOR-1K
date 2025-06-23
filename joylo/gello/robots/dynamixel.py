from typing import Dict, Optional, Sequence, Tuple, List, Union

import numpy as np

from gello.robots.robot import Robot
from gello.dynamixel.driver import OperatingMode


class DynamixelRobot(Robot):
    """A class representing a UR robot."""

    def __init__(
        self,
        joint_ids: Sequence[int],
        joint_offsets: Optional[Sequence[float]] = None,
        joint_signs: Optional[Sequence[int]] = None,
        real: bool = False,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 2000000,
        gripper_config: Optional[Tuple[int, float, float]] = None,
        start_joints: Optional[np.ndarray] = None,
    ):
        from gello.dynamixel.driver import (
            DynamixelDriver,
            DynamixelDriverProtocol,
            FakeDynamixelDriver,
        )

        print(f"attempting to connect to port: {port}")
        self.gripper_open_close: Optional[Tuple[float, float]]
        if gripper_config is not None:
            assert joint_offsets is not None
            assert joint_signs is not None

            # joint_ids.append(gripper_config[0])
            # joint_offsets.append(0.0)
            # joint_signs.append(1)
            joint_ids = tuple(joint_ids) + (gripper_config[0],)
            joint_offsets = tuple(joint_offsets) + (0.0,)
            joint_signs = tuple(joint_signs) + (1,)
            self.gripper_open_close = (
                gripper_config[1] * np.pi / 180,
                gripper_config[2] * np.pi / 180,
            )
        else:
            self.gripper_open_close = None

        self._joint_ids = joint_ids
        self._n_joints = len(self._joint_ids)
        self._driver: DynamixelDriverProtocol

        if joint_offsets is None:
            self._joint_offsets = np.zeros(len(joint_ids))
        else:
            self._joint_offsets = np.array(joint_offsets)

        if joint_signs is None:
            self._joint_signs = np.ones(len(joint_ids))
        else:
            self._joint_signs = np.array(joint_signs)

        assert len(self._joint_ids) == len(self._joint_offsets), (
            f"joint_ids: {len(self._joint_ids)}, "
            f"joint_offsets: {len(self._joint_offsets)}"
        )
        assert len(self._joint_ids) == len(self._joint_signs), (
            f"joint_ids: {len(self._joint_ids)}, "
            f"joint_signs: {len(self._joint_signs)}"
        )
        assert np.all(
            np.abs(self._joint_signs) == 1
        ), f"joint_signs: {self._joint_signs}"

        if real:
            self._driver = DynamixelDriver(joint_ids, port=port, baudrate=baudrate)
        else:
            self._driver = FakeDynamixelDriver(joint_ids)
        self._last_pos = None
        self._last_vel = None
        self._alpha = 0.99

        if start_joints is not None:
            # loop through all joints and add +- 2pi to the joint offsets to get the closest to start joints
            new_joint_offsets = []
            current_joints = self.get_joint_state()
            assert current_joints.shape == start_joints.shape
            if gripper_config is not None:
                current_joints = current_joints[:-1]
                start_joints = start_joints[:-1]
            for idx, (c_joint, s_joint, joint_offset) in enumerate(
                zip(current_joints, start_joints, self._joint_offsets)
            ):
                new_joint_offsets.append(
                    np.pi
                    * 2
                    * np.round((-s_joint + c_joint) / (2 * np.pi))
                    * self._joint_signs[idx]
                    + joint_offset
                )
            if gripper_config is not None:
                new_joint_offsets.append(self._joint_offsets[-1])
            self._joint_offsets = np.array(new_joint_offsets)

    def num_dofs(self) -> int:
        return len(self._joint_ids)

    def get_joint_state(self) -> np.ndarray:
        pos = (self._driver.get_joints() - self._joint_offsets) * self._joint_signs
        assert len(pos) == self.num_dofs()

        if self.gripper_open_close is not None:
            # map pos to [0, 1]
            g_pos = (pos[-1] - self.gripper_open_close[0]) / (
                self.gripper_open_close[1] - self.gripper_open_close[0]
            )
            g_pos = min(max(0, g_pos), 1)
            pos[-1] = g_pos

        if self._last_pos is None:
            self._last_pos = pos
        else:
            # exponential smoothing
            pos = self._last_pos * (1 - self._alpha) + pos * self._alpha
            self._last_pos = pos

        return pos

    def get_joint_velocities(self):
        vel = self._driver.get_velocities() * self._joint_signs

        if self._last_vel is None:
            self._last_vel = vel
        else:
            # exponential smoothing
            vel = self._last_vel * (1 - self._alpha) + vel * self._alpha
            self._last_vel = vel

        return vel

    def command_current(self, current: np.ndarray, idxs: Optional[Sequence[int]] = None) -> None:
        idxs = np.arange(self._n_joints) if idxs is None else idxs
        self._driver.set_currents(current * self._joint_signs[idxs], idxs=idxs)

    def command_joint_state(self, joint_state: np.ndarray, idxs: Optional[Sequence[int]] = None) -> None:
        idxs = np.arange(self._n_joints) if idxs is None else idxs
        self._driver.set_joints((joint_state * self._joint_signs[idxs] + self._joint_offsets[idxs]).tolist(), idxs=idxs)

    # def set_torque_mode(self, mode: Union[bool, Sequence[bool]], idxs: Optional[Sequence[int]] = None):
    #     self._driver.set_torque_mode(mode, idxs=idxs)

    def set_operating_mode(self, mode: Union[OperatingMode, Sequence[OperatingMode]], idxs: Optional[Sequence[int]] = None):
        self._driver.set_operating_mode(mode, idxs=idxs)

    # @property
    # def torque_mode(self) -> Sequence[bool]:
    #     return self._driver._torque_mode

    @property
    def operating_mode(self) -> Sequence[OperatingMode]:
        return self._driver._operating_mode

    def get_observations(self) -> Dict[str, np.ndarray]:
        return {"joint_state": self.get_joint_state()}
