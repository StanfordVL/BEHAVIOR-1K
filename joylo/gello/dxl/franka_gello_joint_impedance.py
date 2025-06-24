from typing import Sequence, Union

import numpy as np

from gello.dxl.joint_impedance import DXLJointImpedanceController


FRANKA_JOINT_LIMIT_HIGH = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
)
FRANKA_JOINT_LIMIT_LOW = np.array(
    [
        -2.8973,
        -1.7628,
        -2.8973,
        -3.0718,
        -2.8973,
        -0.0175,
        -2.8973,
    ]
)


class FrankaGelloJointImpedanceController:
    def __init__(
        self,
        ids: Union[int, Sequence[int]],
        *,
        port: str,
        baudrate: int = 3000000,
        Kp: Union[float, Sequence[float]],
        Kd: Union[float, Sequence[float]],
        reset_joint_positions: np.ndarray,
        joint_offsets: Sequence[float],
        joint_signs: Sequence[int],
    ):
        self._convert_fn = lambda x: x / joint_signs + joint_offsets

        assert reset_joint_positions.shape == FRANKA_JOINT_LIMIT_LOW.shape
        assert np.all(FRANKA_JOINT_LIMIT_LOW <= reset_joint_positions) and np.all(
            reset_joint_positions <= FRANKA_JOINT_LIMIT_HIGH
        )
        reset_motor_positions = self._convert_fn(reset_joint_positions)
        self._controller = DXLJointImpedanceController(
            ids=ids,
            port=port,
            baudrate=baudrate,
            Kp=Kp,
            Kd=Kd,
            reset_joint_positions=reset_motor_positions,
        )

    def set_new_goal(self, goal: np.ndarray):
        assert goal.shape == FRANKA_JOINT_LIMIT_LOW.shape
        assert np.all(FRANKA_JOINT_LIMIT_LOW <= goal) and np.all(
            goal <= FRANKA_JOINT_LIMIT_HIGH
        )
        goal_motor_positions = self._convert_fn(goal)
        self._controller.set_new_goal(goal_motor_positions)

    def start_control(self):
        self._controller.start_control()

    def close(self):
        self._controller.close()

    def __del__(self):
        self.close()
