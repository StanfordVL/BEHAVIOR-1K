from typing import Sequence, Union
from threading import Event, Thread

import numpy as np
from gello.dxl.base import run_at_frequency
from gello.dxl.current_control import DXLCurrentControlDriver


class DXLJointImpedanceController:
    def __init__(
        self,
        ids: Union[int, Sequence[int]],
        *,
        port: str,
        baudrate: int = 3000000,
        Kp: Union[float, Sequence[float]],
        Kd: Union[float, Sequence[float]],
        reset_joint_positions: np.ndarray,
    ):
        self._driver = DXLCurrentControlDriver(
            ids=ids,
            port=port,
            baudrate=baudrate,
            operating_mode="current",
            multithread_read_joints=False,
        )
        if isinstance(Kp, (int, float)):
            self._Kp = np.array([Kp] * len(self._driver.motor_ids), dtype=np.float32)
        else:
            self._Kp = np.array(Kp, dtype=np.float32)
        if isinstance(Kd, (int, float)):
            self._Kd = np.array([Kd] * len(self._driver.motor_ids), dtype=np.float32)
        else:
            self._Kd = np.array(Kd, dtype=np.float32)

        assert len(self._Kp) == len(self._Kd) == len(self._driver.motor_ids)
        assert len(reset_joint_positions) == len(self._driver.motor_ids)

        self._goal = reset_joint_positions

        self._stop_thread = None
        self._control_thread = None

    @run_at_frequency(hz=500)
    def _control(self):
        while not self._stop_thread.is_set():
            curr_positions, curr_velocities = self._driver.get_joints()
            delta_positions = self._goal.copy() - curr_positions
            self._driver.set_joints(
                self._Kp * delta_positions - self._Kd * curr_velocities
            )

    def set_new_goal(self, goal: np.ndarray):
        assert len(goal) == len(self._driver.motor_ids)
        self._goal = goal

    def start_control(self):
        self._driver.set_torque_mode(True)
        self._stop_thread = Event()
        self._control_thread = Thread(target=self._control)
        self._control_thread.daemon = True
        self._control_thread.start()

    def close(self):
        self._stop_thread.set()
        self._control_thread.join()
        self._driver.set_torque_mode(False)
        self._driver.close()

    def __del__(self):
        self.close()



if __name__ == "__main__":
    controller = DXLJointImpedanceController(
        ids=(7),
        port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT8J0W3N-if00-port0",
        baudrate=3000000,
        Kp=10,
        Kd=1,
        reset_joint_positions=np.array([0]),
    )
    controller.start_control()
    controller.set_new_goal(np.array([0.7]))
    while True:
        pass

    
