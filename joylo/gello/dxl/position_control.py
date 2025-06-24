from typing import Sequence

import numpy as np
from dynamixel_sdk import GroupSyncWrite

from gello.dxl.base import DXLBaseDriver
from gello.dxl.constants import (
    GoalPosition,
    DXL_LOBYTE,
    DXL_LOWORD,
    DXL_HIBYTE,
    DXL_HIWORD,
    Comm,
)


class DXLPositionControlDriver(DXLBaseDriver):
    def _create_group_sync_write(self, operating_mode: str) -> GroupSyncWrite:
        assert operating_mode == "position", """Use `operating_mode = "position"`"""
        group_sync_write = GroupSyncWrite(
            self._port_handler,
            self._packet_handler,
            GoalPosition.ADDR.value,
            GoalPosition.LEN.value,
        )
        return group_sync_write

    def _set_joints(self, joint_angles: Sequence[float]):
        for dxl_id, angle in zip(self._ids, joint_angles):
            # Convert the angle to the appropriate value for the servo
            position_value = int(angle * 2048 / np.pi)

            # Allocate goal position value into byte array
            param_goal_position = [
                DXL_LOBYTE(DXL_LOWORD(position_value)),
                DXL_HIBYTE(DXL_LOWORD(position_value)),
                DXL_LOBYTE(DXL_HIWORD(position_value)),
                DXL_HIBYTE(DXL_HIWORD(position_value)),
            ]

            # Add goal position value to the Syncwrite parameter storage
            dxl_addparam_result = self._group_sync_write.addParam(
                dxl_id, param_goal_position
            )
            if not dxl_addparam_result:
                raise RuntimeError(
                    f"Failed to set joint angle for Dynamixel with ID {dxl_id}"
                )

        # Syncwrite goal position
        dxl_comm_result = self._group_sync_write.txPacket()
        if dxl_comm_result != Comm.SUCCESS.value:
            raise RuntimeError(self._packet_handler.getTxRxResult(dxl_comm_result))

        # Clear syncwrite parameter storage
        self._group_sync_write.clearParam()
