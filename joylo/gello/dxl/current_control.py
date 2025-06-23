import time
from typing import Sequence

import numpy as np
from dynamixel_sdk import GroupSyncWrite
from dynamixel_sdk.group_sync_read import GroupSyncRead

from gello.dxl.base import DXLBaseDriver
from gello.dxl.constants import GoalCurrent, CurrentLimit, Comm, DXL_LOBYTE, DXL_HIBYTE


class DXLCurrentControlDriver(DXLBaseDriver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with self._lock:
            current_limit_group_sync_read = GroupSyncRead(
                self._port_handler,
                self._packet_handler,
                CurrentLimit.ADDR.value,
                CurrentLimit.LEN.value,
            )
            # Add parameters for each Dynamixel servo to the group sync read
            for dxl_id in self._ids:
                if not current_limit_group_sync_read.addParam(dxl_id):
                    raise RuntimeError(
                        f"Failed to add parameter for Dynamixel with ID {dxl_id}"
                    )
            # read current limit
            while True:
                dxl_comm_result = current_limit_group_sync_read.txRxPacket()
                if dxl_comm_result == Comm.SUCCESS.value:
                    break
                time.sleep(0.1)
            self._current_limits = []
            for i, dxl_id in enumerate(self._ids):
                if current_limit_group_sync_read.isAvailable(
                    dxl_id, CurrentLimit.ADDR.value, CurrentLimit.LEN.value
                ):
                    current_limit = current_limit_group_sync_read.getData(
                        dxl_id, CurrentLimit.ADDR.value, CurrentLimit.LEN.value
                    )
                    current_limit = CurrentLimit.to_numpy(current_limit)
                    self._current_limits.append(current_limit)
            self._current_limits = np.array(self._current_limits, dtype=np.float32)
            del current_limit_group_sync_read

    def _create_group_sync_write(self, operating_mode: str) -> GroupSyncWrite:
        assert operating_mode == "current", """Use `operating_mode = "current"`"""
        group_sync_write = GroupSyncWrite(
            self._port_handler,
            self._packet_handler,
            GoalCurrent.ADDR.value,
            GoalCurrent.LEN.value,
        )
        return group_sync_write

    def _set_joints(self, joint_angles: Sequence[float]):
        joint_currents = np.array(joint_angles, dtype=np.float32)
        joint_currents = np.clip(
            joint_currents, -self._current_limits, self._current_limits
        )

        for dxl_id, current in zip(self._ids, joint_currents):
            current_value = int(current)
            param = [DXL_LOBYTE(current_value), DXL_HIBYTE(current_value)]
            dxl_addparam_result = self._group_sync_write.addParam(dxl_id, param)
            if not dxl_addparam_result:
                raise RuntimeError(
                    f"Failed to set goal current for Dynamixel with ID {dxl_id}"
                )
        # Syncwrite goal current
        dxl_comm_result = self._group_sync_write.txPacket()
        if dxl_comm_result != Comm.SUCCESS.value:
            raise RuntimeError(
                f"Failed to syncwrite goal current. Returned msg: {dxl_comm_result}"
            )

        # Clear syncwrite parameter storage
        self._group_sync_write.clearParam()
