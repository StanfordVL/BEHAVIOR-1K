import time
from enum import IntEnum
from threading import Event, Lock, Thread
from typing import Protocol, Sequence, Union, List, Optional

import numpy as np
from dynamixel_sdk.group_sync_read import GroupSyncRead
from dynamixel_sdk.group_sync_write import GroupSyncWrite
from dynamixel_sdk.packet_handler import PacketHandler
from dynamixel_sdk.port_handler import PortHandler
from dynamixel_sdk.robotis_def import (
    COMM_SUCCESS,
    DXL_HIBYTE,
    DXL_HIWORD,
    DXL_LOBYTE,
    DXL_LOWORD,
)

# Constants
ADDR_TORQUE_ENABLE = 64
ADDR_MAX_CURRENT = 38
LEN_MAX_CURRENT = 4
ADDR_GOAL_CURRENT = 102
LEN_GOAL_CURRENT = 2
ADDR_PRESENT_CURRENT = 126
LEN_PRESENT_CURRENT = 2
ADDR_PRESENT_VELOCITY = 128
LEN_PRESENT_VELOCITY = 4
ADDR_GOAL_POSITION = 116
LEN_GOAL_POSITION = 4
ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION = 4
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
ADDR_OPERATING_MODE = 11
LEN_OPERATING_MODE = 1
LEN_GAIN = 2
ADDR_MAX_VEL = 44


# Control mode
class OperatingMode(IntEnum):
    NONE = -1                   # This corresponds to disabling torque
    CURRENT = 0
    VELOCITY = 1
    POSITION = 3
    EXTENDED_POSITION = 4
    CURRENT_BASED_POSITION = 5
    PWM = 16


# Control mode
class GainType(IntEnum):
    P = 84
    I = 82
    D = 80


class DynamixelDriverProtocol(Protocol):
    def set_joints(self, joint_angles: Sequence[float], idxs: Optional[Sequence[int]] = None):
        """Set the joint angles for the Dynamixel servos.

        Args:
            joint_angles (Sequence[float]): A list of joint angles.
            idxs (Optional[Sequence[int]]): If specified, idxs to write to
        """
        ...

    def torque_enabled(self) -> Sequence[bool]:
        """Check if torque is enabled for the Dynamixel servos.

        Returns:
            list of bool: Per-motor True if torque is enabled, False if it is disabled.
        """
        ...

    def set_torque_mode(self, enable: Union[bool, Sequence[bool]], idxs: Optional[Sequence[int]] = None):
        """Set the torque mode for the Dynamixel servos.

        Args:
            enable (bool or list of bool): True to enable torque, False to disable. Can be specified per-motor
            idxs (Optional[Sequence[int]]): If specified, idxs to write to
        """
        ...

    def get_joints(self) -> np.ndarray:
        """Get the current joint angles in radians.

        Returns:
            np.ndarray: An array of joint angles.
        """
        ...

    def close(self):
        """Close the driver."""


class FakeDynamixelDriver(DynamixelDriverProtocol):
    def __init__(self, ids: Sequence[int]):
        self._ids = ids
        self._n_joints = len(self._ids)
        self._joint_angles = np.zeros(len(ids), dtype=int)
        self._torque_enabled = np.zeros(self._n_joints, dtype=bool)

    def set_joints(self, joint_angles: Sequence[float], idxs: Optional[Sequence[int]] = None):
        idxs = np.arange(self._n_joints) if idxs is None else idxs
        if len(joint_angles) != len(idxs):
            raise ValueError(
                "The length of joint_angles must match the number of servos"
            )
        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled to set joint angles")
        self._joint_angles = np.array(joint_angles)

    def torque_enabled(self) -> Sequence[bool]:
        return self._torque_enabled

    def set_torque_mode(self, enable: Union[bool, Sequence[bool]], idxs: Optional[Sequence[int]] = None):
        self._torque_enabled = enable

    def get_joints(self) -> np.ndarray:
        return self._joint_angles.copy()

    def close(self):
        pass


class DynamixelDriver(DynamixelDriverProtocol):
    def __init__(
        self, ids: Sequence[int], port: str = "/dev/ttyUSB0", baudrate: int = 2000000
    ):
        """Initialize the DynamixelDriver class.

        Args:
            ids (Sequence[int]): A list of IDs for the Dynamixel servos.
            port (str): The USB port to connect to the arm.
            baudrate (int): The baudrate for communication.
        """

        self._ids = np.array(ids)
        self._n_joints = len(self._ids)
        self._joint_angles = None
        self._lock = Lock()

        # Initialize the port handler, packet handler, and group sync read/write
        self._portHandler = PortHandler(port)
        self._packetHandler = PacketHandler(2.0)
        self._groupSyncRead = GroupSyncRead(
            self._portHandler,
            self._packetHandler,
            ADDR_PRESENT_POSITION,
            LEN_PRESENT_POSITION,
        )
        self._groupSyncWrite = GroupSyncWrite(
            self._portHandler,
            self._packetHandler,
            ADDR_GOAL_POSITION,
            LEN_GOAL_POSITION,
        )

        # Read current values
        self._groupSyncReadCurrent = GroupSyncRead(
            self._portHandler,
            self._packetHandler,
            ADDR_PRESENT_CURRENT,
            LEN_PRESENT_CURRENT,
        )
        self._groupSyncWriteCurrent = GroupSyncWrite(
            self._portHandler,
            self._packetHandler,
            ADDR_GOAL_CURRENT,
            LEN_GOAL_CURRENT,
        )

        # Read velocity values
        self._groupSyncReadVelocity = GroupSyncRead(
            self._portHandler,
            self._packetHandler,
            ADDR_PRESENT_VELOCITY,
            LEN_PRESENT_VELOCITY,
        )

        # Open the port and set the baudrate
        if not self._portHandler.openPort():
            raise RuntimeError("Failed to open the port")

        if not self._portHandler.setBaudRate(baudrate):
            raise RuntimeError(f"Failed to change the baudrate, {baudrate}")

        # Add parameters for each Dynamixel servo to the group sync read
        for dxl_id in self._ids:
            if not self._groupSyncRead.addParam(dxl_id):
                raise RuntimeError(
                    f"Failed to add parameter for Dynamixel with ID {dxl_id}"
                )
            if not self._groupSyncReadCurrent.addParam(dxl_id):
                raise RuntimeError(
                    f"Failed to add parameter for Dynamixel with ID {dxl_id}"
                )
            if not self._groupSyncReadVelocity.addParam(dxl_id):
                raise RuntimeError(
                    f"Failed to add parameter for Dynamixel with ID {dxl_id}"
                )

        # Record max current values
        max_currents = np.zeros(self._n_joints, dtype=int)
        for i, dxl_id in enumerate(self._ids):
            # Read from packet handler and record max currents
            max_curr, result, err = self._packetHandler.read2ByteTxRx(self._portHandler, dxl_id, ADDR_MAX_CURRENT)
            assert result == COMM_SUCCESS, f"comm failed: {result}"
            max_currents[i] = np.int32(np.uint32(max_curr))
        self._max_currents = max_currents

        # Disable torque for each Dynamixel servo
        self._operating_mode = np.array([OperatingMode.NONE] * self._n_joints, dtype=int)
        self._torque_enabled = np.zeros(self._n_joints, dtype=bool)
        self.set_torque_mode(self._torque_enabled)

        self._stop_thread = Event()
        self._start_reading_thread()

    def set_currents(self, current: Sequence[float], idxs: Optional[Sequence[int]] = None) -> None:        
        idxs = np.arange(self._n_joints) if idxs is None else idxs

        # Do nothing if an empty array of idxs is specified
        if len(idxs) == 0:
            return

        with self._lock:
            if len(current) != len(idxs):
                raise ValueError(
                    "The length of current must match the number of servos ids"
                )

            # Convert A -> mA, then clip to max values
            current = (np.array(current) * 1000).astype(np.int16).clip(-self._max_currents[idxs], self._max_currents[idxs])
            for idx, dxl_id, curr in zip(idxs, self._ids[idxs], current):
                # Raise error if we're not the correct operating mode or torque is not enabled
                if self._operating_mode[idx] != OperatingMode.CURRENT:
                    raise RuntimeError("OperatingMode must be set to CURRENT in order to set current")

                if not self._torque_enabled[idx]:
                    raise RuntimeError("Torque must be enabled to set current")

                # Allocate goal current value into byte array
                param_goal_curr = [
                    DXL_LOBYTE(DXL_LOWORD(curr)),
                    DXL_HIBYTE(DXL_LOWORD(curr)),
                ]

                # Add goal current value to the Syncwrite parameter storage
                dxl_addparam_result = self._groupSyncWriteCurrent.addParam(
                    dxl_id, param_goal_curr
                )
                if not dxl_addparam_result:
                    raise RuntimeError(
                        f"Failed to set current for Dynamixel with ID {dxl_id}"
                    )

            # Syncwrite goal current
            dxl_comm_result = self._groupSyncWriteCurrent.txPacket()
            if dxl_comm_result != COMM_SUCCESS:
                raise RuntimeError(f"Failed to syncwrite goal current, error: {dxl_comm_result}")

            # Clear syncwrite parameter storage
            self._groupSyncWriteCurrent.clearParam()

    def set_joints(self, joint_angles: Sequence[float], idxs: Optional[Sequence[int]] = None):
        idxs = np.arange(self._n_joints) if idxs is None else idxs

        with self._lock:
            if len(joint_angles) != len(idxs):
                raise ValueError(
                    "The length of joint_angles must match the number of servos"
                )

            for idx, dxl_id, angle in zip(idxs, self._ids[idxs], joint_angles):
                # Raise error if we're not the correct operating mode or torque is not enabled
                if self._operating_mode[idx] not in {OperatingMode.POSITION, OperatingMode.EXTENDED_POSITION, OperatingMode.CURRENT_BASED_POSITION}:
                    raise RuntimeError("OperatingMode must be set to either POSITION or CURRENT_BASED_POSITION in order to set joint positions")

                if not self._torque_enabled[idx]:
                    raise RuntimeError("Torque must be enabled to set joint positions")

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
                dxl_addparam_result = self._groupSyncWrite.addParam(
                    dxl_id, param_goal_position
                )
                if not dxl_addparam_result:
                    raise RuntimeError(
                        f"Failed to set joint angle for Dynamixel with ID {dxl_id}"
                    )

            # Syncwrite goal position
            dxl_comm_result = self._groupSyncWrite.txPacket()
            if dxl_comm_result != COMM_SUCCESS:
                raise RuntimeError("Failed to syncwrite goal position")

            # Clear syncwrite parameter storage
            self._groupSyncWrite.clearParam()

    def torque_enabled(self) -> Sequence[bool]:
        return self._torque_enabled

    def set_torque_mode(self, enable: Union[bool, Sequence[bool]], idxs: Optional[Sequence[int]] = None):
        idxs = np.arange(self._n_joints) if idxs is None else idxs

        if isinstance(enable, bool):
            enable = np.array([enable] * len(idxs), dtype=bool)

        # If torque is enabled, and we don't have any current operating mode specified, default to current
        torque_values = np.where(enable, TORQUE_ENABLE, TORQUE_DISABLE)

        with self._lock:
            if len(torque_values) != len(idxs):
                raise ValueError(
                    "The length of torque enable must match the number of idxs"
                )

            for dxl_id, torque_value in zip(self._ids[idxs], torque_values):
                dxl_comm_result, dxl_error = self._packetHandler.write1ByteTxRx(
                    self._portHandler, dxl_id, ADDR_TORQUE_ENABLE, torque_value
                )
                if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                    print(dxl_comm_result)
                    print(dxl_error)
                    raise RuntimeError(
                        f"Failed to set torque mode for Dynamixel with ID {dxl_id}"
                    )

        self._torque_enabled[idxs] = enable

    def set_operating_mode(self, mode: Union[OperatingMode, List[OperatingMode]], idxs: Optional[Sequence[int]] = None):
        idxs = np.arange(self._n_joints) if idxs is None else idxs
        set_idxs = set(idxs)

        if isinstance(mode, OperatingMode):
            mode = np.array([mode] * len(idxs), dtype=int)

        # Make sure all operating modes are valid
        modes_set = set(OperatingMode)
        assert all(mo in modes_set for mo in mode), f"Got invalid operating mode in modes: {mode}"

        torque_is_on_idxs = set(np.where(self._torque_enabled)[0]) & set_idxs
        if len(torque_is_on_idxs) > 0:
            self.set_torque_mode(False, idxs=list(torque_is_on_idxs))

        do_not_enable_idxs = set()
        with self._lock:
            if len(mode) != len(idxs):
                raise ValueError(
                    "The length of operating mode must match the number of idxs"
                )

            for idx, dxl_id, mo in zip(idxs, self._ids[idxs], mode):
                # If we're setting to OperatingMode.NONE, set to CURRENT mode by default and also delete the
                # corresponding index if this motor formerly had torque enabled so it is not re-enabled
                if mo == OperatingMode.NONE:
                    mo = OperatingMode.CURRENT
                    do_not_enable_idxs.add(idx)
                dxl_comm_result, dxl_error = self._packetHandler.write1ByteTxRx(
                    self._portHandler, dxl_id, ADDR_OPERATING_MODE, mo,
                )
                if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                    print(dxl_comm_result)
                    print(dxl_error)
                    raise RuntimeError(
                        f"Failed to set operating mode for Dynamixel with ID {dxl_id}"
                    )

        self._operating_mode[idxs] = mode

        # Re-enable torque if it was already on at the beginning of this call
        # NOTE: This skips any where the operating mode is set to NONE
        enable_idxs = set_idxs - do_not_enable_idxs
        if len(enable_idxs) > 0:
            self.set_torque_mode(True, idxs=list(enable_idxs))

    def set_gain(self, gain_type: GainType, val: Union[int, Sequence[int]], idxs: Optional[Sequence[int]] = None):
        idxs = np.arange(self._n_joints) if idxs is None else idxs

        if not isinstance(val, Sequence):
            val = np.array([val] * len(idxs), dtype=int)

        # Gain type is the raw address to write to
        assert gain_type in GainType, f"Got invalid gain type: {gain_type}"

        with self._lock:
            if len(val) != len(idxs):
                raise ValueError(
                    "The length of gain values must match the number of servos"
                )

            for dxl_id, v in zip(self._ids[idxs], val):
                dxl_comm_result, dxl_error = self._packetHandler.write2ByteTxRx(
                    self._portHandler, dxl_id, gain_type, v,
                )
                if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                    print(dxl_comm_result)
                    print(dxl_error)
                    raise RuntimeError(
                        f"Failed to set gain at addr [{gain_type}] with value [{val}] for Dynamixel with ID {dxl_id}"
                    )

    def set_max_velocity(self, val: Union[int, Sequence[int]], idxs: Optional[Sequence[int]] = None):
        idxs = np.arange(self._n_joints) if idxs is None else idxs

        if not isinstance(val, Sequence):
            val = [val] * len(idxs)
        val = np.array(val)
        # Convert rad / sec to RPM to unit scale (0.22888 rev / min per increment)
        val = (val * 30 / (0.22888 * np.pi)).astype(int)

        with self._lock:
            if len(val) != len(idxs):
                raise ValueError(
                    "The length of gain values must match the number of servos"
                )

            for dxl_id, v in zip(self._ids[idxs], val):
                dxl_comm_result, dxl_error = self._packetHandler.write4ByteTxRx(
                    self._portHandler, dxl_id, ADDR_MAX_VEL, v,
                )
                if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                    print(dxl_comm_result)
                    print(dxl_error)
                    raise RuntimeError(
                        f"Failed to set max velocity at addr [{ADDR_MAX_VEL}] with value [{val}] for Dynamixel with ID {dxl_id}"
                    )

    def _start_reading_thread(self):
        self._reading_thread = Thread(target=self._read_joint_angles)
        self._reading_thread.daemon = True
        self._reading_thread.start()

    def _read_joint_angles(self):
        # Continuously read joint angles and update the joint_angles array
        while not self._stop_thread.is_set():
            time.sleep(0.001)
            with self._lock:
                _joint_angles = np.zeros(len(self._ids), dtype=int)
                dxl_comm_result = self._groupSyncRead.txRxPacket()
                if dxl_comm_result != COMM_SUCCESS:
                    print(f"warning, comm failed: {dxl_comm_result}")
                    continue
                for i, dxl_id in enumerate(self._ids):
                    if self._groupSyncRead.isAvailable(
                        dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
                    ):
                        angle = self._groupSyncRead.getData(
                            dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
                        )
                        angle = np.int32(np.uint32(angle))
                        _joint_angles[i] = angle
                    else:
                        raise RuntimeError(
                            f"Failed to get joint angles for Dynamixel with ID {dxl_id}"
                        )
                self._joint_angles = _joint_angles

            #     # Read currents
            #     _joint_currents = np.zeros(len(self._ids), dtype=np.int16)
            #     dxl_comm_result = self._groupSyncReadCurrent.txRxPacket()
            #     if dxl_comm_result != COMM_SUCCESS:
            #         print(f"warning, comm failed: {dxl_comm_result}")
            #         continue
            #     for i, dxl_id in enumerate(self._ids):
            #         if self._groupSyncReadCurrent.isAvailable(
            #             dxl_id, ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT
            #         ):
            #             curr = self._groupSyncReadCurrent.getData(
            #                 dxl_id, ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT
            #             )
            #             curr = np.int16(np.uint16(curr))
            #             _joint_currents[i] = curr
            #         else:
            #             raise RuntimeError(
            #                 f"Failed to get joint angles for Dynamixel with ID {dxl_id}"
            #             )
            #     print(f"Currents: {_joint_currents}")
            # # self._groupSyncRead.clearParam() # TODO what does this do? should i add it

    def get_joints(self) -> np.ndarray:
        # Return a copy of the joint_angles array to avoid race conditions
        while self._joint_angles is None:
            time.sleep(0.1)
        # with self._lock:
        _j = self._joint_angles.copy()
        return _j / 2048.0 * np.pi

    def get_velocities(self):
        with self._lock:
            _joint_vels = np.zeros(len(self._ids), dtype=int)
            dxl_comm_result = self._groupSyncReadVelocity.txRxPacket()
            if dxl_comm_result != COMM_SUCCESS:
                print(f"warning, comm failed: {dxl_comm_result}")
            for i, dxl_id in enumerate(self._ids):
                if self._groupSyncReadVelocity.isAvailable(
                    dxl_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY
                ):
                    vel = self._groupSyncReadVelocity.getData(
                        dxl_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY
                    )
                    vel = np.int32(np.uint32(vel))
                    _joint_vels[i] = vel
                else:
                    raise RuntimeError(
                        f"Failed to get joint velocities for Dynamixel with ID {dxl_id}"
                    )
            # Convert to RPM --> rad / sec
            return _joint_vels * 0.229 * 2 * np.pi / 60

    def close(self):
        self._stop_thread.set()
        self._reading_thread.join()
        self._portHandler.closePort()


def main():
    # Set the port, baudrate, and servo IDs
    ids = [1]

    # Create a DynamixelDriver instance
    try:
        driver = DynamixelDriver(ids)
    except FileNotFoundError:
        driver = DynamixelDriver(ids, port="/dev/cu.usbserial-FT7WBMUB")

    # Test setting torque mode
    driver.set_torque_mode(True)
    driver.set_torque_mode(False)

    # Test reading the joint angles
    try:
        while True:
            joint_angles = driver.get_joints()
            print(f"Joint angles for IDs {ids}: {joint_angles}")
            # print(f"Joint angles for IDs {ids[1]}: {joint_angles[1]}")
    except KeyboardInterrupt:
        driver.close()


if __name__ == "__main__":
    main()  # Test the driver
