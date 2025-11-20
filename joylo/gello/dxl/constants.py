from enum import Enum

import numpy as np


class OperatingMode(Enum):
    ADDR = 11
    LEN = 1
    CURRENT_CONTROL_MODE = 0
    VELOCITY_CONTROL_MODE = 1
    POSITION_CONTROL_MODE = 3
    EXTENDED_POSITION_CONTROL_MODE = 4
    CURRENT_BASED_POSITION_CONTROL_MODE = 5
    PWM_CONTROL_MODE = 16


operating_modes = {
    "current": OperatingMode.CURRENT_CONTROL_MODE,
    "velocity": OperatingMode.VELOCITY_CONTROL_MODE,
    "position": OperatingMode.POSITION_CONTROL_MODE,
    "extended_position": OperatingMode.EXTENDED_POSITION_CONTROL_MODE,
    "current_based_position": OperatingMode.CURRENT_BASED_POSITION_CONTROL_MODE,
    "pwm": OperatingMode.PWM_CONTROL_MODE,
}


class CurrentControlModeWrite(Enum):
    ADDR = 102
    LEN = 2


class PositionControlModeWrite(Enum):
    ADDR = 116
    LEN = 4


class PresentPosition(Enum):
    ADDR = 132
    LEN = 4


class GoalPosition(Enum):
    ADDR = 116
    LEN = 4


class GoalVelocity(Enum):
    ADDR = 104
    LEN = 4


class PresentVelocity(Enum):
    ADDR = 128
    LEN = 4


class GoalCurrent(Enum):
    ADDR = 102
    LEN = 2


class CurrentLimit(Enum):
    ADDR = 38
    LEN = 2

    @staticmethod
    def to_numpy(x):
        return np.int32(np.uint16(x))


class Torque(Enum):
    ADDR = 64
    ENABLE = 1
    DISABLE = 0


class Comm(Enum):
    SUCCESS = 0


class DriveMode(Enum):
    ADDR = 10
    LEN = 1
    NORMAL = 0
    REVERSE = 1


class HomingOffset(Enum):
    ADDR = 20
    LEN = 4
    

# Macro for Control Table Value
def DXL_MAKEWORD(a, b):
    return (a & 0xFF) | ((b & 0xFF) << 8)


def DXL_MAKEDWORD(a, b):
    return (a & 0xFFFF) | (b & 0xFFFF) << 16


def DXL_LOWORD(l):
    return l & 0xFFFF


def DXL_HIWORD(l):
    return (l >> 16) & 0xFFFF


def DXL_LOBYTE(w):
    return w & 0xFF


def DXL_HIBYTE(w):
    return (w >> 8) & 0xFF
