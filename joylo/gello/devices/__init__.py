from gello.devices.joylo import JoyLo
from gello.devices.r1prot import R1ProT

DEVICE_LIB = {
    "JoyLo": JoyLo,
    "R1ProT": R1ProT,
}


__all__ = [
    "DEVICE_LIB"
]