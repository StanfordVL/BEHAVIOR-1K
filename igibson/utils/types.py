from collections import namedtuple
from omni.isaac.core.utils.types import DataFrame, DOFInfo, XFormPrimState, \
    DynamicState, JointsState, ArticulationAction

# Raw Body Contact Information
# See https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.contact_sensor/docs/index.html?highlight=contact%20sensor#omni.isaac.contact_sensor._contact_sensor.CsRawData for more info.
CsRawData = namedtuple("RawBodyData", ["time", "dt", "body0", "body1", "position", "normal", "impulse"])

