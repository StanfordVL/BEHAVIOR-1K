import numpy as np
from collections import namedtuple
from omni.isaac.core.utils.types import DataFrame, DOFInfo, XFormPrimState, \
    DynamicState, ArticulationAction
from omni.isaac.core.utils.types import JointsState as JS

# Raw Body Contact Information
# See https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.contact_sensor/docs/index.html?highlight=contact%20sensor#omni.isaac.contact_sensor._contact_sensor.CsRawData for more info.
CsRawData = namedtuple("RawBodyData", ["time", "dt", "body0", "body1", "position", "normal", "impulse"])

# Valid geom types
GEOM_TYPES = {"Sphere", "Cube", "Capsule", "Cone", "Cylinder", "Mesh"}


class JointsState(JS):
    """
    We extend the native functionality to allow flattening of values
    """
    def flatten(self):
        """
        Flattens the internal state and returns as a 1D numpy array (pos, vel, effort)

        Returns:
            n-array: 1D-flattened array of (pos, vel, effort) values. Note that all values must be filled in (i.e:
                None will raise an error)
        """
        assert self.positions is not None and self.velocities is not None and self.efforts is not None, \
            "Can only flatten JointsState if all values are not None!"

        return np.concatenate([self.positions, self.velocities, self.efforts])
