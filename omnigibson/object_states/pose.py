import numpy as np

from omnigibson.macros import create_module_macros
from omnigibson.object_states.object_state_base import AbsoluteObjectState

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.POSITIONAL_VALIDATION_EPSILON = 1e-10


class Pose(AbsoluteObjectState):
    def __init__(self, obj):
        # Initialize has moved variable
        self._has_moved = False

        # Run super first
        super().__init__(obj=obj)

    def _get_value(self):
        pos = self.obj.get_position()
        orn = self.obj.get_orientation()
        return np.array(pos), np.array(orn)

    def _set_value(self, new_value):
        raise NotImplementedError("Pose state currently does not support setting.")

    @property
    def has_moved(self):
        """
        Returns:
            bool: Whether this object has moved its position within the previous timestep. This is used to optimize
                performance, reducing the number of active computation calls
        """
        return self._has_moved

    # Nothing needs to be done to save/load Pose since it will happen due to pose caching.
    def _cache_info(self, get_value_args):
        # Run super first
        info = super()._cache_info(get_value_args=get_value_args)

        # Store this object's position
        info["pos"] = self.obj.states[Pose].get_value()[0]
        return info

    def _should_clear_cache(self, get_value_args, cache_info):
        # Only clear cache if the squared distance between cached position and current position has
        # changed above some threshold
        dist_squared = np.sum(np.square(self.obj.get_position() - cache_info["pos"]))
        self._has_moved = dist_squared > m.POSITIONAL_VALIDATION_EPSILON
        return self._has_moved
