import itertools
from abc import ABC, abstractmethod

import numpy as np

from igibson.object_states.pose import Pose
from igibson.utils.utils import l2_distance

POSITIONAL_VALIDATION_EPSILON = 1e-5


class MemoizedObjectStateMixin(ABC):
    def __init__(self, *args, **kwargs):
        super(MemoizedObjectStateMixin, self).__init__(*args, **kwargs)
        self._memo = {}
        self._validation_caches = {}

    @abstractmethod
    def get_validation_cache(self, *args, **kwargs):
        pass

    @abstractmethod
    def validate_validation_cache(self, cache, *args, **kwargs):
        pass

    def get_value(self, *args, **kwargs):
        # Convert arguments to memo key.
        key = (tuple(args), tuple(kwargs.items()))

        # If we have a valid memoized result, return it directly.
        if key in self._memo:
            if self.validate_validation_cache(self._validation_caches[key], *args, **kwargs):
                return self._memo[key]

        # Otherwise, recompute the result & memoize.
        validation_cache = self.get_validation_cache(*args, **kwargs)
        result = super(MemoizedObjectStateMixin, self).get_value(*args, **kwargs)
        self._validation_caches[key] = validation_cache
        self._memo[key] = result

        # Return the result.
        return result


class PositionalValidationMemoizedObjectStateMixin(MemoizedObjectStateMixin):
    def get_validation_cache(self, *args, **kwargs):
        # Assume that args contains objects for relative states (and is empty for others).
        return tuple(obj.states[Pose].get_value()[0] for obj in itertools.chain((self.obj,), args))

    def validate_validation_cache(self, cache, *args, **kwargs):
        # Assume that args contains objects for relative states (and is empty for others).
        for obj, old_pos in zip(itertools.chain((self.obj,), args), cache):
            new_pos, _ = obj.states[Pose].get_value()

            # Don't do expensive vector work if the positions are already equal.
            if np.all(np.asarray(new_pos) == np.asarray(old_pos)):
                continue

            dist = l2_distance(new_pos, old_pos)
            if dist > POSITIONAL_VALIDATION_EPSILON:
                return False

        return True
