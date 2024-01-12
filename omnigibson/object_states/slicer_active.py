import numpy as np
from omnigibson.macros import create_module_macros
from omnigibson.object_states.object_state_base import AbsoluteObjectState
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.contact_bodies import ContactBodies
from omnigibson.object_states.update_state_mixin import UpdateStateMixin
import omnigibson as og


# Create settings for this module
m = create_module_macros(module_path=__file__)

class SlicerActive(AbsoluteObjectState, UpdateStateMixin):
    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(AABB)
        deps.add(ContactBodies)
        return deps

    def __init__(self, obj):
        super(SlicerActive, self).__init__(obj)

        self._previously_touching_sliceable = False
        self.value = True

    def _get_value(self):
        return self.value

    def _set_value(self, new_value):
        self.value = new_value
        return True
    
    def _currently_touching_sliceable(self):
        # TODO: Implement something
        # Get the contact prim list from ContactBodies, then get objects from prim paths, then check if they are sliceable
        # Use self.obj.get_state(ContactBodies) to return a set of objects that are currently touching self.obj
        return False

    def _update(self):
        # If we were slicing in the past step, deactivate now.
        if self._previously_touching_sliceable:
            self.set_value(False)

        # Are we currently touching any sliceables?
        currently_touching_sliceable = self._currently_touching_sliceable()

        # If our value is False, we need to consider reverting back.
        if not self.value:
            # If we are not touching any sliceable objects, we can revert to True.
            if currently_touching_sliceable:
                self.set_value(True)

        # Record if we were touching anything previously
        self._previously_touching_sliceable = currently_touching_sliceable


    @property
    def state_size(self):
        return 2

    # For this state, we simply store its value.
    def _dump_state(self):
        return dict(value=self.value, previously_touching_sliceable=self._previously_touching_sliceable)

    def _load_state(self, state):
        self.value = state["value"]
        self._previously_touching_sliceable = state["previously_touching_sliceable"]

    def _serialize(self, state):
        return np.array([state["value"], state["previously_touching_sliceable"]], dtype=float)

    def _deserialize(self, state):
        return dict(value=state[0], previously_touching_sliceable=state[1]), 2
