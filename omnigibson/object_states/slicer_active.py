import numpy as np
from omnigibson.macros import create_module_macros
from omnigibson.object_states.object_state_base import AbsoluteObjectState
from omnigibson.object_states.contact_bodies import ContactBodies
from omnigibson.object_states.update_state_mixin import UpdateStateMixin
import omnigibson as og


# Create settings for this module
m = create_module_macros(module_path=__file__)
m.REACTIVATION_DELAY = 0.5 # number of seconds to wait before reactivating the slicer

class SlicerActive(AbsoluteObjectState, UpdateStateMixin):
    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(ContactBodies)
        return deps

    def __init__(self, obj):
        super(SlicerActive, self).__init__(obj)

        self._previously_touching_sliceable = False
        self.value = True
        self.delay_counter = 0

    def _get_value(self):
        return self.value

    def _set_value(self, new_value):
        self.value = new_value
        return True
    
    def _currently_touching_sliceable(self):
        contact_links = self.obj.states[ContactBodies].get_value()
        contact_link_prim_paths = {contact_link.prim_path for contact_link in contact_links}
        for prim_path in contact_link_prim_paths:
            obj_prim_path, _ = prim_path.rsplit("/", 1)
            candidate_obj = og.sim.scene.object_registry("prim_path", obj_prim_path, None)
            if candidate_obj is None:
                continue
            from omnigibson.objects.stateful_object import StatefulObject
            if isinstance(candidate_obj, StatefulObject) and "sliceable" in candidate_obj.abilities:
                return True
        return False

    def _update(self):
        # If we were slicing in the past step, deactivate now.
        if self._previously_touching_sliceable:
            self.set_value(False)
            self.delay_counter = 0  # Reset the counter when we stop touching a sliceable object

        # Are we currently touching any sliceables?
        currently_touching_sliceable = self._currently_touching_sliceable()

        # If our value is False, we need to consider reverting back.
        if not self.value:
            # If we are not touching any sliceable objects, we can revert to True.
            if not currently_touching_sliceable:
                self.delay_counter += 1  # Increment the counter for each step we're not touching a sliceable object
                steps_to_wait = max(1, m.REACTIVATION_DELAY / og.sim.get_rendering_dt())
                if self.delay_counter >= steps_to_wait:  # Check if the counter has reached the delay
                    self.set_value(True)
            else:
                # If we are touching a sliceable object, reset the counter.
                self.delay_counter = 0

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
