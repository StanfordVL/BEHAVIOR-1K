from igibson.object_states.contact_bodies import ContactBodies
from igibson.object_states.kinematics import KinematicsMixin
from igibson.object_states.object_state_base import BooleanState, RelativeObjectState


class Touching(KinematicsMixin, RelativeObjectState, BooleanState):
    def _set_value(self, other, new_value):
        raise NotImplementedError()

    def _get_value(self, other):
        objA_states = self.obj.states
        objB_states = other.states

        assert ContactBodies in objA_states
        assert ContactBodies in objB_states

        return self.obj.in_contact(objects=other)
