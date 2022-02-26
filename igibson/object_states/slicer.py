import numpy as np
from igibson.object_states import ContactBodies, Sliced
from igibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from igibson.object_states.object_state_base import AbsoluteObjectState, NONE

_SLICER_LINK_NAME = "slicer"


class Slicer(AbsoluteObjectState, LinkBasedStateMixin):
    def __init__(self, obj):
        super(Slicer, self).__init__(obj)

    @staticmethod
    def get_state_link_name():
        return _SLICER_LINK_NAME

    def _initialize(self):
        self.initialize_link_mixin()

    def _update(self):
        slicer_position = self.get_link_position()
        if slicer_position is None:
            return
        contact_points = self.obj.states[ContactBodies].get_value()
        for item in contact_points:
            if item.linkIndexA != self.link_id:
                continue
            contact_obj = self.simulator.scene.objects_by_id[item.bodyUniqueIdB]
            if Sliced in contact_obj.states:
                if (
                    not contact_obj.states[Sliced].get_value()
                    and item.normalForce > contact_obj.states[Sliced].slice_force
                ):
                    contact_obj.states[Sliced].set_value(True)

    def _set_value(self, new_value):
        raise ValueError("set_value not supported for valueless states like Slicer.")

    def _get_value(self):
        pass

    @staticmethod
    def get_dependencies():
        return AbsoluteObjectState.get_dependencies() + [ContactBodies]
