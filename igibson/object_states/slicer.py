import numpy as np
from igibson.object_states import ContactBodies, Sliced
from igibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from igibson.object_states.object_state_base import AbsoluteObjectState, NONE
from omni.isaac.utils._isaac_utils import math as math_utils

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
        contact_list = self.obj.states[ContactBodies].get_value()
        # exclude links from our own object
        link_paths = {link.prim_path for link in self.obj.links.values()}
        to_cut = set()
        for c in contact_list:
            # extract the prim path of the other body
            if c.body0 in link_paths and c.body1 in link_paths:
                continue
            path = c.body0 if c.body0 not in link_paths else c.body1
            path = path.replace("/base_link", "")
            contact_obj = self._simulator.scene.object_registry("prim_path", path)
            if contact_obj is None:
                continue
            # calculate the normal force applied to the contact object
            normal_force = math_utils.dot(c.impulse, c.normal) / c.dt
            if Sliced in contact_obj.states:
                if (
                    not contact_obj.states[Sliced].get_value()
                    and normal_force > contact_obj.states[Sliced].slice_force
                ):
                    # slicer may contact the same body in multiple points, so only cut once
                    # since removing the object from the simulator
                    to_cut.add(contact_obj)
        for whole_obj in to_cut:
            whole_obj.states[Sliced].set_value(True)
                    

    def _set_value(self, new_value):
        raise ValueError("set_value not supported for valueless states like Slicer.")

    def _get_value(self):
        pass

    @staticmethod
    def get_dependencies():
        return AbsoluteObjectState.get_dependencies() + [ContactBodies]
