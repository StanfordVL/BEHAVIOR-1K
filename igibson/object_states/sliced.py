import numpy as np
from collections import OrderedDict
from igibson.object_states import *
from igibson.object_states.object_state_base import AbsoluteObjectState, BooleanState

# TODO: propagate dusty/stained to object parts
_DEFAULT_SLICE_FORCE = 10
_STASH_POSITION = [-100, -100, -100]


class Sliced(AbsoluteObjectState, BooleanState):
    def __init__(self, obj, slice_force=_DEFAULT_SLICE_FORCE):
        super(Sliced, self).__init__(obj)
        self.slice_force = slice_force
        self.value = False

    def _get_value(self):
        return self.value

    def _set_value(self, new_value):
        if self.value == new_value:
            return True

        if not new_value:
            raise ValueError("Cannot set sliced from True to False")

        self.value = new_value

        # We want to return early if set_value(True) is called on a URDFObject
        # (an object part) that does not have multiplexer registered. This is
        # used when we propagate sliced=True from the whole object to all the
        # object parts.
        if not hasattr(self.obj, "multiplexer"):
            return True

        # Object parts offset annotation are w.r.t the base link of the whole object
        pos, orn = self.obj.get_position_orientation()
        body_ids = self.obj.get_body_ids()
        assert len(body_ids) == 1, "Sliceable is expected to be single-body."
        body_id = body_ids[0]
        dynamics_info = p.getDynamicsInfo(body_id, -1)
        inertial_pos = dynamics_info[3]
        inertial_orn = dynamics_info[4]
        inv_inertial_pos, inv_inertial_orn = p.invertTransform(inertial_pos, inertial_orn)
        pos, orn = p.multiplyTransforms(pos, orn, inv_inertial_pos, inv_inertial_orn)
        self.obj.set_position(_STASH_POSITION)

        # force_wakeup is needed to properly update the self.obj pose in the renderer
        self.obj.force_wakeup()

        # Dump the current object's states, for setting on the new object. Note that this might not make sense for
        # things like stains etc. where the halves are supposed to split the state rather than each get an exact copy.
        state_dump = self.obj.dump_state()

        self.obj.multiplexer.set_selection(int(self.value))

        # set the object parts to the base link pose of the whole object
        # ObjectGrouper internally manages the pose offsets of each part
        self.obj.multiplexer.set_base_link_position_orientation(pos, orn)

        # Propagate the original object's states to the halves (the ObjectGrouper takes care of propagating this call
        # to both of its objects).
        self.obj.multiplexer.current_selection().load_state(state_dump)

        return True

    # For this state, we simply store its value. The ObjectMultiplexer will be
    # loaded separately.

    @property
    def settable(self):
        return True

    def _dump_state(self):
        return OrderedDict(sliced=self.value)

    def _load_state(self, state):
        self.value = state["sliced"]

    def _serialize(self, state):
        return np.array([float(state["sliced"])])

    def _deserialize(self, state):
        return OrderedDict(sliced=(state[0] == 1.0)), 1
