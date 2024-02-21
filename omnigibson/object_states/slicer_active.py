import numpy as np
from omnigibson.macros import create_module_macros
from omnigibson.object_states.object_state_base import BooleanStateMixin
from omnigibson.object_states.contact_bodies import ContactBodies
from omnigibson.object_states.tensorized_value_state import TensorizedValueState
import omnigibson as og
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.usd_utils import RigidContactAPI


# Create settings for this module
m = create_module_macros(module_path=__file__)
m.REACTIVATION_DELAY = 0.5  # number of seconds to wait before reactivating the slicer


class SlicerActive(TensorizedValueState, BooleanStateMixin):
    # int: Keep track of how many steps each object is waiting for
    STEPS_TO_WAIT = None

    # np.ndarray: Keep track of the current delay for a given slicer
    DELAY_COUNTER = None

    # np.ndarray: Keep track of whether we touched a sliceable in the previous timestep
    PREVIOUSLY_TOUCHING = None

    # list of list of str: Body prim paths belonging to each slicer obj
    SLICER_LINK_PATHS = None

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(ContactBodies)
        return deps

    @classmethod
    def global_initialize(cls):
        # Call super first
        super().global_initialize()

        # Initialize other global variables
        cls.STEPS_TO_WAIT = max(1, int(np.ceil(m.REACTIVATION_DELAY / og.sim.get_rendering_dt())))
        cls.DELAY_COUNTER = np.array([], dtype=int)
        cls.PREVIOUSLY_TOUCHING = np.array([], dtype=bool)
        cls.SLICER_LINK_PATHS = []

    @classmethod
    def global_clear(cls):
        # Call super first
        super().global_clear()

        # Clear other internal state
        cls.STEPS_TO_WAIT = None
        cls.DELAY_COUNTER = None
        cls.PREVIOUSLY_TOUCHING = None
        cls.SLICER_LINK_PATHS = None

    @classmethod
    def _add_obj(cls, obj):
        # Call super first
        super()._add_obj(obj=obj)

        # Add to previously touching and delay counter
        cls.DELAY_COUNTER = np.concatenate([cls.DELAY_COUNTER, [0]])
        cls.PREVIOUSLY_TOUCHING = np.concatenate([cls.PREVIOUSLY_TOUCHING, [False]])

        # Add this object's prim paths to slicer paths
        cls.SLICER_LINK_PATHS.append([link.prim_path for link in obj.links.values()])

    @classmethod
    def _remove_obj(cls, obj):
        # Grab idx we'll delete before the object is deleted
        deleted_idx = cls.OBJ_IDXS[obj.name]

        # Remove from all internal tracked arrays
        cls.DELAY_COUNTER = np.delete(cls.DELAY_COUNTER, [deleted_idx])
        cls.PREVIOUSLY_TOUCHING = np.delete(cls.PREVIOUSLY_TOUCHING, [deleted_idx])
        del cls.SLICER_LINK_PATHS[deleted_idx]

        # Call super
        super()._remove_obj(obj=obj)

    @classmethod
    def _update_values(cls, values):
        # If we were slicing in the past step, deactivate now
        previously_touching_idxs = np.nonzero(cls.PREVIOUSLY_TOUCHING)[0]
        values[previously_touching_idxs] = False
        cls.DELAY_COUNTER[previously_touching_idxs] = 0  # Reset the counter when we stop touching a sliceable object

        # Are we currently touching any sliceables?
        currently_touching_sliceables = cls._currently_touching_sliceables()

        # If any of our values are False, we need to consider reverting back.
        if not np.all(values):
            not_active_not_touching = ~values & ~currently_touching_sliceables
            not_active_is_touching = ~values & currently_touching_sliceables

            not_active_not_touching_idxs = np.where(not_active_not_touching)[0]
            not_active_is_touching_idxs = np.where(not_active_is_touching)[0]

            # If we are not touching any sliceable objects, we increment the delay "cooldown" counter that will
            # eventually re-activate the slicer
            cls.DELAY_COUNTER[not_active_not_touching_idxs] += 1

            # If we are touching a sliceable object, reset the counter
            cls.DELAY_COUNTER[not_active_is_touching_idxs] = 0

            # If the delay counter is greater than steps to wait, set to True
            values = np.where(cls.DELAY_COUNTER >= cls.STEPS_TO_WAIT, True, values)

        # Record if we were touching anything previously
        cls.PREVIOUSLY_TOUCHING = currently_touching_sliceables

        return values

    @classmethod
    def _currently_touching_sliceables(cls):
        # Initialize return value as all falses
        currently_touching = np.zeros_like(cls.PREVIOUSLY_TOUCHING)

        # Grab all sliceable objects
        sliceable_objs = og.sim.scene.object_registry("abilities", "sliceable", [])

        # If there's no sliceables, then obviously no slicer is touching any sliceable so immediately return all Falses
        if len(sliceable_objs) == 0:
            return currently_touching

        # Aggregate all link prim path indices
        all_slicer_idxs = [[RigidContactAPI.get_body_row_idx(prim_path) for prim_path in link_paths] for link_paths in cls.SLICER_LINK_PATHS]
        sliceable_idxs = [RigidContactAPI.get_body_col_idx(link.prim_path) for obj in sliceable_objs for link in obj.links.values()]
        impulses = RigidContactAPI.get_all_impulses()

        # Batch check each slicer against all sliceables
        for i, slicer_idxs in enumerate(all_slicer_idxs):
            if np.any(impulses[slicer_idxs][:, sliceable_idxs]):
                # We are touching at least one sliceable
                currently_touching[i] = True

        return currently_touching

    @classproperty
    def value_name(cls):
        return "value"

    @classproperty
    def value_type(cls):
        return bool

    def __init__(self, obj):
        # Run super first
        super(SlicerActive, self).__init__(obj)

        # Set value to be default (True)
        self._set_value(True)

    @property
    def state_size(self):
        # Call super first
        size = super().state_size

        # Add additional 2 to keep track of previously touching and delay counter
        return size + 2

    # For this state, we simply store its value.
    def _dump_state(self):
        state = super()._dump_state()
        state["previously_touching"] = bool(self.PREVIOUSLY_TOUCHING[self.OBJ_IDXS[self.obj.name]])
        state["delay_counter"] = int(self.DELAY_COUNTER[self.OBJ_IDXS[self.obj.name]])

        return state

    def _load_state(self, state):
        super()._load_state(state=state)
        self.PREVIOUSLY_TOUCHING[self.OBJ_IDXS[self.obj.name]] = state["previously_touching"]
        self.DELAY_COUNTER[self.OBJ_IDXS[self.obj.name]] = state["delay_counter"]

    def _serialize(self, state):
        state_flat = super()._serialize(state=state)
        return np.concatenate([
            state_flat,
            [state["previously_touching"], state["delay_counter"]],
        ], dtype=float)

    def _deserialize(self, state):
        state_dict, idx = super()._deserialize(state=state)
        state_dict[f"{self.value_name}"] = bool(state_dict[f"{self.value_name}"])
        state_dict["previously_touching"] = bool(state[idx])
        state_dict["delay_counter"] = int(state[idx + 1])
        return state_dict, idx + 2
