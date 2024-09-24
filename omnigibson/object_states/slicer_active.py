import math

import torch as th

import omnigibson as og
from omnigibson.macros import create_module_macros
from omnigibson.object_states.contact_bodies import ContactBodies
from omnigibson.object_states.object_state_base import BooleanStateMixin
from omnigibson.object_states.tensorized_value_state import TensorizedValueState
from omnigibson.utils.python_utils import classproperty, torch_delete
from omnigibson.utils.usd_utils import RigidContactAPI

# Create settings for this module
m = create_module_macros(module_path=__file__)
m.REACTIVATION_DELAY = 0.5  # number of seconds to wait before reactivating the slicer


class SlicerActive(TensorizedValueState, BooleanStateMixin):
    # int: Keep track of how many steps each object is waiting for
    STEPS_TO_WAIT = None

    # th.tensor: Keep track of the current delay for a given slicer
    DELAY_COUNTER = None

    # th.tensor: Keep track of whether we touched a sliceable in the previous timestep
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
        cls.STEPS_TO_WAIT = max(1, int(math.ceil(m.REACTIVATION_DELAY / og.sim.get_sim_step_dt())))
        cls.DELAY_COUNTER = th.empty(0, dtype=int)
        cls.PREVIOUSLY_TOUCHING = th.empty(0, dtype=bool)
        cls.SLICER_LINK_PATHS = []

    @classmethod
    def _add_obj(cls, obj):
        # Call super first
        super()._add_obj(obj=obj)

        # Add to previously touching and delay counter
        cls.DELAY_COUNTER = th.cat([cls.DELAY_COUNTER, th.tensor([0])])
        cls.PREVIOUSLY_TOUCHING = th.cat([cls.PREVIOUSLY_TOUCHING, th.tensor([False])])

        # Add this object's prim paths to slicer paths
        cls.SLICER_LINK_PATHS.append([link.prim_path for link in obj.links.values()])

    @classmethod
    def _remove_obj(cls, obj):
        # Grab idx we'll delete before the object is deleted
        deleted_idx = cls.OBJ_IDXS[obj]

        # Remove from all internal tracked arrays
        cls.DELAY_COUNTER = torch_delete(cls.DELAY_COUNTER, [deleted_idx])
        cls.PREVIOUSLY_TOUCHING = torch_delete(cls.PREVIOUSLY_TOUCHING, [deleted_idx])
        del cls.SLICER_LINK_PATHS[deleted_idx]

        # Call super
        super()._remove_obj(obj=obj)

    @classmethod
    def _update_values(cls, values):
        # If we were slicing in the past step, deactivate now
        previously_touching_idxs = th.nonzero(cls.PREVIOUSLY_TOUCHING)
        values[previously_touching_idxs] = False
        cls.DELAY_COUNTER[previously_touching_idxs] = 0  # Reset the counter when we stop touching a sliceable object

        # Are we currently touching any sliceables?
        currently_touching_sliceables = cls._currently_touching_sliceables()

        # Track changed variables between currently and previously touched sliceables
        changed_idxs = set((cls.PREVIOUSLY_TOUCHING ^ currently_touching_sliceables).nonzero().flatten().tolist())

        # If any of our values are False, we need to consider reverting back.
        if not th.all(values):
            not_active_not_touching = ~values & ~currently_touching_sliceables
            not_active_is_touching = ~values & currently_touching_sliceables

            not_active_not_touching_idxs = th.where(not_active_not_touching)[0]
            not_active_is_touching_idxs = th.where(not_active_is_touching)[0]

            # If we are not touching any sliceable objects, we increment the delay "cooldown" counter that will
            # eventually re-activate the slicer
            cls.DELAY_COUNTER[not_active_not_touching_idxs] += 1

            # If we are touching a sliceable object, reset the counter
            cls.DELAY_COUNTER[not_active_is_touching_idxs] = 0

            # Update changed idxs to include not active not touching / is touching
            changed_idxs = set.union(changed_idxs, not_active_not_touching_idxs, not_active_is_touching_idxs)

            # If the delay counter is greater than steps to wait, set to True
            values = th.where(cls.DELAY_COUNTER >= cls.STEPS_TO_WAIT, True, values)

        # Record if we were touching anything previously
        cls.PREVIOUSLY_TOUCHING = currently_touching_sliceables

        # Add all changed objects to the current state update set in their respective scenes
        for idx in changed_idxs:
            cls.IDX_OBJS[idx].state_updated()

        return values

    @classmethod
    def _currently_touching_sliceables(cls):
        # Initialize return value as all falses
        currently_touching = th.zeros_like(cls.PREVIOUSLY_TOUCHING)

        # Grab all sliceable objects
        for scene_idx, scene in enumerate(og.sim.scenes):
            sliceable_objs = scene.object_registry("abilities", "sliceable", [])

            # If there's no sliceables, then obviously no slicer is touching any sliceable so immediately return all Falses
            if len(sliceable_objs) == 0:
                return currently_touching

            # Aggregate all link prim path indices
            all_slicer_idxs = [
                [list(RigidContactAPI.get_body_row_idx(prim_path))[1] for prim_path in link_paths]
                for link_paths in cls.SLICER_LINK_PATHS
            ]
            sliceable_idxs = [
                list(RigidContactAPI.get_body_col_idx(link.prim_path))[1]
                for obj in sliceable_objs
                for link in obj.links.values()
            ]
            impulses = RigidContactAPI.get_all_impulses(scene_idx)

            # TODO: This can be vectorized. No point in doing this tensorized state to then compute this in a loop.
            # Batch check each slicer against all sliceables
            for i, slicer_idxs in enumerate(all_slicer_idxs):
                if th.any(impulses[slicer_idxs][:, sliceable_idxs]):
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
        state["previously_touching"] = bool(self.PREVIOUSLY_TOUCHING[self.OBJ_IDXS[self.obj]])
        state["delay_counter"] = int(self.DELAY_COUNTER[self.OBJ_IDXS[self.obj]])

        return state

    def _load_state(self, state):
        super()._load_state(state=state)
        self.PREVIOUSLY_TOUCHING[self.OBJ_IDXS[self.obj]] = state["previously_touching"]
        self.DELAY_COUNTER[self.OBJ_IDXS[self.obj]] = state["delay_counter"]

    def serialize(self, state):
        state_flat = super().serialize(state=state)
        return th.cat(
            [
                state_flat,
                th.tensor([state["previously_touching"], state["delay_counter"]]),
            ]
        )

    def deserialize(self, state):
        state_dict, idx = super().deserialize(state=state)
        state_dict[f"{self.value_name}"] = bool(state_dict[f"{self.value_name}"])
        state_dict["previously_touching"] = bool(state[idx])
        state_dict["delay_counter"] = int(state[idx + 1])
        return state_dict, idx + 2
