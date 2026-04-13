import math

import torch as th

import omnigibson as og
from omnigibson.macros import create_module_macros
from omnigibson.object_states.object_state_base import BooleanStateMixin
from omnigibson.object_states.tensorized_value_state import TensorizedValueState
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.usd_utils import RigidContactAPI

# Create settings for this module
m = create_module_macros(module_path=__file__)
m.REACTIVATION_DELAY = 2.0  # number of seconds to wait before reactivating the slicer


class SlicerActive(TensorizedValueState, BooleanStateMixin):
    # int: Keep track of how many steps each object is waiting for
    STEPS_TO_WAIT = None

    # th.tensor (scene_number, obj_number): Keep track of the current delay for a given slicer
    DELAY_COUNTER = None

    # th.tensor (scene_number, obj_number): Keep track of whether we touched a sliceable in the previous timestep
    PREVIOUSLY_TOUCHING = None

    # list[list[str]] indexed by [obj_idx]: relative prim paths of links belonging to each slicer obj
    SLICER_LINK_PATHS = None

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        return deps

    @classmethod
    def global_initialize(cls):
        # Call super first
        super().global_initialize()

        # Compute step-based reactivation threshold (constant for the lifetime of the simulator)
        cls.STEPS_TO_WAIT = max(1, int(math.ceil(m.REACTIVATION_DELAY / og.sim.get_sim_step_dt())))

    @classmethod
    def initialize_view(cls):
        # Snapshot which relative paths existed before the rebuild
        prev_rel_paths = set(cls.OBJ_IDXS.keys()) if cls.OBJ_IDXS is not None else set()

        # Base class rebuilds OBJ_IDXS, IDX_OBJS, VALUES (with value carry-over for survivors)
        super().initialize_view()

        S = len(cls.IDX_OBJS)
        N = len(cls.OBJ_IDXS)

        # Rebuild SLICER_LINK_PATHS indexed by obj_idx.
        # Use relative prim paths so that is_in_contact lookups work with the shared path mapping.
        cls.SLICER_LINK_PATHS = [[] for _ in range(N)]
        for rel_path, obj_idx in cls.OBJ_IDXS.items():
            # Find the first non-None object instance for this obj_idx (representative across scenes)
            for s_row in cls.IDX_OBJS:
                obj = s_row[obj_idx]
                if obj is not None:
                    cls.SLICER_LINK_PATHS[obj_idx] = [link.relative_prim_path for link in obj.links.values()]
                    break

        # Allocate (scene_number, obj_number) tracking tensors; carry-over is not needed for these bookkeeping fields
        # (contact/delay state resets naturally when initialize_view is called after object changes)
        cls.PREVIOUSLY_TOUCHING = th.zeros((S, N), dtype=th.bool, device="cuda")
        cls.DELAY_COUNTER = th.zeros((S, N), dtype=th.float32, device="cuda")

        # Initialize new VALUE slots (not carried over) to True (slicer starts active)
        for rel_path, obj_idx in cls.OBJ_IDXS.items():
            if rel_path not in prev_rel_paths:
                for s_idx in range(S):
                    if cls.IDX_OBJS[s_idx][obj_idx] is not None:
                        cls.VALUES[s_idx, obj_idx] = True
                        cls.VALUES_CPU[s_idx, obj_idx] = True

    @classmethod
    def _update_values(cls, values):
        # values: (scene_number, obj_number) bool tensor
        new_values = values.clone()

        # If we were touching a sliceable last step, deactivate now
        cls.DELAY_COUNTER[cls.PREVIOUSLY_TOUCHING] = 0
        new_values[cls.PREVIOUSLY_TOUCHING] = False

        # Are we currently touching any sliceables?
        currently_touching_sliceables = cls._currently_touching_sliceables()

        # If any values are False, consider reverting back to active
        if not th.all(new_values):
            not_active_not_touching = ~new_values & ~currently_touching_sliceables
            not_active_is_touching = ~new_values & currently_touching_sliceables

            # Increment cooldown when not active and not touching anything sliceable
            cls.DELAY_COUNTER[not_active_not_touching] += 1

            # Reset cooldown when not active but touching a sliceable (touching delays reactivation)
            cls.DELAY_COUNTER[not_active_is_touching] = 0

            # Re-activate once the cooldown has elapsed
            new_values = th.where(cls.DELAY_COUNTER >= cls.STEPS_TO_WAIT, True, new_values)

        # Record current contact state for the next step
        cls.PREVIOUSLY_TOUCHING = currently_touching_sliceables

        return new_values

    @classmethod
    def _currently_touching_sliceables(cls):
        """Returns (S, N) bool tensor: True where slicer n in scene s is touching a sliceable."""
        currently_touching = th.zeros_like(cls.PREVIOUSLY_TOUCHING)
        for scene in og.sim.scenes:
            sliceable_objs = scene.object_registry("abilities", "sliceable", [])

            # If there's no sliceables in the scene, continue to the next scene.
            if len(sliceable_objs) == 0:
                continue

            # Relative prim paths of all sliceable links in this scene
            sliceable_link_paths = [link.relative_prim_path for obj in sliceable_objs for link in obj.links.values()]

            s_idx = scene.idx
            for obj_idx, link_paths in enumerate(cls.SLICER_LINK_PATHS):
                if cls.IDX_OBJS[s_idx][obj_idx] is None:
                    continue
                if RigidContactAPI.is_in_contact(
                    s_idx, link_paths, sliceable_link_paths, ignore_set=None, current_only=False
                ):
                    currently_touching[s_idx, obj_idx] = True

        return currently_touching

    @classproperty
    def value_name(cls):
        return "value"

    @classproperty
    def value_type(cls):
        return th.bool

    @property
    def state_size(self):
        # Call super first
        size = super().state_size

        # Add additional 2 to keep track of previously touching and delay counter
        return size + 2

    # For this state, we simply store its value.
    def _dump_state(self):
        state = super()._dump_state()
        s = self.obj.scene.idx
        obj_idx = self.OBJ_IDXS[self.obj.relative_prim_path]
        state["previously_touching"] = bool(self.PREVIOUSLY_TOUCHING[s, obj_idx])
        state["delay_counter"] = int(self.DELAY_COUNTER[s, obj_idx])
        return state

    def _load_state(self, state):
        super()._load_state(state=state)
        s = self.obj.scene.idx
        obj_idx = self.OBJ_IDXS[self.obj.relative_prim_path]
        self.PREVIOUSLY_TOUCHING[s, obj_idx] = state["previously_touching"]
        self.DELAY_COUNTER[s, obj_idx] = state["delay_counter"]

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
