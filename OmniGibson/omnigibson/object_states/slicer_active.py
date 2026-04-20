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

    # S = number of scenes
    # O = number of objects that have SlicerActive state
    # R_s = number of contact-matrix rows (links on the "who is touching" side) for scene s
    # C_s = number of contact-matrix columns (links on the "what are they touching" side) for scene s

    # list[Tensor(O, R_s) | None] — row mask per slicer object per scene; len(list) = S; GPU
    _slicer_contact_query_masks = None

    # list[Tensor(1, C_s) | None] — col mask for all sliceable links per scene; len(list) = S; GPU
    _sliceable_contact_col_mask = None

    # (S, O) bool — scratch buffer filled each step by _currently_touching_sliceables(); GPU
    _currently_touching = None

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
        cls._slicer_contact_query_masks = None
        cls._sliceable_contact_col_mask = None
        cls._currently_touching = None

    @classmethod
    def initialize_view(cls):
        # Snapshot which relative paths existed before the rebuild
        prev_rel_paths = set(cls.OBJ_IDXS.keys()) if cls.OBJ_IDXS is not None else set()

        # Base class rebuilds OBJ_IDXS, IDX_OBJS, VALUES (with value carry-over for survivors)
        super().initialize_view()

        S = len(cls.IDX_OBJS)
        O = len(cls.OBJ_IDXS)

        # Allocate (scene_number, obj_number) tracking tensors; carry-over is not needed for these bookkeeping fields
        # (contact/delay state resets naturally when initialize_view is called after object changes)
        cls.PREVIOUSLY_TOUCHING = th.zeros((S, O), dtype=th.bool, device="cuda")
        cls.DELAY_COUNTER = th.zeros((S, O), dtype=th.float32, device="cuda")

        # Initialize new VALUE slots (not carried over) to True (slicer starts active)
        for rel_path, obj_idx in cls.OBJ_IDXS.items():
            if rel_path not in prev_rel_paths:
                cls.VALUES[:, obj_idx] = True
                cls.VALUES_CPU[:, obj_idx] = True

        # Pre-allocate scratch buffer for _currently_touching_sliceables(); same shape as PREVIOUSLY_TOUCHING
        cls._currently_touching = th.zeros((S, O), dtype=th.bool, device="cuda")

        # Build per-scene contact masks for batch contact queries
        cls._slicer_contact_query_masks = []
        cls._sliceable_contact_col_mask = []

        for scene_idx, scene in enumerate(og.sim.scenes):
            if not RigidContactAPI.has_contact_view(scene_idx):
                cls._slicer_contact_query_masks.append(None)
                cls._sliceable_contact_col_mask.append(None)
                continue

            sliceable_objs = scene.object_registry("abilities", "sliceable", [])
            # If there's no sliceable objects or slicer, append None
            if not sliceable_objs or O == 0:
                cls._slicer_contact_query_masks.append(None)
                cls._sliceable_contact_col_mask.append(None)
                continue

            # Col mask (1, C_s) for all sliceable links — shared across all O slicer queries
            sliceable_paths = [link.prim_path for obj in sliceable_objs for link in obj.links.values()]
            sliceable_col = RigidContactAPI.get_contact_col_mask(scene_idx, sliceable_paths)  # (C_s,) CPU
            cls._sliceable_contact_col_mask.append(sliceable_col.unsqueeze(0).cuda())  # (1, C_s) GPU

            # Row masks (O, R_s). N slicer object form N querys
            slicer_masks = [
                RigidContactAPI.get_contact_row_mask(
                    scene_idx, [link.prim_path for link in cls.IDX_OBJS[scene_idx][obj_idx].links.values()]
                )
                for obj_idx in range(O)
            ]  # list of (R_s,) CPU bool
            cls._slicer_contact_query_masks.append(th.stack(slicer_masks).cuda())  # (O, R_s) GPU

    @classmethod
    def _update_values(cls, values):
        # If we were touching a sliceable last step, deactivate now
        cls.DELAY_COUNTER[cls.PREVIOUSLY_TOUCHING] = 0
        values[cls.PREVIOUSLY_TOUCHING] = False

        # Are we currently touching any sliceables? Result written into cls._currently_touching in-place
        cls._currently_touching_sliceables()

        not_active_not_touching = ~values & ~cls._currently_touching
        not_active_is_touching = ~values & cls._currently_touching

        # Increment cooldown when not active and not touching anything sliceable
        cls.DELAY_COUNTER[not_active_not_touching] += 1

        # Reset cooldown when not active but touching a sliceable (touching delays reactivation)
        cls.DELAY_COUNTER[not_active_is_touching] = 0

        # Re-activate once the cooldown has elapsed; th.where does not trigger a cpu/gpu sync
        values.copy_(th.where(cls.DELAY_COUNTER >= cls.STEPS_TO_WAIT, True, values))

        # Record current contact state for the next step
        cls.PREVIOUSLY_TOUCHING.copy_(cls._currently_touching)

    @classmethod
    def _currently_touching_sliceables(cls):
        """Fills cls._currently_touching (S, O) bool tensor in-place."""
        cls._currently_touching.fill_(False)
        for scene_idx in range(len(og.sim.scenes)):
            query_masks = cls._slicer_contact_query_masks[scene_idx]  # (O, R_s) GPU | None
            with_mask = cls._sliceable_contact_col_mask[scene_idx]  # (1, C_s) GPU | None
            if query_masks is None or with_mask is None:
                continue
            cls._currently_touching[scene_idx] = RigidContactAPI.is_in_contact_batch(
                scene_idx=scene_idx,
                query_masks=query_masks,  # (O, R_s)
                with_masks=with_mask,  # (1, C_s) — broadcasts to (O, C_s) inside the call
                ignore_masks=None,
                current_only=False,
            )  # (O,) bool

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
        scene_idx = self.obj.scene.idx
        obj_idx = self.OBJ_IDXS[self.obj.relative_prim_path]
        state["previously_touching"] = bool(self.PREVIOUSLY_TOUCHING[scene_idx, obj_idx])
        state["delay_counter"] = int(self.DELAY_COUNTER[scene_idx, obj_idx])
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
