import math

import torch as th
import warp as wp

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import create_module_macros
from omnigibson.object_states.object_state_base import BooleanStateMixin
from omnigibson.object_states.tensorized_absolute_state import TensorizedAbsoluteState
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.usd_utils import RigidContactAPI

# Create settings for this module
m = create_module_macros(module_path=__file__)
m.REACTIVATION_DELAY = 2.0  # number of seconds to wait before reactivating the slicer


@wp.kernel
def _slicer_zero_currently_touching_kernel(
    currently_touching: wp.array2d(dtype=wp.int32),  # (S, O)
):
    """
    Per (scene, obj) thread. Zeroes `currently_touching` so the subsequent per-scene
    is_in_contact_batch_warp atomic_max only needs to set-on-hit.

    Lives inside the captured wp.graph so the zero is stream-ordered with the contact
    queries that consume it — replaces an out-of-graph torch fill_() that relied on
    legacy-default-stream semantics for cross-stream ordering with Warp's capture stream.
    """
    s, o = wp.tid()
    currently_touching[s, o] = wp.int32(0)


@wp.kernel
def _slicer_pre_clear_kernel(
    values: wp.array2d(dtype=wp.uint8),  # (S, O) — public state, stays uint8
    delay_counter: wp.array2d(dtype=wp.float32),  # (S, O)
    prev_touching: wp.array2d(dtype=wp.int32),  # (S, O) — int32, see SlicerActive class doc
):
    """
    Per (scene, obj) thread. Runs at the start of each step. For each cell where prev_touching
    is nonzero, set values=0 (deactivated) and delay_counter=0 (cooldown restarted). Cells
    where prev_touching is zero are left alone.
    """
    s, o = wp.tid()
    if prev_touching[s, o] != wp.int32(0):
        delay_counter[s, o] = 0.0
        values[s, o] = wp.uint8(0)


@wp.kernel
def _slicer_post_update_kernel(
    values: wp.array2d(dtype=wp.uint8),  # (S, O) — public state, stays uint8
    delay_counter: wp.array2d(dtype=wp.float32),  # (S, O)
    currently_touching: wp.array2d(dtype=wp.int32),  # (S, O) — int32, written by atomic in is_in_contact_batch_warp
    prev_touching: wp.array2d(dtype=wp.int32),  # (S, O) — int32 to mirror currently_touching
    steps_to_wait: wp.float32,
):
    """
    Per (scene, obj) thread. Per thread does:
      - If the slicer is currently active: leave delay_counter alone (it's not in cooldown).
      - If the slicer is inactive AND not touching anything: advance delay_counter by 1.
      - If the slicer is inactive AND still touching something: reset delay_counter to 0
        (touching restarts the wait — the slicer doesn't reactivate while in contact).
      - If delay_counter has hit `steps_to_wait`, reactivate the slicer.
      - Finally, mirror currently_touching into prev_touching for the next step's pre-clear pass.
    """
    s, o = wp.tid()
    is_active = values[s, o] != wp.uint8(0)
    is_touching = currently_touching[s, o] != wp.int32(0)
    if not is_active:
        if not is_touching:
            delay_counter[s, o] = delay_counter[s, o] + 1.0
        else:
            delay_counter[s, o] = 0.0
    if delay_counter[s, o] >= steps_to_wait:
        values[s, o] = wp.uint8(1)
    prev_touching[s, o] = currently_touching[s, o]


class SlicerActive(TensorizedAbsoluteState, BooleanStateMixin):
    """
    Slicer-active state.

    Note on dtype: PREVIOUSLY_TOUCHING and _currently_touching are int32, NOT bool/uint8, because
    they depend on`RigidContactAPI.is_in_contact_batch_warp` whose output is int32.
    """

    # int: Keep track of how many steps each object is waiting for
    STEPS_TO_WAIT = None

    # wp.array2d (S, O) float32 — current delay counter per slicer.
    DELAY_COUNTER = None

    # wp.array2d (S, O) int32 — whether each slicer touched a sliceable in the previous step.
    # int32 (not bool) — see class docstring; mirrors _currently_touching's dtype.
    PREVIOUSLY_TOUCHING = None

    # S = number of scenes
    # O = number of objects that have SlicerActive state
    # R_s = number of contact-matrix rows (links on the "who is touching" side) for scene s
    # C_s = number of contact-matrix columns (links on the "what are they touching" side) for scene s

    # list[wp.array(O, R_s) uint8 | None] — row mask per slicer object per scene.
    _slicer_contact_query_masks = None

    # list[wp.array(1, C_s) uint8 | None] — col mask for all sliceable links per scene.
    _sliceable_contact_col_mask = None

    # wp.array2d (S, O) int32 — filled each step by _currently_touching_sliceables().
    _currently_touching = None
    # Per-scene wp row slice views of _currently_touching, used as is_in_contact_batch_warp out=.
    _currently_touching_per_scene = None  # list[wp.array | None]

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

        # Snapshot tracking tensors before rebuild so survivors can be carried over.
        # wp.to_torch shares storage; .cpu() forces a single GPU→CPU copy for the carry-over loop.
        prev_previously_touching_cpu = (
            wp.to_torch(cls.PREVIOUSLY_TOUCHING).cpu() if cls.PREVIOUSLY_TOUCHING is not None else None
        )
        prev_delay_counter_cpu = wp.to_torch(cls.DELAY_COUNTER).cpu() if cls.DELAY_COUNTER is not None else None
        prev_obj_idxs = dict(cls.OBJ_IDXS) if cls.OBJ_IDXS is not None else {}

        # Base class rebuilds OBJ_IDXS, IDX_OBJS, VALUES (with value carry-over for survivors)
        super().initialize_view()

        S = len(cls.IDX_OBJS)
        O = len(cls.OBJ_IDXS)

        # Build fresh tracking tensors via CPU scratch, then ship to GPU wp.arrays.
        # Carry over survivors so PREVIOUSLY_TOUCHING set during a slicing step is not lost when
        # initialize_view() runs again on the next step (e.g. new objects initialized).
        if S == 0 or O == 0:
            cls.PREVIOUSLY_TOUCHING = None
            cls.DELAY_COUNTER = None
            cls._currently_touching = None
        else:
            new_previously_touching_cpu = th.zeros((S, O), dtype=th.int32)
            new_delay_counter_cpu = th.zeros((S, O), dtype=th.float32)
            if prev_previously_touching_cpu is not None and prev_previously_touching_cpu.numel() > 0:
                for rel_path, obj_idx_new in cls.OBJ_IDXS.items():
                    if rel_path not in prev_obj_idxs:
                        continue
                    obj_idx_old = prev_obj_idxs[rel_path]
                    n_scenes = min(prev_previously_touching_cpu.shape[0], S)
                    new_previously_touching_cpu[:n_scenes, obj_idx_new] = prev_previously_touching_cpu[
                        :n_scenes, obj_idx_old
                    ]
                    new_delay_counter_cpu[:n_scenes, obj_idx_new] = prev_delay_counter_cpu[:n_scenes, obj_idx_old]
            cls.PREVIOUSLY_TOUCHING = lazy.isaacsim.core.utils.warp.tensor.create_tensor_from_list(
                new_previously_touching_cpu, "int32", device="cuda"
            )
            cls.DELAY_COUNTER = lazy.isaacsim.core.utils.warp.tensor.create_tensor_from_list(
                new_delay_counter_cpu, "float32", device="cuda"
            )
            cls._currently_touching = wp.zeros((S, O), dtype=wp.int32, device="cuda")

        # Initialize new VALUE slots (not carried over) to True (slicer starts active)
        for rel_path, obj_idx in cls.OBJ_IDXS.items():
            if rel_path not in prev_rel_paths:
                cls.VALUES[:, obj_idx] = True
                cls.VALUES_CPU[:, obj_idx] = True

        # Build per-scene contact masks (wp.array uint8) and per-scene out-row slices.
        cls._slicer_contact_query_masks = []
        cls._sliceable_contact_col_mask = []
        cls._currently_touching_per_scene = []

        def _append_none_for_scene():
            cls._slicer_contact_query_masks.append(None)
            cls._sliceable_contact_col_mask.append(None)
            cls._currently_touching_per_scene.append(None)

        for scene_idx, scene in enumerate(og.sim.scenes):
            if not RigidContactAPI.has_contact_view(scene_idx):
                _append_none_for_scene()
                continue

            sliceable_objs = scene.object_registry("abilities", "sliceable", [])
            # No sliceable objects in this scene, or no slicers tracked anywhere.
            if not sliceable_objs or O == 0:
                _append_none_for_scene()
                continue

            # Col mask (1, C_s) for all sliceable links — shared across all O slicer queries.
            sliceable_paths = [link.prim_path for obj in sliceable_objs for link in obj.links.values()]
            sliceable_col = RigidContactAPI.get_contact_col_mask(scene_idx, sliceable_paths)  # (C_s,) CPU bool

            # Row masks (O, R_s) — one query per slicer object. If any slicer object isn't
            # initialized yet, skip this scene; will be retried on the next initialize_view.
            slicer_masks = []
            any_uninitialized = False
            for obj_idx in range(O):
                if cls.IDX_OBJS[scene_idx][obj_idx] is None:
                    any_uninitialized = True
                    break
                slicer_masks.append(
                    RigidContactAPI.get_contact_row_mask(
                        scene_idx, [link.prim_path for link in cls.IDX_OBJS[scene_idx][obj_idx].links.values()]
                    )
                )  # (R_s,) CPU bool
            if any_uninitialized:
                _append_none_for_scene()
                continue

            with_mask_data = sliceable_col.unsqueeze(0).to(th.uint8)  # (1, C_s) CPU uint8 tensor
            query_masks_data = th.stack(slicer_masks).to(th.uint8)  # (O, R_s) CPU uint8 tensor

            cls._slicer_contact_query_masks.append(
                lazy.isaacsim.core.utils.warp.tensor.create_tensor_from_list(query_masks_data, "uint8", device="cuda")
            )
            cls._sliceable_contact_col_mask.append(
                lazy.isaacsim.core.utils.warp.tensor.create_tensor_from_list(with_mask_data, "uint8", device="cuda")
            )
            cls._currently_touching_per_scene.append(cls._currently_touching[scene_idx])

    @classmethod
    def _update_values(cls, values):
        if cls.PREVIOUSLY_TOUCHING is None:
            return
        S, O = values.shape[:2]

        wp.launch(
            kernel=_slicer_pre_clear_kernel,
            dim=(S, O),
            inputs=[cls.VALUES_WP, cls.DELAY_COUNTER, cls.PREVIOUSLY_TOUCHING],
            device="cuda",
        )

        # Zero _currently_touching inside the graph so it's stream-ordered with the per-scene
        # is_in_contact_batch_warp calls below (which atomic_max set-on-hit and require a
        # pre-zeroed output). Absent scenes have no kernel launched, so their rows stay 0.
        wp.launch(
            kernel=_slicer_zero_currently_touching_kernel,
            dim=(S, O),
            inputs=[cls._currently_touching],
            device="cuda",
        )

        cls._currently_touching_sliceables()

        wp.launch(
            kernel=_slicer_post_update_kernel,
            dim=(S, O),
            inputs=[
                cls.VALUES_WP,
                cls.DELAY_COUNTER,
                cls._currently_touching,
                cls.PREVIOUSLY_TOUCHING,
                wp.float32(cls.STEPS_TO_WAIT),
            ],
            device="cuda",
        )

    @classmethod
    def _currently_touching_sliceables(cls):
        """
        Per-scene Warp contact batch query into the corresponding row of _currently_touching.
        Caller must have pre-zeroed _currently_touching (see _slicer_zero_currently_touching_kernel);
        empty/missing scenes have no kernel launched and leave their row at 0.
        """
        for scene_idx in range(len(og.sim.scenes)):
            query_masks = cls._slicer_contact_query_masks[scene_idx]
            with_mask = cls._sliceable_contact_col_mask[scene_idx]
            out = cls._currently_touching_per_scene[scene_idx]
            if query_masks is None or with_mask is None or out is None:
                continue
            RigidContactAPI.is_in_contact_batch_warp(
                scene_idx=scene_idx,
                query_masks_wp=query_masks,
                with_masks_wp=with_mask,
                ignore_masks_wp=None,
                current_only=False,
                out_wp=out,
            )

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

    def _dump_state(self):
        if self.OBJ_IDXS is None or self.obj.relative_prim_path not in self.OBJ_IDXS:
            return dict(value=True, previously_touching=False, delay_counter=0)
        state = super()._dump_state()
        scene_idx = self.obj.scene.idx
        obj_idx = self.OBJ_IDXS[self.obj.relative_prim_path]
        # wp.to_torch is a zero-copy view of the wp.array storage.
        prev_view = wp.to_torch(type(self).PREVIOUSLY_TOUCHING)
        delay_view = wp.to_torch(type(self).DELAY_COUNTER)
        state["previously_touching"] = bool(prev_view[scene_idx, obj_idx])
        state["delay_counter"] = int(delay_view[scene_idx, obj_idx])
        return state

    def _load_state(self, state):
        super()._load_state(state=state)
        s = self.obj.scene.idx
        obj_idx = self.OBJ_IDXS[self.obj.relative_prim_path]
        wp.to_torch(type(self).PREVIOUSLY_TOUCHING)[s, obj_idx] = int(state["previously_touching"])
        wp.to_torch(type(self).DELAY_COUNTER)[s, obj_idx] = float(state["delay_counter"])

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
