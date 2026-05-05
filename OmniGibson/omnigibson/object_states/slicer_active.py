import math

import torch as th
import warp as wp

import omnigibson as og
from omnigibson.macros import create_module_macros
from omnigibson.object_states.object_state_base import BooleanStateMixin
from omnigibson.object_states.tensorized_value_state import TensorizedValueState
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.usd_utils import RigidContactAPI

# Create settings for this module
m = create_module_macros(module_path=__file__)
m.REACTIVATION_DELAY = 2.0  # number of seconds to wait before reactivating the slicer


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


class SlicerActive(TensorizedValueState, BooleanStateMixin):
    """
    Slicer-active state.

    Note on dtype: PREVIOUSLY_TOUCHING and _currently_touching are int32, NOT bool/uint8, because
    they depend on`RigidContactAPI.is_in_contact_batch_warp` whose output is int32.
    """

    # int: Keep track of how many steps each object is waiting for
    STEPS_TO_WAIT = None

    # th.tensor (scene_number, obj_number): Keep track of the current delay for a given slicer
    DELAY_COUNTER = None

    # th.tensor (scene_number, obj_number) int32: Whether we touched a sliceable in the previous timestep.
    # int32 (not bool) — see class docstring; mirrors _currently_touching's dtype.
    PREVIOUSLY_TOUCHING = None

    # S = number of scenes
    # O = number of objects that have SlicerActive state
    # R_s = number of contact-matrix rows (links on the "who is touching" side) for scene s
    # C_s = number of contact-matrix columns (links on the "what are they touching" side) for scene s

    # list[Tensor(O, R_s) | None] — row mask per slicer object per scene; len(list) = S; GPU
    _slicer_contact_query_masks = None
    # Per-scene wp.array wrappers (uint8) of _slicer_contact_query_masks, cached for kernels.
    _slicer_contact_query_masks_wp = None  # list[wp.array | None]

    # list[Tensor(1, C_s) | None] — col mask for all sliceable links per scene; len(list) = S; GPU
    _sliceable_contact_col_mask = None
    _sliceable_contact_col_mask_wp = None  # list[wp.array | None]

    # (S, O) int32 — filled each step by _currently_touching_sliceables() via
    # is_in_contact_batch_warp.
    _currently_touching = None

    _currently_touching_wp = None  # wp.array int32 view
    # Per-scene wp.array slice-views of _currently_touching, used as is_in_contact_batch_warp out=
    _currently_touching_per_scene_wp = None  # list[wp.array | None]
    # Cached wp.array wrappers of PREVIOUSLY_TOUCHING / DELAY_COUNTER (int32 / float32).
    PREVIOUSLY_TOUCHING_WP = None
    DELAY_COUNTER_WP = None

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

        # Snapshot tracking tensors before rebuild so survivors can be carried over
        prev_previously_touching = cls.PREVIOUSLY_TOUCHING.clone() if cls.PREVIOUSLY_TOUCHING is not None else None
        prev_delay_counter = cls.DELAY_COUNTER.clone() if cls.DELAY_COUNTER is not None else None
        prev_obj_idxs = dict(cls.OBJ_IDXS) if cls.OBJ_IDXS is not None else {}

        # Base class rebuilds OBJ_IDXS, IDX_OBJS, VALUES (with value carry-over for survivors)
        super().initialize_view()

        S = len(cls.IDX_OBJS)
        O = len(cls.OBJ_IDXS)

        # Allocate fresh tracking tensors; carry over values for surviving slicers so that
        # PREVIOUSLY_TOUCHING=True set during a slicing step is not lost when initialize_view()
        # is called again on the next step (e.g. when new objects are initialized).
        cls.PREVIOUSLY_TOUCHING = th.zeros((S, O), dtype=th.int32, device="cuda")
        cls.DELAY_COUNTER = th.zeros((S, O), dtype=th.float32, device="cuda")
        if prev_previously_touching is not None and prev_previously_touching.numel() > 0:
            for rel_path, obj_idx_new in cls.OBJ_IDXS.items():
                if rel_path not in prev_obj_idxs:
                    continue
                obj_idx_old = prev_obj_idxs[rel_path]
                n_scenes = min(prev_previously_touching.shape[0], S)
                cls.PREVIOUSLY_TOUCHING[:n_scenes, obj_idx_new] = prev_previously_touching[:n_scenes, obj_idx_old]
                cls.DELAY_COUNTER[:n_scenes, obj_idx_new] = prev_delay_counter[:n_scenes, obj_idx_old]

        # Initialize new VALUE slots (not carried over) to True (slicer starts active)
        for rel_path, obj_idx in cls.OBJ_IDXS.items():
            if rel_path not in prev_rel_paths:
                cls.VALUES[:, obj_idx] = True
                cls.VALUES_CPU[:, obj_idx] = True

        # Pre-allocate scratch buffer for _currently_touching_sliceables(); same shape as PREVIOUSLY_TOUCHING
        cls._currently_touching = th.zeros((S, O), dtype=th.int32, device="cuda")

        # Wrap whole-class GPU tensors as wp.array handles. PREVIOUSLY_TOUCHING / _currently_touching
        # are int32 (atomic_max constraint — see class docstring); contact masks below are bool → wp.uint8.
        if cls.PREVIOUSLY_TOUCHING.numel() > 0:
            cls.PREVIOUSLY_TOUCHING_WP = wp.from_torch(cls.PREVIOUSLY_TOUCHING)
            cls.DELAY_COUNTER_WP = wp.from_torch(cls.DELAY_COUNTER)
            cls._currently_touching_wp = wp.from_torch(cls._currently_touching)
        else:
            cls.PREVIOUSLY_TOUCHING_WP = None
            cls.DELAY_COUNTER_WP = None
            cls._currently_touching_wp = None

        # Build per-scene contact masks (torch) and their wp.array wrappers in one pass.
        cls._slicer_contact_query_masks = []
        cls._sliceable_contact_col_mask = []
        cls._slicer_contact_query_masks_wp = []
        cls._sliceable_contact_col_mask_wp = []
        cls._currently_touching_per_scene_wp = []

        def _append_none_for_scene():
            cls._slicer_contact_query_masks.append(None)
            cls._sliceable_contact_col_mask.append(None)
            cls._slicer_contact_query_masks_wp.append(None)
            cls._sliceable_contact_col_mask_wp.append(None)
            cls._currently_touching_per_scene_wp.append(None)

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
            sliceable_col = RigidContactAPI.get_contact_col_mask(scene_idx, sliceable_paths)  # (C_s,) CPU
            with_mask_torch = sliceable_col.unsqueeze(0).cuda()  # (1, C_s) GPU
            cls._sliceable_contact_col_mask.append(with_mask_torch)

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
                # Already appended sliceable_col; keep the lists aligned by undoing it.
                cls._sliceable_contact_col_mask.pop()
                _append_none_for_scene()
                continue
            query_masks_torch = th.stack(slicer_masks).cuda()  # (O, R_s) GPU
            cls._slicer_contact_query_masks.append(query_masks_torch)

            # wp.array wrappers for the kernels (uint8 for bool masks, int32 row-view for the out=).
            cls._slicer_contact_query_masks_wp.append(
                wp.from_torch(query_masks_torch.contiguous().view(th.uint8), dtype=wp.uint8)
            )
            cls._sliceable_contact_col_mask_wp.append(
                wp.from_torch(with_mask_torch.contiguous().view(th.uint8), dtype=wp.uint8)
            )
            cls._currently_touching_per_scene_wp.append(wp.from_torch(cls._currently_touching[scene_idx]))

    @classmethod
    def pre_update(cls):
        """
        CPU-side / outside-graph prep:
          - super(): PREV_VALUES <- VALUES_CPU
          - Zero _currently_touching here (PyTorch fill_) so the captured _update_values stays
            Warp-only. The per-scene is_in_contact_batch_warp inside the graph only writes to
            scenes that have a contact view; absent scenes need to be False from the start.
        """
        super().pre_update()
        if cls._currently_touching is not None and cls._currently_touching.numel() > 0:
            # int32 dtype now (atomic_max constraint); fill with 0.
            cls._currently_touching.fill_(0)

    @classmethod
    def _update_values(cls, values):
        if cls.PREVIOUSLY_TOUCHING_WP is None:
            return
        S, O = values.shape[:2]

        wp.launch(
            kernel=_slicer_pre_clear_kernel,
            dim=(S, O),
            inputs=[cls.VALUES_WP, cls.DELAY_COUNTER_WP, cls.PREVIOUSLY_TOUCHING_WP],
            device="cuda",
        )

        cls._currently_touching_sliceables()

        wp.launch(
            kernel=_slicer_post_update_kernel,
            dim=(S, O),
            inputs=[
                cls.VALUES_WP,
                cls.DELAY_COUNTER_WP,
                cls._currently_touching_wp,
                cls.PREVIOUSLY_TOUCHING_WP,
                wp.float32(cls.STEPS_TO_WAIT),
            ],
            device="cuda",
        )

    @classmethod
    def _currently_touching_sliceables(cls):
        """
        Per-scene Warp contact batch query into the corresponding row of _currently_touching.
        Empty/missing scenes leave their row as the .fill_(False) value.
        """
        for scene_idx in range(len(og.sim.scenes)):
            query_masks_wp = cls._slicer_contact_query_masks_wp[scene_idx]
            with_mask_wp = cls._sliceable_contact_col_mask_wp[scene_idx]
            out_wp = cls._currently_touching_per_scene_wp[scene_idx]
            if query_masks_wp is None or with_mask_wp is None or out_wp is None:
                continue
            RigidContactAPI.is_in_contact_batch_warp(
                scene_idx=scene_idx,
                query_masks_wp=query_masks_wp,
                with_masks_wp=with_mask_wp,
                ignore_masks_wp=None,
                current_only=False,
                out_wp=out_wp,
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
