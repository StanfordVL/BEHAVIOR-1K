import torch as th

from omnigibson.utils.usd_utils import RigidContactAPI
from omnigibson.object_states.kinematics_mixin import KinematicsMixin
from omnigibson.object_states.object_state_base import BooleanStateMixin, RelativeObjectState
from omnigibson.utils.constants import PrimType


class Touching(KinematicsMixin, RelativeObjectState, BooleanStateMixin):
    # ── Core output ─────────────────────────────────────────────────────────
    _INTER_OBJECT_TOUCHING_MATRIX = None  # (N_total, N_total) bool — single tensor, updated in-place

    # ── Global object tracking ──────────────────────────────────────────────
    _OBJ_IDXS = None  # dict[obj → int] — global index across all scenes
    _IDX_OBJS = None  # list[obj]        — global list

    # ── Rigid path — pre-built at init/add/remove, never rebuilt mid-step ───
    _BLOCK_OBJ_TO_CONTACT_ROWS = None  # (N_total, R_total) float — pre-built block-diagonal
    _BLOCK_OBJ_TO_CONTACT_COLS = None  # (N_total, C_total) float — pre-built block-diagonal
    _BLOCK_CONTACT_MATRIX = None  # (R_total, C_total) float — pre-allocated, filled in-place each step
    _ACTIVE_SCENE_IDXS = None  # list[int]  — ordered scene list
    _SCENE_ROW_OFFSETS = None  # list[int]  — row start per scene in block layout
    _SCENE_COL_OFFSETS = None  # list[int]  — col start per scene in block layout

    @classmethod
    def global_initialize(cls):
        cls._INTER_OBJECT_TOUCHING_MATRIX = None
        cls._OBJ_IDXS = {}
        cls._IDX_OBJS = []
        cls._BLOCK_OBJ_TO_CONTACT_ROWS = None
        cls._BLOCK_OBJ_TO_CONTACT_COLS = None
        cls._BLOCK_CONTACT_MATRIX = None
        cls._ACTIVE_SCENE_IDXS = None
        cls._SCENE_ROW_OFFSETS = None
        cls._SCENE_COL_OFFSETS = None

    def _initialize(self):
        super()._initialize()
        if Touching._OBJ_IDXS is None:
            Touching.global_initialize()
        idx = len(Touching._IDX_OBJS)
        Touching._IDX_OBJS.append(self.obj)
        Touching._OBJ_IDXS[self.obj] = idx
        Touching._rebuild_block_state()

    def remove(self):
        if self.obj in Touching._OBJ_IDXS:
            old_idx = Touching._OBJ_IDXS.pop(self.obj)
            Touching._IDX_OBJS.pop(old_idx)
            # Rebuild the index map from the (now shortened) list
            Touching._OBJ_IDXS = {obj: i for i, obj in enumerate(Touching._IDX_OBJS)}
            Touching._rebuild_block_state()
        super().remove()

    @classmethod
    def _rebuild_block_state(cls):
        if not cls._IDX_OBJS:
            cls._BLOCK_OBJ_TO_CONTACT_ROWS = None
            cls._BLOCK_OBJ_TO_CONTACT_COLS = None
            cls._BLOCK_CONTACT_MATRIX = None
            cls._INTER_OBJECT_TOUCHING_MATRIX = None
            return

        cls._ACTIVE_SCENE_IDXS = sorted({obj.scene.idx for obj in cls._IDX_OBJS})
        per_scene_row_masks, per_scene_col_masks = [], []
        cls._SCENE_ROW_OFFSETS, cls._SCENE_COL_OFFSETS = [], []
        r_off = c_off = 0

        for scene_idx in cls._ACTIVE_SCENE_IDXS:
            cm = RigidContactAPI.get_contact_matrix(scene_idx)
            if cm is None:
                continue
            R, C = cm.shape
            scene_objs = [obj for obj in cls._IDX_OBJS if obj.scene.idx == scene_idx]
            N = len(scene_objs)
            row_masks = th.zeros((N, R), dtype=th.float32)
            col_masks = th.zeros((N, C), dtype=th.float32)
            for local_i, obj in enumerate(scene_objs):
                if obj.prim_type != PrimType.CLOTH:
                    row_idxs = RigidContactAPI.get_contact_row_indices(scene_idx, [obj])
                    col_idxs = RigidContactAPI.get_contact_col_indices(scene_idx, [obj])
                    if len(row_idxs):
                        row_masks[local_i, row_idxs] = 1.0
                    if len(col_idxs):
                        col_masks[local_i, col_idxs] = 1.0
            per_scene_row_masks.append(row_masks)
            per_scene_col_masks.append(col_masks)
            cls._SCENE_ROW_OFFSETS.append(r_off)
            cls._SCENE_COL_OFFSETS.append(c_off)
            r_off += R
            c_off += C

        if not per_scene_row_masks:
            return

        cls._BLOCK_OBJ_TO_CONTACT_ROWS = th.block_diag(*per_scene_row_masks)  # (N_total, R_total)
        cls._BLOCK_OBJ_TO_CONTACT_COLS = th.block_diag(*per_scene_col_masks)  # (N_total, C_total)
        # Pre-allocate block contact matrix — filled in-place each step, never reallocated
        cls._BLOCK_CONTACT_MATRIX = th.zeros(
            cls._BLOCK_OBJ_TO_CONTACT_ROWS.shape[1],  # R_total
            cls._BLOCK_OBJ_TO_CONTACT_COLS.shape[1],  # C_total
            dtype=th.float32,
        )
        N_total = len(cls._IDX_OBJS)
        cls._INTER_OBJECT_TOUCHING_MATRIX = th.zeros((N_total, N_total), dtype=th.bool)

    @classmethod
    def global_update(cls):
        if cls._BLOCK_CONTACT_MATRIX is None:
            return

        # Fill pre-allocated block contact matrix in-place (no allocation)
        RigidContactAPI.update_block_contact_matrix(
            cls._ACTIVE_SCENE_IDXS,
            cls._BLOCK_CONTACT_MATRIX,
            cls._SCENE_ROW_OFFSETS,
            cls._SCENE_COL_OFFSETS,
        )

        # Two matmuls → (N_total, N_total) one-way contact
        one_way = RigidContactAPI.compute_pairwise_contacts(
            cls._BLOCK_CONTACT_MATRIX,
            cls._BLOCK_OBJ_TO_CONTACT_ROWS,
            cls._BLOCK_OBJ_TO_CONTACT_COLS,
        )

        # Symmetrize — kinematic objects have no rows, so OR both directions
        cls._INTER_OBJECT_TOUCHING_MATRIX[:] = one_way | one_way.T
        cls._INTER_OBJECT_TOUCHING_MATRIX.fill_diagonal_(False)

    @staticmethod
    def _check_cloth_contact(cloth_obj, other_obj):
        other_link_paths = set(other_obj.link_prim_paths)
        return any(contact_prim_path in other_link_paths for contact_prim_path, _ in cloth_obj.root_link.get_contacts())

    def _get_value(self, other):
        # Cloth path: fall back to per-call check
        if self.obj.prim_type == PrimType.CLOTH or other.prim_type == PrimType.CLOTH:
            if self.obj.prim_type == PrimType.CLOTH and other.prim_type == PrimType.CLOTH:
                raise ValueError("Cannot detect contact between two cloth objects.")
            cloth_obj = self.obj if self.obj.prim_type == PrimType.CLOTH else other
            rigid_obj = other if self.obj.prim_type == PrimType.CLOTH else self.obj
            return self._check_cloth_contact(cloth_obj, rigid_obj)

        # Rigid path: O(1) matrix lookup
        if Touching._INTER_OBJECT_TOUCHING_MATRIX is None:
            return False
        i = Touching._OBJ_IDXS[self.obj]
        j = Touching._OBJ_IDXS[other]
        return bool(Touching._INTER_OBJECT_TOUCHING_MATRIX[i, j].item())
