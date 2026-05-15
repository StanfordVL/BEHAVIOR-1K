import torch as th
import warp as wp

from omnigibson.object_states.kinematics_mixin import KinematicsMixin
from omnigibson.object_states.object_state_base import BooleanStateMixin
from omnigibson.object_states.tensorized_relative_state import TensorizedRelativeState
from omnigibson.utils.constants import PrimType
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.usd_utils import RigidContactAPI


# Tensorized Touching state.
#
# VALUES shape: (S, N, N) bool — VALUES[s, a, b] is True iff object a is currently in
# rigid contact with object b in scene s. Symmetric (a touches b ⇔ b touches a).
# Diagonal cells and cross-scene cells are always False.
#
# Computation: per-scene two-stage atomic-max contraction over the contact matrix.
#
#   Stage 1   obj_to_col[s, i, c] = ∃ r : obj_row_mask[s, i, r] ∧ contact_matrix[s, r, c]
#   Stage 2   pair[s, i, j]       = ∃ c : obj_to_col[s, i, c] ∧ obj_col_mask[s, j, c]
#   Stage 3   VALUES[s, i, j]     = (pair[s, i, j] ∨ pair[s, j, i]) ∧ (i ≠ j)
#
# Two stages instead of one collapsed kernel because launch dims would be (S, N, N, R, C);
# the contraction split keeps each launch ≤ 3D and keeps scratch memory at O(N·C + N²)
# instead of O(N² · max(R, C)).
#
# Cloth path: cloth has no rigid contact matrix entries, so the kernel writes 0 for any
# pair touching a cloth obj. `_get_value` overrides the base CPU read to fall back to
# `_check_cloth_contact` for cloth pairs — preserving the pre-tensorized behavior used
# by Overlaid.


@wp.kernel
def _touching_obj_to_col_kernel(
    obj_row_mask: wp.array2d(dtype=wp.uint8),  # (N, R_s)
    contact_matrix: wp.array2d(dtype=wp.uint8),  # (R_s, C_s)
    obj_to_col: wp.array2d(dtype=wp.int32),  # (N, C_s) — atomic_max target
):
    """Per (i, r, c) thread: set obj_to_col[i, c] = 1 if obj i's row r is in contact with col c."""
    i, r, c = wp.tid()
    if obj_row_mask[i, r] == wp.uint8(0):
        return
    if contact_matrix[r, c] == wp.uint8(0):
        return
    wp.atomic_max(obj_to_col, i, c, wp.int32(1))


@wp.kernel
def _touching_pair_kernel(
    obj_to_col: wp.array2d(dtype=wp.int32),  # (N, C_s)
    obj_col_mask: wp.array2d(dtype=wp.uint8),  # (N, C_s)
    pair: wp.array2d(dtype=wp.int32),  # (N, N) — per-scene view of pair scratch
):
    """Per (i, j, c) thread: set pair[i, j] = 1 if obj i hits col c that belongs to obj j."""
    i, j, c = wp.tid()
    if obj_to_col[i, c] == wp.int32(0):
        return
    if obj_col_mask[j, c] == wp.uint8(0):
        return
    wp.atomic_max(pair, i, j, wp.int32(1))


@wp.kernel
def _touching_finalize_kernel(
    pair: wp.array3d(dtype=wp.int32),  # (S, N, N)
    values: wp.array3d(dtype=wp.uint8),  # (S, N, N) — uint8 view of bool VALUES
):
    """Symmetrize, zero diagonal, int32 → uint8."""
    s, i, j = wp.tid()
    if i == j:
        values[s, i, j] = wp.uint8(0)
        return
    if pair[s, i, j] > wp.int32(0) or pair[s, j, i] > wp.int32(0):
        values[s, i, j] = wp.uint8(1)
    else:
        values[s, i, j] = wp.uint8(0)


class Touching(TensorizedRelativeState, KinematicsMixin, BooleanStateMixin):
    """
    Pairwise rigid-body Touching state.

    Cloth-rigid and cloth-cloth pairs go through `_get_value`'s cloth fallback,
    not the tensorized pipeline.
    """

    # Per-scene contact mask tables (rebuilt in initialize_view, sized to that scene's R_s / C_s).
    # Each entry covers N rows where N = len(OBJ_IDXS); rows for cloth / absent-in-scene objects are zero.
    _obj_row_mask = None  # list[Tensor(N, R_s) bool | None]
    _obj_row_mask_wp = None  # list[wp.array2d(uint8) | None]
    _obj_col_mask = None  # list[Tensor(N, C_s) bool | None]
    _obj_col_mask_wp = None  # list[wp.array2d(uint8) | None]

    # Stage 1 scratch: per-scene (N, C_s) int32 atomic_max target.
    _obj_to_col = None  # list[Tensor(N, C_s) int32 | None]
    _obj_to_col_wp = None  # list[wp.array2d(int32) | None]

    # Stage 2 scratch: single (S, N, N) int32 atomic_max target + per-scene 2D slice wrappers.
    _pair_scratch = None  # Tensor(S, N, N) int32
    _pair_scratch_wp = None  # wp.array3d(int32)
    _pair_scratch_per_scene_wp = None  # list[wp.array2d(int32)]

    @classproperty
    def value_shape(cls):
        return ()

    @classproperty
    def value_type(cls):
        return th.bool

    @classproperty
    def value_name(cls):
        return "touching"

    @staticmethod
    def _check_cloth_contact(cloth_obj, other_obj):
        other_link_paths = set(other_obj.link_prim_paths)
        return any(contact_prim_path in other_link_paths for contact_prim_path, _ in cloth_obj.root_link.get_contacts())

    @classmethod
    def global_initialize(cls):
        super().global_initialize()
        cls._obj_row_mask = None
        cls._obj_row_mask_wp = None
        cls._obj_col_mask = None
        cls._obj_col_mask_wp = None
        cls._obj_to_col = None
        cls._obj_to_col_wp = None
        cls._pair_scratch = None
        cls._pair_scratch_wp = None
        cls._pair_scratch_per_scene_wp = None

    @classmethod
    def initialize_view(cls):
        # Base class rebuilds OBJ_IDXS, IDX_OBJS, VALUES (with value carry-over).
        super().initialize_view()

        S = len(cls.IDX_OBJS)
        N = len(cls.OBJ_IDXS)

        cls._obj_row_mask = []
        cls._obj_row_mask_wp = []
        cls._obj_col_mask = []
        cls._obj_col_mask_wp = []
        cls._obj_to_col = []
        cls._obj_to_col_wp = []
        cls._pair_scratch_per_scene_wp = []

        if S == 0 or N == 0:
            cls._pair_scratch = None
            cls._pair_scratch_wp = None
            return

        # Allocate single (S, N, N) int32 scratch for stage 2 output; per-scene 2D views feed stage 2 kernel.
        cls._pair_scratch = th.zeros((S, N, N), dtype=th.int32, device="cuda")
        cls._pair_scratch_wp = wp.from_torch(cls._pair_scratch)

        for scene_idx, scene_row in enumerate(cls.IDX_OBJS):
            cls._pair_scratch_per_scene_wp.append(wp.from_torch(cls._pair_scratch[scene_idx]))

            shape = RigidContactAPI.get_contact_matrix_shape(scene_idx)
            if shape is None:
                cls._obj_row_mask.append(None)
                cls._obj_row_mask_wp.append(None)
                cls._obj_col_mask.append(None)
                cls._obj_col_mask_wp.append(None)
                cls._obj_to_col.append(None)
                cls._obj_to_col_wp.append(None)
                continue

            R_s, C_s = shape

            row_mask = th.zeros((N, R_s), dtype=th.bool)
            col_mask = th.zeros((N, C_s), dtype=th.bool)

            for obj_idx, obj in enumerate(scene_row):
                if obj is None:
                    continue
                if obj.prim_type == PrimType.CLOTH:
                    # Cloth gets handled by the _get_value fallback; rigid kernel sees zero masks.
                    continue
                link_paths = [link.prim_path for link in obj.links.values()]
                row_mask[obj_idx] = RigidContactAPI.get_contact_row_mask(scene_idx, link_paths)
                col_mask[obj_idx] = RigidContactAPI.get_contact_col_mask(scene_idx, link_paths)

            row_mask_gpu = row_mask.cuda()
            col_mask_gpu = col_mask.cuda()
            cls._obj_row_mask.append(row_mask_gpu)
            cls._obj_col_mask.append(col_mask_gpu)
            cls._obj_row_mask_wp.append(wp.from_torch(row_mask_gpu.view(th.uint8), dtype=wp.uint8))
            cls._obj_col_mask_wp.append(wp.from_torch(col_mask_gpu.view(th.uint8), dtype=wp.uint8))

            obj_to_col = th.zeros((N, C_s), dtype=th.int32, device="cuda")
            cls._obj_to_col.append(obj_to_col)
            cls._obj_to_col_wp.append(wp.from_torch(obj_to_col))

    @classmethod
    def _update_values(cls, values):
        if cls._pair_scratch_wp is None or cls.VALUES_WP is None:
            return

        S, N, _ = values.shape
        if S == 0 or N == 0:
            return

        # Zero the (S, N, N) pair scratch in one shot.
        cls._pair_scratch_wp.zero_()

        # Per-scene Stage 1 + Stage 2 launches.
        for scene_idx in range(S):
            obj_to_col_wp = cls._obj_to_col_wp[scene_idx]
            if obj_to_col_wp is None:
                continue
            row_mask_wp = cls._obj_row_mask_wp[scene_idx]
            col_mask_wp = cls._obj_col_mask_wp[scene_idx]
            contact_matrix_wp = RigidContactAPI.get_contact_matrix_wp(scene_idx, current_only=True)
            if contact_matrix_wp is None:
                continue
            R_s = contact_matrix_wp.shape[0]
            C_s = contact_matrix_wp.shape[1]

            # Zero stage-1 scratch.
            obj_to_col_wp.zero_()

            # Stage 1: (N, R_s, C_s) — obj_to_col[i, c] |= row_mask[i, r] & contact[r, c]
            wp.launch(
                kernel=_touching_obj_to_col_kernel,
                dim=(N, R_s, C_s),
                inputs=[row_mask_wp, contact_matrix_wp, obj_to_col_wp],
                device="cuda",
            )

            # Stage 2: (N, N, C_s) — pair[i, j] |= obj_to_col[i, c] & col_mask[j, c]
            wp.launch(
                kernel=_touching_pair_kernel,
                dim=(N, N, C_s),
                inputs=[obj_to_col_wp, col_mask_wp, cls._pair_scratch_per_scene_wp[scene_idx]],
                device="cuda",
            )

        # Stage 3: symmetrize, zero diagonal, int32 → uint8 into VALUES.
        wp.launch(
            kernel=_touching_finalize_kernel,
            dim=(S, N, N),
            inputs=[cls._pair_scratch_wp, cls.VALUES_WP],
            device="cuda",
        )

    def _get_value(self, other):
        # Cloth path: tensorized pipeline doesn't see cloth contacts (cloth isn't in the
        # rigid contact matrix), so fall back to the per-call PhysX get_contacts() check.
        self_is_cloth = self.obj.prim_type == PrimType.CLOTH
        other_is_cloth = other.prim_type == PrimType.CLOTH
        if self_is_cloth and other_is_cloth:
            raise ValueError("Cannot detect contact between two cloth objects.")
        if self_is_cloth:
            return self._check_cloth_contact(self.obj, other)
        if other_is_cloth:
            return self._check_cloth_contact(other, self.obj)

        return super()._get_value(other)
