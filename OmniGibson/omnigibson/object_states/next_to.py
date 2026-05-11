import torch as th
import warp as wp

from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.adjacency import Adjacency, _ADJ_AXIS_COUNT
from omnigibson.object_states.kinematics_mixin import KinematicsMixin
from omnigibson.object_states.object_state_base import BooleanStateMixin
from omnigibson.object_states.tensorized_relative_state import TensorizedRelativeState
from omnigibson.object_states.touching import _aabb_stale_for_pair
from omnigibson.utils.python_utils import classproperty


# Adjacency axis layout: k=0..1 vertical (+Z, -Z), k=2..(K-1) horizontal.
# NextTo(self, other) is true when AABBs are close enough AND other lies on any of self's
# horizontal Adjacency axes (or, symmetrically, self lies on any of other's).
_HORIZONTAL_K_START = 2
_HORIZONTAL_K_END = _ADJ_AXIS_COUNT  # exclusive

# Distance / size threshold ratio: NextTo iff (axis-aligned gap norm) ≤ (sum of dims of both AABBs) / 18.
# That equals avg_aabb_length / 6 where avg_aabb_length = mean over 3 dims of (dim_a + dim_b).
_DISTANCE_RATIO = 1.0 / 6.0


@wp.kernel
def _next_to_kernel(
    aabb_values: wp.array3d(dtype=wp.float32),  # (S, N_aabb, 6) — [lo_x, lo_y, lo_z, hi_x, hi_y, hi_z]
    adjacency_values: wp.array4d(dtype=wp.uint8),  # (S, N_adj, N_adj, K)
    aabb_idx: wp.array(dtype=wp.int32),  # (N,) NextTo idx → AABB idx, -1 if missing
    adj_idx: wp.array(dtype=wp.int32),  # (N,) NextTo idx → Adjacency idx, -1 if missing
    values: wp.array3d(dtype=wp.uint8),  # (S, N, N) — NextTo.VALUES uint8 view
):
    s, i, j = wp.tid()

    if i == j:
        values[s, i, j] = wp.uint8(0)
        return

    a_i = aabb_idx[i]
    a_j = aabb_idx[j]
    adj_i = adj_idx[i]
    adj_j = adj_idx[j]
    if a_i < 0 or a_j < 0 or adj_i < 0 or adj_j < 0:
        values[s, i, j] = wp.uint8(0)
        return

    lo_i_x = aabb_values[s, a_i, 0]
    lo_i_y = aabb_values[s, a_i, 1]
    lo_i_z = aabb_values[s, a_i, 2]
    hi_i_x = aabb_values[s, a_i, 3]
    hi_i_y = aabb_values[s, a_i, 4]
    hi_i_z = aabb_values[s, a_i, 5]

    lo_j_x = aabb_values[s, a_j, 0]
    lo_j_y = aabb_values[s, a_j, 1]
    lo_j_z = aabb_values[s, a_j, 2]
    hi_j_x = aabb_values[s, a_j, 3]
    hi_j_y = aabb_values[s, a_j, 4]
    hi_j_z = aabb_values[s, a_j, 5]

    # Per-axis gap = max(0, max(lo_i, lo_j) - min(hi_i, hi_j)). Zero when AABBs overlap on this axis.
    gap_x = wp.max(wp.float32(0.0), wp.max(lo_i_x, lo_j_x) - wp.min(hi_i_x, hi_j_x))
    gap_y = wp.max(wp.float32(0.0), wp.max(lo_i_y, lo_j_y) - wp.min(hi_i_y, hi_j_y))
    gap_z = wp.max(wp.float32(0.0), wp.max(lo_i_z, lo_j_z) - wp.min(hi_i_z, hi_j_z))
    distance = wp.sqrt(gap_x * gap_x + gap_y * gap_y + gap_z * gap_z)

    # avg_aabb_length = th.mean(objA_dims + objB_dims) = sum(dim_a + dim_b for d in xyz) / 3
    sum_dims = (
        (hi_i_x - lo_i_x)
        + (hi_i_y - lo_i_y)
        + (hi_i_z - lo_i_z)
        + (hi_j_x - lo_j_x)
        + (hi_j_y - lo_j_y)
        + (hi_j_z - lo_j_z)
    )
    threshold = (sum_dims / wp.float32(3.0)) * wp.float32(_DISTANCE_RATIO)

    if distance > threshold:
        values[s, i, j] = wp.uint8(0)
        return

    # OR over horizontal axes in both directions. Branchless — drops the legacy short-circuit
    # between the self and other paths; total work is bounded by 2 * HORIZONTAL_DIRECTION_COUNT.
    hit = wp.uint8(0)
    for k in range(_HORIZONTAL_K_START, _HORIZONTAL_K_END):
        if adjacency_values[s, adj_i, adj_j, k] != wp.uint8(0):
            hit = wp.uint8(1)
        if adjacency_values[s, adj_j, adj_i, k] != wp.uint8(0):
            hit = wp.uint8(1)

    values[s, i, j] = hit


class NextTo(TensorizedRelativeState, KinematicsMixin, BooleanStateMixin):
    _aabb_idx = None
    _aabb_idx_wp = None
    _adj_idx = None
    _adj_idx_wp = None

    @classproperty
    def value_shape(cls):
        return ()

    @classproperty
    def value_type(cls):
        return th.bool

    @classproperty
    def value_name(cls):
        return "next_to"

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.update({Adjacency, AABB})
        return deps

    @classmethod
    def initialize_view(cls):
        super().initialize_view()
        N = len(cls.OBJ_IDXS)
        if N == 0:
            cls._aabb_idx = None
            cls._aabb_idx_wp = None
            cls._adj_idx = None
            cls._adj_idx_wp = None
            return

        aabb_idx = th.full((N,), -1, dtype=th.int32)
        adj_idx = th.full((N,), -1, dtype=th.int32)
        aabb_map = AABB.OBJ_IDXS or {}
        adj_map = Adjacency.OBJ_IDXS or {}
        for rel_path, idx in cls.OBJ_IDXS.items():
            aabb_idx[idx] = aabb_map.get(rel_path, -1)
            adj_idx[idx] = adj_map.get(rel_path, -1)

        cls._aabb_idx = aabb_idx.cuda()
        cls._adj_idx = adj_idx.cuda()
        cls._aabb_idx_wp = wp.from_torch(cls._aabb_idx)
        cls._adj_idx_wp = wp.from_torch(cls._adj_idx)

    @classmethod
    def _update_values(cls, values):
        if (
            cls.VALUES_WP is None
            or cls._aabb_idx_wp is None
            or cls._adj_idx_wp is None
            or AABB.VALUES_WP is None
            or Adjacency.VALUES_WP is None
        ):
            return
        S, N, _ = values.shape
        if S == 0 or N == 0:
            return
        wp.launch(
            kernel=_next_to_kernel,
            dim=(S, N, N),
            inputs=[
                AABB.VALUES_WP,
                Adjacency.VALUES_WP,
                cls._aabb_idx_wp,
                cls._adj_idx_wp,
                cls.VALUES_WP,
            ],
            device="cuda",
        )

    def _get_value(self, other):
        # Stale-cache fallback: AABB.get_value and Adjacency.get_value both handle their own
        # stale paths, so re-run the original distance + horizontal-adjacency computation.
        if _aabb_stale_for_pair(self.obj, other):
            objA_lower, objA_upper = self.obj.states[AABB].get_value()
            objB_lower, objB_upper = other.states[AABB].get_value()
            distance_vec = []
            for dim in range(3):
                glb = max(objA_lower[dim], objB_lower[dim])
                lub = min(objA_upper[dim], objB_upper[dim])
                distance_vec.append(max(0, glb - lub))
            distance = th.norm(th.tensor(distance_vec, dtype=th.float32))
            avg_aabb_length = th.mean(objA_upper - objA_lower + objB_upper - objB_lower)
            if distance > avg_aabb_length * (1.0 / 6.0):
                return False
            adj_self = self.obj.states[Adjacency].get_value(other)
            if bool(adj_self[2:].any()):
                return True
            adj_other = other.states[Adjacency].get_value(self.obj)
            return bool(adj_other[2:].any())

        return super()._get_value(other)
