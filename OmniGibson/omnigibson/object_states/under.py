import torch as th
import warp as wp

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.object_states.adjacency import Adjacency
from omnigibson.object_states.kinematics_mixin import KinematicsMixin
from omnigibson.object_states.object_state_base import BooleanStateMixin
from omnigibson.object_states.tensorized_relative_state import TensorizedRelativeState
from omnigibson.utils.constants import PrimType
from omnigibson.utils.object_state_utils import m as os_m
from omnigibson.utils.object_state_utils import sample_kinematics
from omnigibson.utils.python_utils import classproperty


# Adjacency axis layout: k=0 is +Z (other above self), k=1 is -Z (other below self).
# Under(self, other) is true when:
#   - other is above self (Adjacency[s, i, j, 0])
#   - AND other is NOT below self (Adjacency[s, i, j, 1])
#   - AND self is NOT above other (Adjacency[s, j, i, 0]) — guards against mutual-above ambiguity.


@wp.kernel
def _under_kernel(
    adjacency_values: wp.array4d(dtype=wp.uint8),  # (S, N_a, N_a, K)
    adj_idx: wp.array(dtype=wp.int32),  # (N,) — Under idx → Adjacency idx, -1 if missing
    values: wp.array3d(dtype=wp.uint8),  # (S, N, N) — Under.VALUES uint8 view
):
    s, i, j = wp.tid()

    if i == j:
        values[s, i, j] = wp.uint8(0)
        return

    a_i = adj_idx[i]
    a_j = adj_idx[j]
    if a_i < 0 or a_j < 0:
        values[s, i, j] = wp.uint8(0)
        return

    other_above = adjacency_values[s, a_i, a_j, 0] != wp.uint8(0)
    other_below = adjacency_values[s, a_i, a_j, 1] != wp.uint8(0)
    self_above = adjacency_values[s, a_j, a_i, 0] != wp.uint8(0)

    if other_above and not other_below and not self_above:
        values[s, i, j] = wp.uint8(1)
    else:
        values[s, i, j] = wp.uint8(0)


class Under(TensorizedRelativeState, KinematicsMixin, BooleanStateMixin):
    _adj_idx = None  # wp.array (N,) int32 — Under idx → Adjacency idx, -1 if missing

    @classproperty
    def value_shape(cls):
        return ()

    @classproperty
    def value_type(cls):
        return th.bool

    @classproperty
    def value_name(cls):
        return "under"

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(Adjacency)
        return deps

    @classmethod
    def initialize_view(cls):
        super().initialize_view()
        N = len(cls.OBJ_IDXS)
        if N == 0:
            cls._adj_idx = None
            return

        adj_idx_cpu = th.full((N,), -1, dtype=th.int32)
        adj_map = Adjacency.OBJ_IDXS or {}
        for rel_path, idx in cls.OBJ_IDXS.items():
            adj_idx_cpu[idx] = adj_map.get(rel_path, -1)

        cls._adj_idx = lazy.isaacsim.core.utils.warp.tensor.create_tensor_from_list(adj_idx_cpu, "int32", device="cuda")

    @classmethod
    def _update_values(cls, values):
        if cls.VALUES_WP is None or cls._adj_idx is None or Adjacency.VALUES_WP is None:
            return
        S, N, _ = values.shape
        if S == 0 or N == 0:
            return
        wp.launch(
            kernel=_under_kernel,
            dim=(S, N, N),
            inputs=[Adjacency.VALUES_WP, cls._adj_idx, cls.VALUES_WP],
            device="cuda",
        )

    def _get_value(self, other):
        if other.prim_type == PrimType.CLOTH:
            raise ValueError("Cannot detect if an object is under a cloth object.")

        return super()._get_value(other)

    def _set_value(self, other, new_value, reset_before_sampling=False, use_trav_map=False):
        if not new_value:
            raise NotImplementedError("Under does not support set_value(False)")

        if other.prim_type == PrimType.CLOTH:
            raise ValueError("Cannot set an object under a cloth object.")

        state = og.sim.dump_state(serialized=False)

        # Possibly reset this object if requested
        if reset_before_sampling:
            self.obj.reset()

        for _ in range(os_m.DEFAULT_HIGH_LEVEL_SAMPLING_ATTEMPTS):
            if sample_kinematics("under", self.obj, other, use_trav_map=use_trav_map) and self.get_value(other):
                return True
            else:
                og.sim.load_state(state, serialized=False)

        return False
