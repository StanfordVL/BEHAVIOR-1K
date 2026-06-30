import torch as th
import warp as wp

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.object_states.adjacency import Adjacency
from omnigibson.object_states.kinematics_mixin import KinematicsMixin
from omnigibson.object_states.object_state_base import BooleanStateMixin
from omnigibson.object_states.tensorized_relative_state import TensorizedRelativeState
from omnigibson.object_states.touching import Touching
from omnigibson.utils.constants import PrimType
from omnigibson.utils.object_state_utils import get_reachability_sampling_context
from omnigibson.utils.object_state_utils import m as os_m
from omnigibson.utils.object_state_utils import sample_kinematics
from omnigibson.utils.python_utils import classproperty


# Adjacency axis layout: k=0 is +Z (other above self), k=1 is -Z (other below self).
# OnTop(self, other) is true when:
#   - self is touching other
#   - other is directly below self (Adjacency k=1)
#   - other is NOT also above self (Adjacency k=0) — disambiguates the rare both-axes case.


@wp.kernel
def _on_top_kernel(
    touching_values: wp.array3d(dtype=wp.uint8),  # (S, N_t, N_t) — Touching.VALUES uint8 view
    adjacency_values: wp.array4d(dtype=wp.uint8),  # (S, N_a, N_a, K) — Adjacency.VALUES uint8 view
    touch_idx: wp.array(dtype=wp.int32),  # (N,) — OnTop idx → Touching idx, -1 if missing
    adj_idx: wp.array(dtype=wp.int32),  # (N,) — OnTop idx → Adjacency idx, -1 if missing
    values: wp.array3d(dtype=wp.uint8),  # (S, N, N) — OnTop.VALUES uint8 view
):
    s, i, j = wp.tid()

    if i == j:
        values[s, i, j] = wp.uint8(0)
        return

    t_i = touch_idx[i]
    t_j = touch_idx[j]
    a_i = adj_idx[i]
    a_j = adj_idx[j]
    if t_i < 0 or t_j < 0 or a_i < 0 or a_j < 0:
        values[s, i, j] = wp.uint8(0)
        return

    touching = touching_values[s, t_i, t_j] != wp.uint8(0)
    above = adjacency_values[s, a_i, a_j, 0] != wp.uint8(0)  # k=0 → other above self
    below = adjacency_values[s, a_i, a_j, 1] != wp.uint8(0)  # k=1 → other below self

    if touching and below and not above:
        values[s, i, j] = wp.uint8(1)
    else:
        values[s, i, j] = wp.uint8(0)


class OnTop(TensorizedRelativeState, KinematicsMixin, BooleanStateMixin):
    # (N,) int32 mappings from OnTop's N-index to the corresponding state's N-index.
    _touch_idx = None  # wp.array(N,) int32 — OnTop idx → Touching idx, -1 if missing
    _adj_idx = None  # wp.array(N,) int32 — OnTop idx → Adjacency idx, -1 if missing

    @classproperty
    def value_shape(cls):
        return ()

    @classproperty
    def value_type(cls):
        return th.bool

    @classproperty
    def value_name(cls):
        return "on_top"

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.update({Touching, Adjacency})
        return deps

    @classmethod
    def initialize_view(cls):
        super().initialize_view()
        N = len(cls.OBJ_IDXS)
        if N == 0:
            cls._touch_idx = None
            cls._adj_idx = None
            return

        touch_idx_cpu = th.full((N,), -1, dtype=th.int32)
        adj_idx_cpu = th.full((N,), -1, dtype=th.int32)
        touch_map = Touching.OBJ_IDXS or {}
        adj_map = Adjacency.OBJ_IDXS or {}
        for rel_path, idx in cls.OBJ_IDXS.items():
            touch_idx_cpu[idx] = touch_map.get(rel_path, -1)
            adj_idx_cpu[idx] = adj_map.get(rel_path, -1)

        cls._touch_idx = lazy.isaacsim.core.utils.warp.tensor.create_tensor_from_list(
            touch_idx_cpu, "int32", device="cuda"
        )
        cls._adj_idx = lazy.isaacsim.core.utils.warp.tensor.create_tensor_from_list(adj_idx_cpu, "int32", device="cuda")

    @classmethod
    def _update_values(cls, values):
        if cls.VALUES_WP is None or cls._touch_idx is None or cls._adj_idx is None:
            return
        if Touching.VALUES_WP is None or Adjacency.VALUES_WP is None:
            # Upstream not ready — leave VALUES as-is (will be zeros from initial alloc).
            return

        S, N, _ = values.shape
        if S == 0 or N == 0:
            return

        wp.launch(
            kernel=_on_top_kernel,
            dim=(S, N, N),
            inputs=[
                Touching.VALUES_WP,
                Adjacency.VALUES_WP,
                cls._touch_idx,
                cls._adj_idx,
                cls.VALUES_WP,
            ],
            device="cuda",
        )

    def _get_value(self, other):
        if other.prim_type == PrimType.CLOTH:
            raise ValueError("Cannot detect if an object is on top of a cloth object.")

        return super()._get_value(other)

    def _set_value(self, other, new_value, reset_before_sampling=False, use_trav_map=True):
        if not new_value:
            raise NotImplementedError("OnTop does not support set_value(False)")

        if other.prim_type == PrimType.CLOTH:
            raise ValueError("Cannot set an object on top of a cloth object.")

        state = og.sim.dump_state(serialized=False)

        # Possibly reset this object if requested
        if reset_before_sampling:
            self.obj.reset()

        reachability_context = get_reachability_sampling_context(other, "onTop", use_trav_map=use_trav_map)
        for _ in range(os_m.DEFAULT_HIGH_LEVEL_SAMPLING_ATTEMPTS):
            if sample_kinematics(
                "onTop", self.obj, other, use_trav_map=use_trav_map, reachability_context=reachability_context
            ) and self.get_value(other):
                return True
            else:
                og.sim.load_state(state, serialized=False)

        return False
