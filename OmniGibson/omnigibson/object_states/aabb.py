import torch as th
import warp as wp

from omnigibson.object_states.tensorized_absolute_state import TensorizedAbsoluteState
from omnigibson.object_states.tensorized_state import TensorizedState
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.usd_utils import RigidBodyViewAPI
from omnigibson.utils.constants import PrimType


@wp.kernel
def _aabb_init_kernel(out_values: wp.array2d(dtype=wp.float32)):
    """
    Initialize the (S*O, 6) AABB output buffer to [+inf, +inf, +inf, -inf, -inf, -inf].
    Replaces the PyTorch `values.fill_(inf); values[:, :, 3:] = -inf` so AABB._update_values
    is fully Warp-only and capturable inside wp.graph.
    """
    i = wp.tid()
    out_values[i, 0] = wp.float32(wp.inf)
    out_values[i, 1] = wp.float32(wp.inf)
    out_values[i, 2] = wp.float32(wp.inf)
    out_values[i, 3] = wp.float32(-wp.inf)
    out_values[i, 4] = wp.float32(-wp.inf)
    out_values[i, 5] = wp.float32(-wp.inf)


class AABB(TensorizedAbsoluteState):
    """
    Axis-aligned bounding box state, computed in bulk across all objects and scenes.

    For rigid objects, batched AABB computation is delegated to RigidBodyViewAPI.get_aabb(),
    which uses each link's wp.Mesh (built from collision_mesh_cpu_data) plus POSE_MATRICES. AABB-tracking
    config (which links to track, where their AABBs go) is handed off once per
    initialize_view() via RigidBodyViewAPI.prepare_aabb_kernel_inputs(...).

    Cloth objects are disabled for now.

    VALUES shape: (S, O, 6) — [lo_x, lo_y, lo_z, hi_x, hi_y, hi_z]
    """

    @classproperty
    def value_shape(cls):
        return (6,)

    @classproperty
    def value_name(cls):
        return "aabb"

    @classmethod
    def initialize_view(cls):
        # Snapshot before super() calls global_initialize() which resets OBJ_IDXS
        prev_rel_paths = set(cls.OBJ_IDXS.keys()) if cls.OBJ_IDXS is not None else set()

        # Base class rebuilds OBJ_IDXS, IDX_OBJS, VALUES (S, O, 6)
        super().initialize_view()

        S = len(cls.IDX_OBJS)
        O = len(cls.OBJ_IDXS)

        if S == 0 or O == 0:
            empty = th.zeros((0,), dtype=th.int32)
            RigidBodyViewAPI.prepare_aabb_kernel_inputs(empty, empty, empty, empty)
            return

        # Build the (link, output-row) mapping AABB-tracked rigid bodies need:
        #   - prim_body_idx[i] / link_idx[i]: for the i-th tracked link with collision geom,
        #     its flat body idx in RigidBodyViewAPI and its output row in (S*O,)-flattened VALUES.
        #   - base_link_*[j]: same but per rigid object's base link, used by the fallback kernel
        #     to write a point AABB for objects whose links have no collision geometry.
        prim_body_idx, link_idx = [], []
        base_link_body_idx, base_link_values_idx = [], []

        for scene_idx, scene_row in enumerate(cls.IDX_OBJS):
            for obj_index, obj in enumerate(scene_row):
                if obj is None:
                    continue
                if obj.prim_type == PrimType.CLOTH:
                    continue

                # Add the base link to the base link indices
                this_base_link_body_idx = RigidBodyViewAPI.get_flat_idx(obj.root_link.prim_path)
                assert this_base_link_body_idx is not None, "Base link not in RigidBodyViewAPI"
                base_link_body_idx.append(this_base_link_body_idx)
                base_link_values_idx.append(scene_idx * O + obj_index)

                for link in obj.links.values():
                    flat_idx = RigidBodyViewAPI.get_flat_idx(link.prim_path)
                    assert flat_idx is not None, "Articulated link not in RigidBodyViewAPI"
                    if RigidBodyViewAPI.LINK_VERTEX_COUNTS[flat_idx].item() == 0:
                        continue  # no collision geometry for this link

                    prim_body_idx.append(flat_idx)
                    link_idx.append(scene_idx * O + obj_index)

        # Hand off the index tensors to RigidBodyViewAPI. It builds the K-length kernel
        # input tables and caches them for subsequent get_aabb() calls. We don't keep
        # these locally — the only readers are inside RigidBodyViewAPI.
        RigidBodyViewAPI.prepare_aabb_kernel_inputs(
            th.tensor(prim_body_idx, dtype=th.int32),
            th.tensor(link_idx, dtype=th.int32),
            th.tensor(base_link_body_idx, dtype=th.int32),
            th.tensor(base_link_values_idx, dtype=th.int32),
        )

        # Initialize new VALUE slots for objects that just appeared
        for rel_path, obj_idx in cls.OBJ_IDXS.items():
            if rel_path not in prev_rel_paths:
                for s_idx in range(S):
                    if cls.IDX_OBJS[s_idx][obj_idx] is not None:
                        cls.VALUES[s_idx, obj_idx] = th.nan

    @classmethod
    def _update_values(cls, values):
        S = values.shape[0]
        O = values.shape[1]

        # Init the (S*O, 6) output buffer to [+inf, +inf, +inf, -inf, -inf, -inf] via a Warp
        # kernel (no PyTorch CUDA ops — keeps this method graph-capturable).
        flat_view = values.view(S * O, 6)
        out_arr = wp.from_torch(flat_view)
        wp.launch(kernel=_aabb_init_kernel, dim=S * O, inputs=[out_arr], device="cuda")

        # Batched AABB for all rigid links (physx_tracked + physx_untracked kinematic).
        # get_aabb() short-circuits internally when nothing is tracked — no guard needed.
        # It runs both the per-(link, vertex) reduce kernel and the base-link fallback kernel.
        RigidBodyViewAPI.get_aabb(flat_view)

        # Cloth — disabled for now

    def _get_value(self):
        # Bring VALUES_CPU back in sync if caches are dirty (e.g. since the last
        # set_position_orientation / step_physics), then unpack the (6,) row into (lo, hi)
        # to match EntityPrim.aabb's return shape.
        TensorizedState.maybe_refresh_caches()
        s = self.obj.scene.idx
        obj_idx = self.OBJ_IDXS[self.obj.relative_prim_path]
        v = self.VALUES_CPU[s, obj_idx]  # (6,) — CPU mirror, no GPU stall
        return v[:3], v[3:]  # (lo, hi) — matches EntityPrim.aabb return type

    def _set_value(self, new_value):
        raise NotImplementedError("AABB is read-only; it is derived from pose and cannot be set directly.")

    def _dump_state(self):
        # Return the raw flat (6,) VALUES_CPU row so serialize() receives a CPU tensor
        s = self.obj.scene.idx
        if self.obj.relative_prim_path in self.OBJ_IDXS:
            obj_idx = self.OBJ_IDXS[self.obj.relative_prim_path]
            return {self.value_name: self.VALUES_CPU[s, obj_idx].clone()}
        else:
            # obj not initialized yet. return a empty (6,) tensor
            return {self.value_name: th.zeros(self.value_shape)}

    def _load_state(self, state):
        # AABB is fully derived from pose; it will be recomputed on the next step.
        pass
