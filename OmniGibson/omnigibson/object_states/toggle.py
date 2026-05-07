import torch as th
import warp as wp

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros
from omnigibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from omnigibson.object_states.object_state_base import BooleanStateMixin
from omnigibson.object_states.open_state import Open
from omnigibson.object_states.tensorized_absolute_state import TensorizedAbsoluteState
from omnigibson.prims.geom_prim import GeomPrim
from omnigibson.utils.constants import PrimType
from omnigibson.utils.numpy_utils import vtarray_to_torch
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.usd_utils import (
    RigidBodyViewAPI,
    RigidContactAPI,
    absolute_prim_path_to_scene_relative,
    create_primitive_mesh,
    rigid_inverse_mat44,
)

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.TOGGLE_META_LINK_TYPE = "togglebutton"
m.DEFAULT_SCALE = 0.1
m.CAN_TOGGLE_STEPS = 5


@wp.kernel
def _check_overlap_kernel(
    pose_matrices: wp.array(dtype=wp.mat44),  # RigidBodyViewAPI.POSE_MATRICES (N_links,)
    mesh_ids: wp.array(dtype=wp.uint64),  # RigidBodyViewAPI.LINK_MESH_IDS  (N_links,)
    marker_parent_link_idx: wp.array(dtype=wp.int32),  # (n_markers,) flat link idx of marker's parent rigid body
    marker_local_offset: wp.array(dtype=wp.vec3),  # (n_markers,) marker center in its parent link's local frame
    marker_radii: wp.array(dtype=wp.float32),  # (n_markers,) marker radius
    marker_finger_pair: wp.array(dtype=wp.vec2i),  # (P,) — pair[p] = vec2i(marker_idx, finger_link_flat_idx)
    marker_to_obj_idx_flat: wp.array(
        dtype=wp.int32
    ),  # (n_markers,) flat (scene_idx*num_objects + obj_idx) for each marker
    mask_can_toggle_flat: wp.array(dtype=wp.int32),  # (S*O,)
):
    """
    Each thread checks a (marker, finger_link) pair.

    This kernel does the following things:
    - get marker's world center = parent rigid body's pose @ static local offset, then transform into the
    finger link's local frame,
    - use wp.mesh_query_point_no_sign(finger_mesh, marker_position, marker_radius) to check whether overlap
    - On hit, atomic-max the (scene, obj) mask cell up to 2

    mask needs to be int 32 because Warp's atomic_max doesn't support uint8.
    """
    thread_id = wp.tid()
    pair = marker_finger_pair[thread_id]
    marker_idx = pair[0]
    obj_idx = marker_to_obj_idx_flat[marker_idx]
    # Skip if this marker not touched by any fingers
    if mask_can_toggle_flat[obj_idx] != wp.int32(1):
        return
    finger_link_idx = pair[1]
    finger_mesh_id = mesh_ids[finger_link_idx]
    if finger_mesh_id == wp.uint64(0):
        return

    # Derive marker world center from its parent link's current pose + static local offset.
    parent_pose = pose_matrices[marker_parent_link_idx[marker_idx]]
    offset = marker_local_offset[marker_idx]
    marker_center_world_frame = wp.mul(parent_pose, wp.vec4(offset[0], offset[1], offset[2], 1.0))
    # Transform marker center from world to finger link local frame.
    inv_pose = rigid_inverse_mat44(pose_matrices[finger_link_idx])
    cl4 = wp.mul(inv_pose, marker_center_world_frame)
    marker_center_local_frame = wp.vec3(cl4[0], cl4[1], cl4[2])

    marker_radius = marker_radii[marker_idx]
    query = wp.mesh_query_point_no_sign(finger_mesh_id, marker_center_local_frame, marker_radius)
    if query.result:
        wp.atomic_max(mask_can_toggle_flat, obj_idx, wp.int32(2))


@wp.kernel
def _check_requires_closed_kernel(
    requires_closed_idx_in_this: wp.array(dtype=wp.int32),  # (R,) flat idx into VALUES (s*O + obj)
    requires_closed_idx_in_open: wp.array(dtype=wp.int32),  # (R,) flat idx into Open.VALUES
    open_values: wp.array(dtype=wp.uint8),  # Open.VALUES_WP flattened (S*O_open,)
    toggle_values: wp.array(dtype=wp.uint8),  # VALUES.view(-1) (S*O_toggle,)
    robots_can_toggle_steps: wp.array(dtype=wp.float32),  # (S*O,)
    mask_can_toggle: wp.array(dtype=wp.int32),  # (S*O,)
):
    """
    Each thread checks 1 object in 1 scene.
    If Open.VALUES == True, Toggle.VALUES = False, robots_can_toggle_steps = 0, mask_can_toggle = False.
    """
    thread_id = wp.tid()
    if open_values[requires_closed_idx_in_open[thread_id]] != wp.uint8(0):
        idx = requires_closed_idx_in_this[thread_id]
        toggle_values[idx] = wp.uint8(0)
        robots_can_toggle_steps[idx] = 0.0
        mask_can_toggle[idx] = wp.int32(0)


@wp.kernel
def _set_toggle_value_kernel(
    values: wp.array2d(dtype=wp.uint8),  # (S, O)
    mask_can_toggle_flat: wp.array(dtype=wp.int32),  # (S*O,) — single cached 1D view (see _update_values)
    robots_can_toggle_steps: wp.array2d(dtype=wp.float32),  # (S, O)
    O: wp.int32,  # second dim of `values` — to flatten (s, o) → s*O+o for the mask
    can_toggle_threshold: wp.float32,
):
    """
    Each thread works on 1 object in 1 scene.
    When mask == 2, all three requirements passed (contact, requires_closed, overlaps).
    Increment step and flip values where the step reach threshold.
    Reset mask to 0.
    """
    s, o = wp.tid()
    idx = s * O + o
    eligible = wp.int32(0)
    if mask_can_toggle_flat[idx] == wp.int32(2):
        eligible = wp.int32(1)

    if eligible != wp.int32(0):
        robots_can_toggle_steps[s, o] = robots_can_toggle_steps[s, o] + 1.0
    else:
        robots_can_toggle_steps[s, o] = 0.0

    flip = wp.int32(0)
    if robots_can_toggle_steps[s, o] == can_toggle_threshold:
        flip = eligible

    mask_can_toggle_flat[idx] = wp.int32(0)

    if flip != wp.int32(0):
        values[s, o] = wp.uint8(1) - values[s, o]


class ToggledOn(TensorizedAbsoluteState, BooleanStateMixin, LinkBasedStateMixin):
    """
    Boolean state representing whether an object has been toggled on.
    """

    # S = number of scenes
    # O = number of toggleable objects

    # th.Tensor (S, O) int
    _robots_can_toggle_steps = None
    _robots_can_toggle_steps_wp = None

    # (O_requires_closed,) int64
    # These are indices into flattened views of Open.VALUES and ToggledOn.VALUES.
    _requires_closed_obj_idxes_in_open_values = None
    _requires_closed_obj_idxes_in_this_values = None
    _requires_closed_obj_idxes_in_open_values_wp = None
    _requires_closed_obj_idxes_in_this_values_wp = None

    # list[list[GeomPrim]]: visual toggle-button markers, one per tracked object. Shape (S, O).
    # Used in _check_overlap and color updates.
    visual_markers = None

    # Contact masks
    # R_s = number of contact-matrix rows (links on the "who is touching" side) for scene s
    # C_s = number of contact-matrix columns (links on the "what are they touching" side) for scene s
    _finger_query_mask = None  # list[Tensor(1, R_s) | None] — finger row mask per scene
    _finger_query_mask_wp = None
    _toggable_objs_with_mask = None  #  list[Tensor(O, C_s) | None] — toggle-object col masks per scene
    _toggable_objs_with_mask_wp = None

    # list[list of links of manipulation robots in scene s], len = S
    _finger_links = []

    # Scratch masks, helper for calculate update, (S, O) int32 storage; only the flat 1D
    # wp.array view is exposed (single source of truth — refreshed in initialize_view alongside
    # the underlying tensor, plus a cached per-scene row view for is_in_contact_batch_warp).
    _mask_can_toggle = None
    _mask_can_toggle_flat_wp = None

    # Per-scene out-row of _mask_can_toggle (uint8) for is_in_contact_batch_warp.
    _mask_can_toggle_per_scene_wp = None  # list[wp.array | None] — (O,) per scene

    # === Marker info — filled in initialize_view from USD reads. ===
    # Marker is a static visual child of the togglebutton meta link, so its local offset and
    # radius never change. Per-step world center is derived inside the overlap kernel from the
    # parent link's current pose matrix in RigidBodyViewAPI.POSE_MATRICES.
    _marker_parent_link_idx_gpu = None  # (n_markers,) int32 — flat link idx in RigidBodyViewAPI
    _marker_parent_link_idx_wp = None
    _marker_local_offset_gpu = None  # (n_markers, 3) float32 — marker center in parent link's local frame
    _marker_local_offset_wp = None
    _marker_radii_gpu = None  # (n_markers,) float32 — sphere radius
    _marker_radii_wp = None

    # === Pair index buffers built once in initialize_view (covers all (marker, finger_link) pairs across scenes). ===
    # (P,) wp.vec2i — each row is (marker_idx, finger_link_flat_idx) for one (marker, finger) pair.
    _marker_finger_pair = None
    _marker_to_obj_idx_flat_gpu = None  # (n_markers,) int32 — flat (s*O + obj_idx) per marker (storage anchor)
    _marker_to_obj_idx_flat_wp = None  # wp.array view

    COLOR_ON = th.tensor([0, 1.0, 0])  # green  — toggle is on
    COLOR_OFF = th.tensor([1.0, 0, 0])  # red    — toggle is off

    @classproperty
    def value_type(cls):
        return th.bool

    @classproperty
    def value_name(cls):
        return "toggle"

    @classmethod
    def global_initialize(cls):
        super().global_initialize()

        cls._robots_can_toggle_steps = None
        cls._requires_closed_obj_idxes_in_open_values = None
        cls._requires_closed_obj_idxes_in_this_values = None
        cls.visual_markers = []

        cls._finger_links = []
        cls._finger_query_mask = None
        cls._toggable_objs_with_mask = None

        cls._mask_can_toggle = None

    @classmethod
    def initialize_view(cls):
        """
        Rebuild all class-level tensors after scene changes.
        """
        # Snapshot existing states
        prev_obj_idxs = dict(cls.OBJ_IDXS) if cls.OBJ_IDXS is not None else {}
        prev_steps = cls._robots_can_toggle_steps.clone() if cls._robots_can_toggle_steps is not None else None

        # Base class rebuilds OBJ_IDXS, IDX_OBJS, VALUES (with value carry-over for toggle bool)
        super().initialize_view()

        S, O = len(cls.IDX_OBJS), len(cls.OBJ_IDXS)

        # Carry over _can_toggle_steps for surviving objects
        cls._robots_can_toggle_steps = th.zeros((S, O), dtype=th.float32, device="cuda")
        if prev_steps is not None:
            for relative_prim_path, obj_idx_old in prev_obj_idxs.items():
                if relative_prim_path not in cls.OBJ_IDXS:
                    continue
                obj_idx = cls.OBJ_IDXS[relative_prim_path]
                for scene_idx in range(min(prev_steps.shape[0], S)):
                    cls._robots_can_toggle_steps[scene_idx, obj_idx] = prev_steps[scene_idx, obj_idx_old]
        cls._robots_can_toggle_steps_wp = wp.from_torch(cls._robots_can_toggle_steps)

        cls._init_requires_closed_logic(O)

        if S == 0 or O == 0:
            cls._init_empty_states()
            return

        marker_finger_pairs = cls._init_finger(S, O)
        cls._init_marker(S, O, marker_finger_pairs)

    @classmethod
    def _init_requires_closed_logic(cls, O):
        """
        Build the (R,) flat-index lookup tables consumed by `_check_requires_closed_kernel`:
        for each (scene, toggle_obj) pair where requires_closed=True, store its flat index in
        ToggledOn.VALUES (s*O + o_toggle) AND its flat index in Open.VALUES (s*O_open + o_open).
        Two different O dims are why we need both.
        """
        requires_closed_obj_idxes_in_open_values = []
        requires_closed_obj_idxes_in_this_values = []
        for scene_idx, scene in enumerate(cls.IDX_OBJS):
            for obj_idx, toggle_obj in enumerate(scene):
                if toggle_obj is None:
                    continue
                if not toggle_obj.states[ToggledOn].requires_closed:
                    continue
                requires_closed_obj_idxes_in_this_values.append(scene_idx * O + obj_idx)
                idx_in_open_object_dim = Open.OBJ_IDXS[toggle_obj.relative_prim_path]
                open_values_object_dim_size = Open.VALUES.shape[1]
                requires_closed_obj_idxes_in_open_values.append(
                    scene_idx * open_values_object_dim_size + idx_in_open_object_dim
                )
        # int32 so the kernel can index with wp.int32.
        cls._requires_closed_obj_idxes_in_open_values = th.tensor(
            requires_closed_obj_idxes_in_open_values, dtype=th.int32, device="cuda"
        )
        cls._requires_closed_obj_idxes_in_this_values = th.tensor(
            requires_closed_obj_idxes_in_this_values, dtype=th.int32, device="cuda"
        )
        if cls._requires_closed_obj_idxes_in_open_values.numel() > 0:
            cls._requires_closed_obj_idxes_in_open_values_wp = wp.from_torch(
                cls._requires_closed_obj_idxes_in_open_values
            )
            cls._requires_closed_obj_idxes_in_this_values_wp = wp.from_torch(
                cls._requires_closed_obj_idxes_in_this_values
            )
        else:
            cls._requires_closed_obj_idxes_in_open_values_wp = None
            cls._requires_closed_obj_idxes_in_this_values_wp = None

    @classmethod
    def _init_empty_states(cls):
        """
        Empty-scene (S == 0 or O == 0) early-init: clear every per-scene/per-marker buffer
        to a safe default (None or empty list) so `_update_values` short-circuits cleanly.
        """
        cls._finger_query_mask = []
        cls._toggable_objs_with_mask = []
        cls._mask_can_toggle = th.zeros((0, 0), dtype=th.int32, device="cuda")
        cls._mask_can_toggle_flat_wp = None
        cls._robots_can_toggle_steps_wp = None
        cls._requires_closed_obj_idxes_in_open_values_wp = None
        cls._requires_closed_obj_idxes_in_this_values_wp = None
        cls._finger_query_mask_wp = []
        cls._toggable_objs_with_mask_wp = []
        cls._mask_can_toggle_per_scene_wp = []
        cls._marker_parent_link_idx_gpu = None
        cls._marker_parent_link_idx_wp = None
        cls._marker_local_offset_gpu = None
        cls._marker_local_offset_wp = None
        cls._marker_radii_gpu = None
        cls._marker_radii_wp = None
        cls._marker_finger_pair = None
        cls._marker_to_obj_idx_flat_gpu = None
        cls._marker_to_obj_idx_flat_wp = None

    @classmethod
    def _init_finger(cls, S, O):
        """
        Per-scene init for is_in_contact_batch_wp_kernel
          - Collect a scene's manipulation-robot finger links.
          - Build finger query_mask + toggle-object with_mask for is_in_contact_batch_warp.
          - Allocate the (S, O) int32 mask buffer + per-scene row wp.array slices used as output tensor for kernel
          - Collect (marker_idx, finger_link_flat_idx) pairs that the overlap kernel will iterate
            over (Stage 4); only valid markers are added so the kernel never sees garbage poses.

        Returns:
            list[(int, int)]: marker_finger_pairs, consumed by _init_marker.
        """
        marker_finger_pairs = []  # list[(marker_idx, finger_link_flat_idx)]
        cls._finger_query_mask = []
        cls._toggable_objs_with_mask = []
        cls._finger_query_mask_wp = []
        cls._toggable_objs_with_mask_wp = []
        cls._mask_can_toggle_per_scene_wp = []

        cls._mask_can_toggle = th.zeros((S, O), dtype=th.int32, device="cuda")
        cls._mask_can_toggle_flat_wp = wp.from_torch(cls._mask_can_toggle.view(-1))

        for scene_idx, scene in enumerate(og.sim.scenes):
            # Get all finger links and their idx in RigidBodyViewAPI in this scene
            finger_links = []
            finger_link_flat_idxs = []
            for robot in scene.robots:
                if robot.is_manipulation:
                    for links in robot.finger_links.values():
                        for link in links:
                            finger_links.append(link)
                            finger_link_flat_idxs.append(RigidBodyViewAPI.get_flat_idx(link.prim_path))

            cls._finger_links.append(finger_links)
            if not finger_links:
                # Keep all 5 per-scene lists in lockstep so the per-step loop can
                # index any of them by scene_idx without an IndexError.
                cls._finger_query_mask.append(None)
                cls._toggable_objs_with_mask.append(None)
                cls._finger_query_mask_wp.append(None)
                cls._toggable_objs_with_mask_wp.append(None)
                cls._mask_can_toggle_per_scene_wp.append(None)
                continue

            for obj_idx in range(O):
                toggle_obj = cls.IDX_OBJS[scene_idx][obj_idx]
                # Skip pair generation for objects whose marker isn't set yet (object not
                # fully _initialize'd, or asset has no togglebutton meta link). Their marker
                # static info is left at zeros, so we must also skip here so the kernel
                # doesn't query BVH with a garbage parent pose.
                if toggle_obj is None or toggle_obj.states[ToggledOn].marker is None:
                    continue
                marker_idx_flat = scene_idx * O + obj_idx
                for link_flat in finger_link_flat_idxs:
                    marker_finger_pairs.append((marker_idx_flat, link_flat))

            # Build finger query mask
            row_mask = RigidContactAPI.get_contact_row_mask(scene_idx, finger_links)  # (R_s,) CPU
            cls._finger_query_mask.append(row_mask.unsqueeze(0).cuda())  # (1, R_s) GPU

            # Build toggle-able object with mask — shape (O, C_s)
            toggleable_obj_with_mask = []
            any_uninitialized = False
            for obj_idx in range(O):
                if cls.IDX_OBJS[scene_idx][obj_idx] is None:
                    any_uninitialized = True
                    break
                toggleable_obj_with_mask.append(
                    RigidContactAPI.get_contact_col_mask(
                        scene_idx, list(cls.IDX_OBJS[scene_idx][obj_idx].links.values())
                    )
                )
            if any_uninitialized:
                cls._toggable_objs_with_mask.append(None)
            else:
                cls._toggable_objs_with_mask.append(th.stack(toggleable_obj_with_mask).cuda())

            qmask = cls._finger_query_mask[scene_idx]
            wmask = cls._toggable_objs_with_mask[scene_idx]
            if qmask is None or wmask is None:
                cls._finger_query_mask_wp.append(None)
                cls._toggable_objs_with_mask_wp.append(None)
                cls._mask_can_toggle_per_scene_wp.append(None)
                continue
            cls._finger_query_mask_wp.append(wp.from_torch(qmask.contiguous().view(th.uint8), dtype=wp.uint8))
            cls._toggable_objs_with_mask_wp.append(wp.from_torch(wmask.contiguous().view(th.uint8), dtype=wp.uint8))
            cls._mask_can_toggle_per_scene_wp.append(wp.from_torch(cls._mask_can_toggle[scene_idx]))

        return marker_finger_pairs

    @classmethod
    def _init_marker(cls, S, O, marker_finger_pairs):
        """
        Init marker info for check_overlap_kernel
          - cls.visual_markers[s][o]: GeomPrim handle (for color updates in post_update).
          - marker_to_obj_idx_flat: m → flat (s*O + o) for the atomic_max target.
          - marker_parent_link_idx: m → flat link idx in RigidBodyViewAPI.POSE_MATRICES.
          - marker_local_offset: m → marker center expressed in parent link's local frame.
          - marker_radii: m → BVH query radius (scale * min mesh extent).
          - cls._marker_finger_pair: (P,) wp.vec2i wrapping the (marker, finger) pair list
            collected by _init_finger.

        Marker world center is derived inside the kernel each step from the parent link's
        current pose @ this static local offset — no per-step USD reads.
        """
        n_markers = S * O
        cls.visual_markers = [[None] * O for _ in range(S)]
        marker_to_obj_idx_flat = th.zeros((n_markers,), dtype=th.int32)
        marker_parent_link_idx = th.zeros((n_markers,), dtype=th.int32)
        marker_local_offset = th.zeros((n_markers, 3), dtype=th.float32)
        marker_radii = th.zeros((n_markers,), dtype=th.float32)

        for scene_idx, scene_row in enumerate(cls.IDX_OBJS):
            for obj_idx, toggle_obj in enumerate(scene_row):
                if toggle_obj is None:
                    continue
                state = toggle_obj.states[ToggledOn]
                cls.visual_markers[scene_idx][obj_idx] = state.marker

                marker_idx_flat = scene_idx * O + obj_idx
                marker_to_obj_idx_flat[marker_idx_flat] = marker_idx_flat
                # Skip if marker isn't initialized yet — state.link would assert and there's
                # nothing meaningful to bake. _init_finger also skips pair generation for these
                # markers, so the kernel never reads parent_link_idx / local_offset / radii here.
                if state.marker is None:
                    continue
                link = state.link  # safe: marker exists ⇒ _initialize completed ⇒ link valid
                # Compute marker center in link's local frame from current world poses:
                marker_pos, _ = state.marker.get_position_orientation()
                link_pos, link_ori = link.get_position_orientation()
                marker_parent_link_idx[marker_idx_flat] = RigidBodyViewAPI.get_flat_idx(link.prim_path)
                marker_local_offset[marker_idx_flat] = T.quat2mat(link_ori).T @ (marker_pos - link_pos)
                marker_radii[marker_idx_flat] = th.min(state.marker.extent * state.marker.scale).item()

        # Move marker data to GPU and wrap as wp.arrays.
        cls._marker_to_obj_idx_flat_gpu = marker_to_obj_idx_flat.cuda()
        cls._marker_parent_link_idx_gpu = marker_parent_link_idx.cuda()
        cls._marker_local_offset_gpu = marker_local_offset.cuda()
        cls._marker_radii_gpu = marker_radii.cuda()
        cls._marker_to_obj_idx_flat_wp = wp.from_torch(cls._marker_to_obj_idx_flat_gpu)
        cls._marker_parent_link_idx_wp = wp.from_torch(cls._marker_parent_link_idx_gpu)
        cls._marker_local_offset_wp = wp.from_torch(cls._marker_local_offset_gpu, dtype=wp.vec3)
        cls._marker_radii_wp = wp.from_torch(cls._marker_radii_gpu)

        # Wrap (marker, finger) pair list as wp.array of vec2i (each row a 2-element int32 vec).
        if marker_finger_pairs:
            cls._marker_finger_pair = wp.from_torch(
                th.tensor(marker_finger_pairs, dtype=th.int32, device="cuda"),
                dtype=wp.vec2i,
            )
        else:
            cls._marker_finger_pair = None

    def __init__(self, obj, scale=None, requires_closed=False):
        self.scale = scale

        if requires_closed:
            assert Open in obj.states, f"ToggledOn requires_closed=True but {obj.name} has no Open state."

        # Only used for being written into class tensor by initialize_view()
        self._requires_closed_individual = requires_closed

        self.marker = None  # init as None, will be filled in initialize()

        super().__init__(obj)

    @property
    def requires_closed(self):
        return self._requires_closed_individual

    @classmethod
    def is_compatible(cls, obj, **kwargs):
        # Run super first
        compatible, reason = super().is_compatible(obj, **kwargs)
        if not compatible:
            return compatible, reason

        # Check whether this state has toggledon if required or open if required
        if kwargs.get("requires_closed", False) and Open not in obj.states:
            return False, f"{cls.__name__} has requires_closed but obj has no Open state!"

        return True, None

    @classmethod
    def is_compatible_asset(cls, prim, **kwargs):
        # Run super first
        compatible, reason = super().is_compatible_asset(prim, **kwargs)
        if not compatible:
            return compatible, reason

        # Check whether this state has toggledon if required or open if required
        if kwargs.get("requires_closed", False) and not Open.is_compatible_asset(prim=prim, **kwargs)[0]:
            return False, f"{cls.__name__} has requires_closed but obj has no Open state!"

        return True, None

    @classmethod
    def get_optional_dependencies(cls):
        deps = super().get_optional_dependencies()
        deps.add(Open)
        return deps

    @classproperty
    def meta_link_types(cls):
        return [m.TOGGLE_META_LINK_TYPE]

    @classmethod
    def _check_overlap(cls, scene_idx, obj_idx):
        """
        Deprecated in warp version. TODO (vector) delete this func
        Check whether any robot finger overlaps the toggle-button marker sphere for the object
        at class-level index (s_idx, obj_idx).

        Args:
            s_idx (int): Scene index.
            obj_idx (int): Object type index into cls.IDX_OBJS / cls.visual_markers.

        Returns:
            bool: True if a robot finger overlaps the marker sphere.
        """
        valid_hit = False
        finger_prim_paths = {link.prim_path for link in cls._finger_links[scene_idx]}

        def overlap_callback(hit):
            nonlocal valid_hit
            valid_hit = hit.rigid_body in finger_prim_paths
            # Continue traversal only if we don't have a valid hit yet
            return not valid_hit

        marker = cls.visual_markers[scene_idx][obj_idx]
        # TODO: This is a temporary fix for flatcache before we properly implement trigger volumes
        if marker is None:
            return False
        og.sim.psqi.overlap_sphere(
            radius=th.min(marker.extent * marker.scale).item(),
            pos=marker.get_position_orientation()[0].tolist(),
            reportFn=overlap_callback,
        )
        return valid_hit

    @classmethod
    def _update_values(cls, values):
        """
        Single-mask tri-state filter chain. The mask carries 3 different meanings across stages:
            0 = shouldn't be toggled
            1 = a finger is in contact with this toggle object
            2 = a finger is in contact AND a finger physically overlaps the marker sphere

        Stages (all run inside wp.graph):
        1. Zero the mask.
        2. use is_in_contact_batch_warp to check whether finger and marekr is in contact,
            writes mask in {0, 1}.
        3. requires_closed: for objects that are Open yet require closed, force values=0,
           steps=0, mask=0.
        4. check_overlap kernel: for mask==1, run BVH point-mesh query; on hit, atomic_max
           the cell to 2.
        5. Finalize: if mask == 2, increment step counter, XOR-flip values
           when counter hits the threshold, then normalize mask back to {0, 1}.
        """
        if cls._mask_can_toggle_flat_wp is None:
            return
        S, O = values.shape[:2]

        mask_flat_wp = cls._mask_can_toggle_flat_wp
        values_flat_wp = wp.from_torch(values.view(-1).view(th.uint8), dtype=wp.uint8)
        steps_flat_wp = wp.from_torch(cls._robots_can_toggle_steps.view(-1))

        mask_flat_wp.zero_()

        # check whether finger & marker touching
        for scene_idx in range(S):
            query_mask_wp = cls._finger_query_mask_wp[scene_idx]
            with_mask_wp = cls._toggable_objs_with_mask_wp[scene_idx]
            out_wp = cls._mask_can_toggle_per_scene_wp[scene_idx]
            if query_mask_wp is None or with_mask_wp is None or out_wp is None:
                continue
            RigidContactAPI.is_in_contact_batch_warp(
                scene_idx=scene_idx,
                query_masks_wp=query_mask_wp,  # (1, R_s)
                with_masks_wp=with_mask_wp,  # (O, C_s)
                ignore_masks_wp=None,
                current_only=False,
                out_wp=out_wp,
            )

        # check requires_closed
        if cls._requires_closed_obj_idxes_in_open_values_wp is not None and Open.VALUES_WP is not None:
            R = cls._requires_closed_obj_idxes_in_open_values.shape[0]
            open_flat_wp = wp.from_torch(Open.VALUES.view(-1).view(th.uint8), dtype=wp.uint8)
            wp.launch(
                kernel=_check_requires_closed_kernel,
                dim=R,
                inputs=[
                    cls._requires_closed_obj_idxes_in_this_values_wp,
                    cls._requires_closed_obj_idxes_in_open_values_wp,
                    open_flat_wp,
                    values_flat_wp,
                    steps_flat_wp,
                    mask_flat_wp,
                ],
                device="cuda",
            )

        # check finger & marker overlap
        if cls._marker_finger_pair is not None:
            wp.launch(
                kernel=_check_overlap_kernel,
                dim=cls._marker_finger_pair.shape[0],
                inputs=[
                    wp.from_torch(RigidBodyViewAPI.POSE_MATRICES, dtype=wp.mat44),
                    RigidBodyViewAPI.LINK_MESH_IDS,
                    cls._marker_parent_link_idx_wp,
                    cls._marker_local_offset_wp,
                    cls._marker_radii_wp,
                    cls._marker_finger_pair,
                    cls._marker_to_obj_idx_flat_wp,
                    mask_flat_wp,
                ],
                device="cuda",
            )

        # flip value and increment step
        wp.launch(
            kernel=_set_toggle_value_kernel,
            dim=(S, O),
            inputs=[
                cls.VALUES_WP,
                cls._mask_can_toggle_flat_wp,
                cls._robots_can_toggle_steps_wp,
                wp.int32(O),
                wp.float32(m.CAN_TOGGLE_STEPS),
            ],
            device="cuda",
        )

    @classmethod
    def post_update(cls):
        """Sync visual marker colors for changed objects."""
        diff = cls.VALUES_CPU != cls.PREV_VALUES
        changed_mask = th.any(diff, dim=tuple(range(2, diff.ndim))) if diff.ndim > 2 else diff
        for s_idx in range(cls.VALUES_CPU.shape[0]):
            for obj_idx in th.where(changed_mask[s_idx])[0].tolist():
                obj = cls.IDX_OBJS[s_idx][obj_idx]
                obj.state_updated()
                marker = cls.visual_markers[s_idx][obj_idx]
                marker.color = cls.COLOR_ON if bool(cls.VALUES_CPU[s_idx, obj_idx].item()) else cls.COLOR_OFF

    def _get_value(self):
        # Return toggle boolean from the (scene_idx, obj_idx, 0) entry of the shared VALUES tensor.
        s = self.obj.scene.idx
        obj_idx = self.OBJ_IDXS[self.obj.relative_prim_path]
        return bool(self.VALUES[s, obj_idx].item())

    def _set_value(self, new_value):
        """
        Set the toggle state directly (e.g. from BDDL task initialization or external scripts).
        Also syncs the visual marker color using the class-level COLOR_ON / COLOR_OFF constants.

        Args:
            new_value (bool): Desired toggle on/off state.

        Returns:
            bool: True if set successfully; False if blocked by requires_closed + Open state.
        """
        if new_value and self.requires_closed and self.obj.states[Open].get_value():
            # If the object is open, we cannot toggle it on
            return False

        s = self.obj.scene.idx
        obj_idx = self.OBJ_IDXS[self.obj.relative_prim_path]
        self.VALUES[s, obj_idx] = 1.0 if new_value else 0.0
        if self.marker is not None:
            self.marker.color = type(self).COLOR_ON if new_value else type(self).COLOR_OFF
        return True

    def _initialize(self):
        super()._initialize()
        self.initialize_link_mixin()

        # Make sure this object is not cloth
        assert self.obj.prim_type != PrimType.CLOTH, f"Cannot create ToggledOn state for cloth object {self.obj.name}!"

        # See if the mesh exists at the latest dataset's target location
        mesh_prim_path = f"{self.link.prim_path}/visuals/mesh_0"
        pre_existing_mesh = lazy.isaacsim.core.utils.prims.get_prim_at_path(mesh_prim_path)

        # If not, see if it exists in the legacy format's location
        # TODO: Remove this after new dataset release
        if not pre_existing_mesh:
            mesh_prim_path = f"{self.link.prim_path}/mesh_0"
            pre_existing_mesh = lazy.isaacsim.core.utils.prims.get_prim_at_path(mesh_prim_path)

        # Create a primitive mesh if neither option exists
        if not pre_existing_mesh:
            mesh_prim_path = f"{self.link.prim_path}/visuals/mesh_0"
            self.scale = m.DEFAULT_SCALE if self.scale is None else self.scale
            # Note: We have to create a mesh (instead of a sphere shape) because physx complains
            # about non-uniform scaling for non-meshes
            create_primitive_mesh(prim_path=mesh_prim_path, primitive_type="Sphere", extents=1.0)
        else:
            # Infer radius from mesh if not specified as an input
            with og.sim.editing_usd():
                lazy.isaacsim.core.utils.bounds.recompute_extents(prim=pre_existing_mesh)
            self.scale = vtarray_to_torch(pre_existing_mesh.GetAttribute("xformOp:scale").Get())

        # Create the visual geom instance referencing the generated mesh prim
        relative_prim_path = absolute_prim_path_to_scene_relative(self.obj.scene, mesh_prim_path)
        self.marker = GeomPrim(relative_prim_path=relative_prim_path, name=f"{self.obj.name}_visual_marker")
        self.marker.load(self.obj.scene)
        self.marker.scale = self.scale
        self.marker.initialize()
        self.marker.visible = True
        self.marker.color = type(self).COLOR_OFF

    @staticmethod
    def get_texture_change_params():
        # By default, it keeps the original albedo unchanged.
        albedo_add = 0.0
        diffuse_tint = th.tensor([1.0, 1.0, 1.0])
        return albedo_add, diffuse_tint

    @property
    def state_size(self):
        # Two floats: toggle_state + robot_can_toggle_steps. Same as the pre-tensorized value.
        return 2

    def _dump_state(self):
        if self.OBJ_IDXS is None or self.obj.relative_prim_path not in self.OBJ_IDXS:
            return dict(value=False, hand_in_marker_steps=0)
        s = self.obj.scene.idx
        obj_idx = self.OBJ_IDXS[self.obj.relative_prim_path]
        return dict(
            value=bool(self.VALUES[s, obj_idx].item()),
            hand_in_marker_steps=int(self._robots_can_toggle_steps[s, obj_idx].item()),
        )

    def _load_state(self, state):
        # Restore toggle via _set_value so the visual marker color is also updated.
        self._set_value(state["value"])
        # Restore step counter directly into the tensor.
        s = self.obj.scene.idx
        obj_idx = self.OBJ_IDXS[self.obj.relative_prim_path]
        type(self)._robots_can_toggle_steps[s, obj_idx] = float(state["hand_in_marker_steps"])

    def serialize(self, state):
        # [toggle_state, can_toggle_steps] as float32
        return th.tensor([state["value"], state["hand_in_marker_steps"]], dtype=th.float32)

    def deserialize(self, state):
        return dict(value=bool(state[0].item()), hand_in_marker_steps=int(state[1].item())), 2
