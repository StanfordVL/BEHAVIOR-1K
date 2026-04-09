import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import create_module_macros
from omnigibson.object_states.link_based_state_mixin import LinkBasedStateMixin
from omnigibson.object_states.object_state_base import BooleanStateMixin
from omnigibson.object_states.open_state import Open
from omnigibson.object_states.tensorized_value_state import TensorizedValueState
from omnigibson.prims.geom_prim import GeomPrim
from omnigibson.utils.constants import PrimType
from omnigibson.utils.numpy_utils import vtarray_to_torch
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.usd_utils import RigidContactAPI, absolute_prim_path_to_scene_relative, create_primitive_mesh

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.TOGGLE_META_LINK_TYPE = "togglebutton"
m.DEFAULT_SCALE = 0.1
m.CAN_TOGGLE_STEPS = 5


class ToggledOn(TensorizedValueState, BooleanStateMixin, LinkBasedStateMixin):
    """
    Boolean state representing whether a button-like object has been toggled on.

    Tensorized storage
    ------------------
    Inherits TensorizedValueState. All per-object mutable values are stored in class-level
    tensors (same (scene_idx, obj_idx) coordinate as VALUES):

        VALUES          (S, N) float32       — toggle on/off stored as 0.0/1.0
        _can_toggle_steps (S, N) float32    — robot hand-in-marker step counter (not in VALUES)
        _requires_closed  (S, N) bool       — construction-time constant

    The contact masks below are pre-built once in initialize_view() and reused every step:

        _finger_contact_query_mask  (S, 1, R)  — True for robot finger rows in the contact matrix
        _obj_contact_with_mask      (S, N, C)  — True for each toggle object's col mask

    Per-step scratch masks
    ----------------------
    Named (S, N) bool tensors allocated in initialize_view(); reused in-place every step to
    avoid per-call allocations:
        _mask_force_off, _mask_active, _mask_in_contact, _mask_can_toggle, _mask_flip

    Per-instance state:
        self.scale      — temporary constructor arg; used during _initialize() to populate
                          cls.scales[obj_idx], not accessed at runtime
    """

    # th.Tensor (S, N) float32: robot hand-in-marker step counter per (scene, object).
    # Separate from VALUES so get_value() only returns the toggle bool.
    _can_toggle_steps = None

    # th.Tensor (S, N) bool: whether each tracked object requires Open state to be False
    # before it can be toggled. Construction-time constant.
    _requires_closed = None

    # (N,) long: index into Open.VALUES columns for each toggle object (-1 if no Open state).
    # Built once in initialize_view(); used to vectorize the requires_closed guard.
    _open_col_idx = None

    # list[GeomPrim]: visual toggle-button markers, one per tracked object type (N index).
    # Populated during _initialize(); used in _check_overlap and color updates.
    visual_markers = None

    # list[float | th.Tensor]: per-object marker scale (N index).
    # Populated during _initialize(); used in _check_overlap for sphere radius.
    scales = None

    # Pre-computed contact masks built in initialize_view(); reused every step.
    # _finger_row_mask:            (R,)     — finger rows in the contact matrix; computed once (robots are fixed)
    # _finger_contact_query_mask:  (S, 1, R) — _finger_row_mask broadcast to (S, 1, R); rebuilt when S or R changes
    # _obj_contact_with_mask:      (S, N, C) — column masks for each toggle object
    _finger_row_mask = None
    _finger_contact_query_mask = None
    _obj_contact_with_mask = None

    # (S, N) bool tensor: result of is_in_contact_batch from previous global_update call.
    # Written by global_update(), read by _update_values().
    _finger_contact_result = None

    # Scratch masks, helper for calculate update: (S, N) bool, allocated in initialize_view().
    _mask_force_off = None
    _mask_active = None
    _mask_in_contact = None
    _mask_can_toggle = None
    _mask_flip = None

    # Class-level color constants: defined once, reused every flip and _set_value call.
    # Avoids allocating new th.tensor([...]) on every color update.
    COLOR_ON = th.tensor([0, 1.0, 0])  # green  — toggle is on
    COLOR_OFF = th.tensor([1.0, 0, 0])  # red    — toggle is off

    @classproperty
    def value_type(cls):
        # float32; toggle_state is 0.0/1.0 (bool semantics)
        return th.float32

    @classproperty
    def value_name(cls):
        # Used by the parent's _dump_state / _load_state for the raw tensor column.
        return "toggle"

    @classmethod
    def global_initialize(cls):
        """
        Initialize all class-level state for ToggledOn. Called once at simulator start
        and on each clear/reset. Allocates VALUES (via super) and all auxiliary structures.
        """
        # Allocate VALUES, OBJ_IDXS, IDX_OBJS, STATE_SIZE
        super().global_initialize()

        cls._can_toggle_steps = None
        cls._requires_closed = None
        cls._open_col_idx = None
        cls.visual_markers = []
        cls.scales = []
        cls._finger_contact_query_mask = None
        cls._obj_contact_with_mask = None
        cls._finger_contact_result = None
        cls._mask_force_off = None
        cls._mask_active = None
        cls._mask_in_contact = None
        cls._mask_can_toggle = None
        cls._mask_flip = None

    @classmethod
    def initialize_view(cls):
        """
        Rebuild all class-level tensors after scene changes.

        Carries over VALUES and _can_toggle_steps for objects that still exist.
        Rebuilds _requires_closed, contact masks, visual_markers list, and scratch masks.
        """
        # Snapshot all per-object state before super() resets OBJ_IDXS and calls global_initialize()
        # (global_initialize sets visual_markers=[], which is falsy — must snapshot before that)
        prev_obj_idxs = dict(cls.OBJ_IDXS) if cls.OBJ_IDXS is not None else {}
        prev_steps = cls._can_toggle_steps.clone() if cls._can_toggle_steps is not None else None
        prev_markers = list(cls.visual_markers) if cls.visual_markers is not None else []
        prev_scales = list(cls.scales) if cls.scales is not None else []

        # Base class rebuilds OBJ_IDXS, IDX_OBJS, VALUES (with value carry-over for toggle bool)
        super().initialize_view()

        S, N = len(cls.IDX_OBJS), len(cls.OBJ_IDXS)

        # Carry over _can_toggle_steps for surviving objects
        cls._can_toggle_steps = th.zeros((S, N), dtype=th.float32, device="cuda")
        if prev_steps is not None:
            for relative_prim_path, obj_idx_old in prev_obj_idxs.items():
                if relative_prim_path not in cls.OBJ_IDXS:
                    continue
                obj_idx_new = cls.OBJ_IDXS[relative_prim_path]
                for s_idx in range(min(prev_steps.shape[0], S)):
                    cls._can_toggle_steps[s_idx, obj_idx_new] = prev_steps[s_idx, obj_idx_old]

        # Build _requires_closed: (S, N) — read _requires_closed_init directly to avoid
        # going through the property before the tensor is fully populated.
        cls._requires_closed = th.zeros((S, N), dtype=th.bool, device="cuda")
        for s_idx, s_row in enumerate(cls.IDX_OBJS):
            for obj_idx, obj in enumerate(s_row):
                if obj is not None:
                    cls._requires_closed[s_idx, obj_idx] = obj.states[ToggledOn]._requires_closed_init

        # Build _open_col_idx: (N,) long — maps each toggle-object N-index to its column in
        # Open.VALUES.  -1 for objects that have no Open state.
        # Uses scene-0 objects as representative (all scenes share the same object types).
        cls._open_col_idx = th.full((N,), -1, dtype=th.long, device="cuda")
        for obj_idx, obj in enumerate(cls.IDX_OBJS[0] if cls.IDX_OBJS else []):
            if obj is not None and Open in obj.states:
                open_idx = Open.OBJ_IDXS.get(obj.relative_prim_path, -1)
                cls._open_col_idx[obj_idx] = open_idx

        # Rebuild visual_markers and scales lists indexed by N
        # Carry over existing markers/scales for surviving objects; new slots stay None
        cls.visual_markers = [None] * N
        cls.scales = [None] * N
        for relative_prim_path, obj_idx_new in cls.OBJ_IDXS.items():
            obj_idx_old = prev_obj_idxs.get(relative_prim_path)
            if obj_idx_old is not None and obj_idx_old < len(prev_markers):
                cls.visual_markers[obj_idx_new] = prev_markers[obj_idx_old]
                cls.scales[obj_idx_new] = prev_scales[obj_idx_old]

        if S == 0 or N == 0:
            # No objects — allocate empty tensors and return early
            # _finger_contact_query_mask and _obj_contact_with_mask stay CPU (RigidContactAPI interface)
            cls._finger_contact_query_mask = th.zeros((0, 1, 0), dtype=th.bool)
            cls._obj_contact_with_mask = th.zeros((0, 0, 0), dtype=th.bool)
            cls._finger_contact_result = th.zeros((0, 0), dtype=th.bool, device="cuda")
            cls._mask_force_off = th.zeros((0, 0), dtype=th.bool, device="cuda")
            cls._mask_active = th.zeros((0, 0), dtype=th.bool, device="cuda")
            cls._mask_in_contact = th.zeros((0, 0), dtype=th.bool, device="cuda")
            cls._mask_can_toggle = th.zeros((0, 0), dtype=th.bool, device="cuda")
            cls._mask_flip = th.zeros((0, 0), dtype=th.bool, device="cuda")
            return

        # ----- Contact masks via public API (no _* access to RigidContactAPI internals) -----
        # All scenes share identical relative prim paths → compute once from scene 0, broadcast.

        # Finger query mask: (S, 1, R). R is num_rows in RigidContactAPI
        # _finger_row_mask is computed once (robots are fixed); _finger_contact_query_mask is
        # reshaped from it whenever S or R changes.
        if cls._finger_row_mask is None:
            scene_0 = og.sim.scenes[0]
            finger_links = [
                link
                for robot in scene_0.robots
                if robot.is_manipulation
                for links in robot.finger_links.values()
                for link in links
            ]
            if finger_links:
                cls._finger_row_mask = RigidContactAPI.get_contact_row_mask(finger_links)  # (R,)
        if cls._finger_row_mask is not None and (
            cls._finger_contact_query_mask is None or cls._finger_contact_query_mask.shape[0] != S
        ):
            cls._finger_contact_query_mask = cls._finger_row_mask.view(1, 1, -1).expand(S, 1, -1).clone()

        # Object with-mask: (S, N, C) — scene 0 is representative; all scenes share identical objects.
        # Only rebuild when S or N changes (new scene or new toggle object type).
        # C is num_cols in RigidContactAPI
        if cls._obj_contact_with_mask is None or cls._obj_contact_with_mask.shape[:2] != (S, N):
            cls._obj_contact_with_mask = (
                th.stack(
                    [
                        RigidContactAPI.get_contact_col_mask(list(cls.IDX_OBJS[0][obj_idx].links.values()))
                        for obj_idx in range(N)
                    ]
                )
                .unsqueeze(0)
                .expand(S, -1, -1)
                .clone()
            )  # (S, N, C)

        # Allocate per-step scratch masks — GPU for computation; contact query masks stay CPU
        cls._finger_contact_result = th.zeros((S, N), dtype=th.bool, device="cuda")
        cls._mask_force_off = th.zeros((S, N), dtype=th.bool, device="cuda")
        cls._mask_active = th.zeros((S, N), dtype=th.bool, device="cuda")
        cls._mask_in_contact = th.zeros((S, N), dtype=th.bool, device="cuda")
        cls._mask_can_toggle = th.zeros((S, N), dtype=th.bool, device="cuda")
        cls._mask_flip = th.zeros((S, N), dtype=th.bool, device="cuda")

    def __init__(self, obj, scale=None, requires_closed=False):
        # self.scale is a temporary instance variable used only during _initialize() to
        # resolve the final marker scale and store it in cls.scales[obj_idx].
        # It is not accessed after initialization.
        self.scale = scale

        if requires_closed:
            assert Open in obj.states, f"ToggledOn requires_closed=True but {obj.name} has no Open state."

        # Store requires_closed as instance attribute; written into class tensor by initialize_view()
        self._requires_closed_init = requires_closed

        super().__init__(obj)

    @property
    def requires_closed(self):
        """Read-only bool view into the class-level _requires_closed tensor for this object."""
        s = self.obj.scene.idx
        obj_idx = self.OBJ_IDXS[self.obj.relative_prim_path]
        return bool(type(self)._requires_closed[s, obj_idx].item())

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

    @classmethod
    def global_update(cls):
        """
        Step 1 — Batch contact check:
            Query which toggle objects are touched by robot fingers using is_in_contact_batch.
            Pre-computed masks (_finger_contact_query_mask, _obj_contact_with_mask) are reused
            from initialize_view(); no per-step path rebuild.

        Step 2 — Tensorized value update:
            Delegate to TensorizedValueState.global_update(), which calls _update_values() and
            fires state_updated() for any object whose VALUES row changed.
        """
        if cls._finger_contact_query_mask is None or cls._finger_contact_query_mask.numel() == 0:
            return

        # Step 1 - batch contact check: returns (S, N) bool
        cls._finger_contact_result.copy_(
            RigidContactAPI.is_in_contact_batch(
                query_masks=cls._finger_contact_query_mask,  # (S, 1, R)
                with_masks=cls._obj_contact_with_mask,  # (S, N, C)
                ignore_masks=None,
                current_only=False,
            )
        )

        # Step 2 - batch value update (calls _update_values, then fires state_updated)
        super().global_update()

    @classproperty
    def meta_link_types(cls):
        return [m.TOGGLE_META_LINK_TYPE]

    @classmethod
    def _check_overlap(cls, s_idx, obj_idx):
        """
        Check whether any robot finger overlaps the toggle-button marker sphere for the object
        at class-level index (s_idx, obj_idx).

        Args:
            s_idx (int): Scene index.
            obj_idx (int): Object type index into cls.IDX_OBJS / cls.visual_markers / cls.scales.

        Returns:
            bool: True if a robot finger overlaps the marker sphere.
        """
        valid_hit = False

        # Collect all robot finger prim paths across all robots in the scene
        scene = og.sim.scenes[s_idx]
        all_finger_paths = {
            link.prim_path
            for robot in scene.robots
            if robot.is_manipulation
            for finger_links in robot.finger_links.values()
            for link in finger_links
        }

        def overlap_callback(hit):
            nonlocal valid_hit
            valid_hit = hit.rigid_body in all_finger_paths
            # Continue traversal only if we don't have a valid hit yet
            return not valid_hit

        marker = cls.visual_markers[obj_idx]
        scale = cls.scales[obj_idx]
        # TODO: This is a temporary fix for flatcache before we properly implement trigger volumes
        if marker is None:
            return False
        og.sim.psqi.overlap_sphere(
            radius=th.min(marker.extent * scale).item(),
            pos=marker.get_position_orientation()[0].tolist(),
            reportFn=overlap_callback,
        )
        return valid_hit

    @classmethod
    def _update_values(cls, values):
        """
        Vectorized per-step update for all tracked ToggledOn instances across all scenes.

        Uses pre-allocated named scratch masks (_mask_force_off, _mask_active, etc.) to avoid
        per-call allocations. Contact result (_finger_contact_result) was already computed by
        global_update() before this method is called.

        Logic:
          1. requires_closed guard  — force off objects that must be closed but are open.
          2. Contact mask           — which active objects have a finger nearby (from is_in_contact_batch).
          3. Overlap check          — of those, which truly overlap the marker sphere.
          4. Counter update         — increment for can_toggle, reset for active & ~can_toggle.
          5. Toggle flip            — flip state for objects whose counter hit CAN_TOGGLE_STEPS.
          6. Color sync             — update visual marker color for flipped objects.

        Args:
            values (th.Tensor): Shape (S, N). Toggle state stored as 0.0/1.0.

        Returns:
            th.Tensor: Updated values tensor (clone of input with modifications applied).
        """
        # Single clone — required so the base class can detect changes via new_values != cls.VALUES.
        new_values = values.clone()
        S = values.shape[0]

        # Reset all scratch masks for this step (in-place fill, no allocation).
        cls._mask_force_off.fill_(False)
        cls._mask_active.fill_(False)
        cls._mask_in_contact.fill_(False)
        cls._mask_can_toggle.fill_(False)
        cls._mask_flip.fill_(False)

        # Step 1: requires_closed guard — vectorized via Open.VALUES.
        # _open_col_idx (N,): column index into Open.VALUES for each toggle object (-1 = no Open state).
        if cls._requires_closed.any():
            valid = cls._requires_closed & (cls._open_col_idx >= 0).unsqueeze(0)  # (S, N)
            if valid.any():
                safe_idxs = cls._open_col_idx.clamp(min=0)  # (N,) — guard -1 before indexing
                open_vals = Open.VALUES[:, safe_idxs] > 0.5  # (S, N) bool
                cls._mask_force_off.copy_(valid & open_vals)

        th.logical_not(cls._mask_force_off, out=cls._mask_active)
        # Zero out toggle state for force-off objects (in-place).
        new_values[cls._mask_force_off] = 0.0
        cls._can_toggle_steps[cls._mask_force_off] = 0.0

        # Step 2: contact mask.
        # _finger_contact_result was populated in global_update() via is_in_contact_batch.
        cls._mask_in_contact.copy_(cls._finger_contact_result)
        # Only active objects proceed to overlap check.
        cls._mask_in_contact &= cls._mask_active

        # Step 3: overlap check (unavoidable per-object; only for in-contact subset).
        for scene_idx in range(S):
            for obj_idx in th.where(cls._mask_in_contact[scene_idx])[0].tolist():
                cls._mask_can_toggle[scene_idx, obj_idx] = cls._check_overlap(scene_idx, obj_idx)

        # Step 4: step counter update.
        # Increment counter for can_toggle objects (can_toggle ⊆ in_contact ⊆ active).
        cls._can_toggle_steps[cls._mask_can_toggle] += 1
        # Reset counter for active objects that cannot toggle.
        active_no_toggle = cls._mask_active & ~cls._mask_can_toggle
        cls._can_toggle_steps[active_no_toggle] = 0.0

        # Step 5: toggle flip.
        th.eq(cls._can_toggle_steps, m.CAN_TOGGLE_STEPS, out=cls._mask_flip)
        cls._mask_flip &= cls._mask_active
        new_values[cls._mask_flip] = 1.0 - values[cls._mask_flip]

        # Step 6: sync visual marker colors for flipped objects.
        # Per-object USD color write; unavoidable loop, runs only for the flipped subset.
        # Uses class-level COLOR_ON / COLOR_OFF constants — no per-flip tensor allocation.
        for scene_idx in range(S):
            for obj_idx in th.where(cls._mask_flip[scene_idx])[0].tolist():
                cls.visual_markers[obj_idx].color = (
                    cls.COLOR_ON if bool(new_values[scene_idx, obj_idx].item()) else cls.COLOR_OFF
                )

        return new_values

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
        # _set_value is possible to be called before _initialize() so maker might not be initialized yet
        marker = type(self).visual_markers[obj_idx]
        if marker is not None:
            marker.color = type(self).COLOR_ON if new_value else type(self).COLOR_OFF
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
            lazy.isaacsim.core.utils.bounds.recompute_extents(prim=pre_existing_mesh)
            self.scale = vtarray_to_torch(pre_existing_mesh.GetAttribute("xformOp:scale").Get())

        # Create the visual geom instance referencing the generated mesh prim
        relative_prim_path = absolute_prim_path_to_scene_relative(self.obj.scene, mesh_prim_path)
        marker = GeomPrim(relative_prim_path=relative_prim_path, name=f"{self.obj.name}_visual_marker")
        marker.load(self.obj.scene)
        marker.scale = self.scale
        marker.initialize()
        marker.visible = True

        # Store marker and resolved scale in the class-level lists at this object's N index.
        # Both are accessed by _check_overlap() and _update_values() at the class level.
        obj_idx = self.OBJ_IDXS.get(self.obj.relative_prim_path)
        assert obj_idx is not None
        type(self).visual_markers[obj_idx] = marker
        type(self).scales[obj_idx] = self.scale
        marker.color = type(self).COLOR_OFF

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

    # For this state, we store the toggle value and the robot_can_toggle_steps counter.
    def _dump_state(self):
        s = self.obj.scene.idx
        obj_idx = self.OBJ_IDXS[self.obj.relative_prim_path]
        return dict(
            value=bool(self.VALUES[s, obj_idx].item()),
            hand_in_marker_steps=int(self._can_toggle_steps[s, obj_idx].item()),
        )

    def _load_state(self, state):
        # Restore toggle via _set_value so the visual marker color is also updated.
        self._set_value(state["value"])
        # Restore step counter directly into the tensor.
        s = self.obj.scene.idx
        obj_idx = self.OBJ_IDXS[self.obj.relative_prim_path]
        type(self)._can_toggle_steps[s, obj_idx] = float(state["hand_in_marker_steps"])

    def serialize(self, state):
        # [toggle_state, can_toggle_steps] as float32
        return th.tensor([state["value"], state["hand_in_marker_steps"]], dtype=th.float32)

    def deserialize(self, state):
        return dict(value=bool(state[0].item()), hand_in_marker_steps=int(state[1].item())), 2
