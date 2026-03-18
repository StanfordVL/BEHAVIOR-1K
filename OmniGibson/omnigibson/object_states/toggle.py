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
from omnigibson.utils.python_utils import classproperty, torch_delete
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
    tensors and lists (same index as VALUES):

        VALUES[:, 0]     — toggle on/off (float32, 0.0 = off / 1.0 = on)
        VALUES[:, 1]     — robot_can_toggle_steps counter (float32, used as int)
        _requires_closed — bool tensor (N,); construction-time constant
        visual_markers   — list of GeomPrim; toggle-button visual indicator per object
        scales           — list of float or Tensor; marker scale per object

    Per-step scratch buffer
    -----------------------
    _scratch_masks is a pre-allocated (N, 6) bool tensor reused every step to avoid
    per-call mask allocations. See column layout in the class body.

    Per-instance state (not in class-level arrays):
        self.scale      — temporary storage of constructor arg; used only during _initialize()
                          to populate cls.scales[idx], not accessed at runtime
    """

    # list[set[str]]: robot finger prim paths per scene; refreshed every step in global_update
    _robot_finger_paths = None

    # set[StatefulObject]: objects in contact with any robot finger; refreshed every step
    _finger_contact_objs = None

    # th.Tensor shape (N,) bool: whether each tracked object requires closed state to toggle on.
    # Construction-time constant; same indexing as VALUES.
    # Named with underscore to avoid collision with the @property requires_closed below.
    _requires_closed = None

    # list[GeomPrim]: visual toggle-button markers, one per tracked object (same idx as VALUES).
    # Populated during _initialize(); used in _check_overlap and color updates.
    visual_markers = None

    # list[float | th.Tensor]: per-object marker scale (same idx as VALUES).
    # Populated during _initialize(); used in _check_overlap for sphere radius.
    scales = None

    # th.Tensor shape (N, 6) bool: pre-allocated scratch buffer for _update_values.
    # Resized when objects are added / removed; never reallocated mid-step.
    # Column layout (fixed):
    #   0  COL_OPEN        open_values (reused as active & ~can_toggle in step 4)
    #   1  COL_FORCE_OFF   _requires_closed & open
    #   2  COL_ACTIVE      ~force_off
    #   3  COL_IN_CONTACT  in contact with finger & active
    #   4  COL_CAN_TOGGLE  sphere overlap confirmed & in_contact
    #   5  COL_FLIP        counter == CAN_TOGGLE_STEPS & active
    _scratch_masks = None

    # Class-level color constants: defined once, reused every flip and _set_value call.
    # Avoids allocating new th.tensor([...]) on every color update.
    COLOR_ON = th.tensor([0, 1.0, 0])  # green  — toggle is on
    COLOR_OFF = th.tensor([1.0, 0, 0])  # red    — toggle is off

    @classproperty
    def value_shape(cls):
        # Two floats per object: [toggle_state, robot_can_toggle_steps]
        return (2,)

    @classproperty
    def value_type(cls):
        # float32 for both fields; toggle_state is 0.0/1.0 (bool semantics),
        # robot_can_toggle_steps is an integer stored as float.
        return th.float32

    @classproperty
    def value_name(cls):
        # Used by the parent's _dump_state / _load_state for the raw tensor column.
        # Individual fields are accessed by column index, not by this key directly.
        return "toggle_state"

    @classmethod
    def global_initialize(cls):
        """
        Initialize all class-level state for ToggledOn. Called once at simulator start
        and on each clear/reset. Allocates VALUES (via super) and all auxiliary structures.
        """
        # Allocate VALUES, OBJ_IDXS, IDX_OBJS, CALLBACKS_ON_REMOVE, STATE_SIZE
        super().global_initialize()

        cls._robot_finger_paths = []
        cls._finger_contact_objs = set()
        cls._requires_closed = th.empty(0, dtype=th.bool)
        cls.visual_markers = []
        cls.scales = []
        # Start with zero rows; resized in _add_obj / _remove_obj as objects arrive.
        cls._scratch_masks = th.zeros((0, 6), dtype=th.bool)

    @classmethod
    def _add_obj(cls, obj):
        # Append a row to VALUES and update OBJ_IDXS / IDX_OBJS via parent.
        super()._add_obj(obj=obj)
        # _requires_closed placeholder (False); overwritten in __init__ immediately after.
        cls._requires_closed = th.cat([cls._requires_closed, th.tensor([False])])
        # visual_markers and scales are populated in _initialize(); use None as placeholder.
        cls.visual_markers.append(None)
        cls.scales.append(None)
        # Grow scratch buffer by one zero row to match new N.
        cls._scratch_masks = th.cat([cls._scratch_masks, th.zeros((1, 6), dtype=th.bool)], dim=0)

    @classmethod
    def _remove_obj(cls, obj):
        """Remove all class-level entries for obj before delegating to the parent."""
        # Always read deleted_idx first, before any list/tensor deletions and before super().
        deleted_idx = cls.OBJ_IDXS[obj]
        cls._requires_closed = torch_delete(cls._requires_closed, [deleted_idx])
        del cls.visual_markers[deleted_idx]
        del cls.scales[deleted_idx]
        cls._scratch_masks = torch_delete(cls._scratch_masks, [deleted_idx])
        # Parent handles VALUES, OBJ_IDXS, IDX_OBJS cleanup.
        super()._remove_obj(obj=obj)

    def __init__(self, obj, scale=None, requires_closed=False):
        # self.scale is a temporary instance variable used only during _initialize() to
        # resolve the final marker scale and store it in cls.scales[idx].
        # It is not accessed after initialization.
        self.scale = scale

        if requires_closed:
            assert Open in obj.states, f"ToggledOn requires_closed=True but {obj.name} has no Open state."

        # super().__init__ calls _add_obj, which appends rows to VALUES, _requires_closed,
        # visual_markers, scales, and _scratch_masks.
        super().__init__(obj)

        # Write the constructor flag into the class-level bool tensor at this object's index.
        # Done after super().__init__ so that OBJ_IDXS[obj] is already populated.
        type(self)._requires_closed[self.OBJ_IDXS[obj]] = requires_closed

    @property
    def requires_closed(self):
        """Read-only bool view into the class-level _requires_closed tensor for this object."""
        return bool(type(self)._requires_closed[self.OBJ_IDXS[self.obj]].item())

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
        Step 1 — Contact collection:
            Refresh _robot_finger_paths and _finger_contact_objs from RigidContactAPI once per step.
            This mirrors the original global_update() behavior.

        Step 2 — Tensorized value update:
            Delegate to TensorizedValueState.global_update(), which calls _update_values() and
            fires state_updated() for any object whose VALUES row changed.
        """
        # Step 1 - refresh finger contact data
        cls._finger_contact_objs = set()

        cls._robot_finger_paths = [
            {
                link.prim_path
                for robot in scene.robots
                if robot.is_manipulation
                for finger_links in robot.finger_links.values()
                for link in finger_links
            }
            for scene in og.sim.scenes
        ]

        # Only query contacts if at least one manipulation robot is present.
        if any(len(paths) > 0 for paths in cls._robot_finger_paths):
            for scene_idx, (scene, scene_finger_paths) in enumerate(zip(og.sim.scenes, cls._robot_finger_paths)):
                if len(scene_finger_paths) == 0:
                    continue

                # Get the robot finger prim paths' contacts
                contact_pairs = RigidContactAPI.get_contact_pairs(scene_idx, scene_finger_paths)
                for contact_pair in contact_pairs:
                    obj = scene.object_registry("prim_path", "/".join(contact_pair[1].split("/")[:-1]))
                    if obj is not None:
                        cls._finger_contact_objs.add(obj)

        # Step 2 - batch value update (calls _update_values, then fires state_updated)
        super().global_update()

    @classproperty
    def meta_link_types(cls):
        return [m.TOGGLE_META_LINK_TYPE]

    @classmethod
    def _check_overlap(cls, idx):
        """
        Check whether any robot finger overlaps the toggle-button marker sphere for the object
        at class-level index idx.

        Args:
            idx (int): Index into cls.IDX_OBJS / cls.visual_markers / cls.scales.

        Returns:
            bool: True if a robot finger overlaps the marker sphere.
        """
        valid_hit = False
        all_finger_paths = {path for path_set in cls._robot_finger_paths for path in path_set}

        def overlap_callback(hit):
            nonlocal valid_hit
            valid_hit = hit.rigid_body in all_finger_paths
            # Continue traversal only if we don't have a valid hit yet
            return not valid_hit

        marker = cls.visual_markers[idx]
        scale = cls.scales[idx]
        # TODO: This is a temporary fix for flatcache before we properly implement trigger volumes
        og.sim.psqi.overlap_sphere(
            radius=th.min(marker.extent * scale).item(),
            pos=marker.get_position_orientation()[0].tolist(),
            reportFn=overlap_callback,
        )
        # if marker.prim.GetTypeName() == "Mesh":
        #     og.sim.psqi.overlap_mesh(*projection_mesh_ids, reportFn=overlap_callback)
        # else:
        #     og.sim.psqi.overlap_shape(*projection_mesh_ids, reportFn=overlap_callback)
        return valid_hit

    @classmethod
    def _update_values(cls, values):
        """
        Vectorized per-step update for all tracked ToggledOn instances. Mimics the original
        per-object _update() logic using tensor operations and a pre-allocated scratch buffer.

        Scratch buffer column layout (see class variables):
            COL_OPEN=0  COL_FORCE_OFF=1  COL_ACTIVE=2
            COL_IN_CONTACT=3  COL_CAN_TOGGLE=4  COL_FLIP=5
        Column 0 (COL_OPEN) is reused in step 4 as "active & ~can_toggle" temp once step 1 is done.

        Logic:
          1. requires_closed guard  — force off objects that must be closed but are open.
          2. Contact mask           — which active objects have a finger nearby.
          3. Overlap check          — of those, which truly overlap the marker sphere.
          4. Counter update         — increment for can_toggle, reset for active & ~can_toggle.
          5. Toggle flip            — flip state for objects whose counter hit CAN_TOGGLE_STEPS.
          6. Color sync             — update visual marker color for flipped objects.

        Args:
            values (th.Tensor): Shape (N, 2). col 0 = toggle state, col 1 = step counter.

        Returns:
            th.Tensor: Updated values tensor (clone of input with modifications applied).
        """
        # Single clone — required so the base class can detect changes via new_values != cls.VALUES.
        new_values = values.clone()

        # Column index constants (defined locally for readability; no object creation overhead).
        COL_OPEN, COL_FORCE_OFF, COL_ACTIVE, COL_IN_CONTACT, COL_CAN_TOGGLE, COL_FLIP = range(6)

        # Reset all scratch columns for this step (in-place fill, no allocation).
        cls._scratch_masks.fill_(False)

        # Step 1: requires_closed guard.
        # Open is not tensorized; loop only over the _requires_closed subset.
        for i in th.where(cls._requires_closed)[0].tolist():
            cls._scratch_masks[i, COL_OPEN] = cls.IDX_OBJS[i].states[Open].get_value()

        th.logical_and(cls._requires_closed, cls._scratch_masks[:, COL_OPEN], out=cls._scratch_masks[:, COL_FORCE_OFF])
        th.logical_not(cls._scratch_masks[:, COL_FORCE_OFF], out=cls._scratch_masks[:, COL_ACTIVE])
        # Zero out both toggle state and step counter for force-off objects (in-place).
        new_values[cls._scratch_masks[:, COL_FORCE_OFF]] = 0.0

        # Step 2: contact mask.
        for i, obj in enumerate(cls.IDX_OBJS):
            cls._scratch_masks[i, COL_IN_CONTACT] = obj in cls._finger_contact_objs
        # Only active objects proceed to overlap check.
        cls._scratch_masks[:, COL_IN_CONTACT] &= cls._scratch_masks[:, COL_ACTIVE]

        # Step 3: overlap check (unavoidable per-object; only for in-contact subset).
        for i in th.where(cls._scratch_masks[:, COL_IN_CONTACT])[0].tolist():
            cls._scratch_masks[i, COL_CAN_TOGGLE] = cls._check_overlap(i)

        # Step 4: step counter update.
        # Increment counter for can_toggle objects (can_toggle ⊆ in_contact ⊆ active).
        new_values[cls._scratch_masks[:, COL_CAN_TOGGLE], 1] += 1
        # Reset counter for active objects that cannot toggle.
        # Reuse COL_OPEN (step 1 data no longer needed) to avoid allocating a new mask tensor.
        th.logical_and(
            cls._scratch_masks[:, COL_ACTIVE],
            ~cls._scratch_masks[:, COL_CAN_TOGGLE],
            out=cls._scratch_masks[:, COL_OPEN],
        )
        new_values[cls._scratch_masks[:, COL_OPEN], 1] = 0.0

        # Step 5: toggle flip.
        th.eq(new_values[:, 1], m.CAN_TOGGLE_STEPS, out=cls._scratch_masks[:, COL_FLIP])
        cls._scratch_masks[:, COL_FLIP] &= cls._scratch_masks[:, COL_ACTIVE]
        new_values[cls._scratch_masks[:, COL_FLIP], 0] = 1.0 - values[cls._scratch_masks[:, COL_FLIP], 0]

        # Step 6: sync visual marker colors for flipped objects.
        # Per-object USD color write; unavoidable loop, runs only for the flipped subset.
        # Uses class-level COLOR_ON / COLOR_OFF constants — no per-flip tensor allocation.
        for i in th.where(cls._scratch_masks[:, COL_FLIP])[0].tolist():
            cls.visual_markers[i].color = cls.COLOR_ON if bool(new_values[i, 0].item()) else cls.COLOR_OFF

        return new_values

    def _get_value(self):
        # Return toggle boolean from column 0 of the shared VALUES tensor.
        return bool(self.VALUES[self.OBJ_IDXS[self.obj], 0].item())

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

        idx = self.OBJ_IDXS[self.obj]
        self.VALUES[idx, 0] = 1.0 if new_value else 0.0
        # Green = on, red = off. Use class-level constants to avoid per-call tensor allocation.
        type(self).visual_markers[idx].color = type(self).COLOR_ON if new_value else type(self).COLOR_OFF
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

        # Store the projection mesh's IDs
        # projection_mesh_ids = lazy.pxr.PhysicsSchemaTools.encodeSdfPath(marker.prim_path)

        # Store marker and resolved scale in the class-level lists at this object's index.
        # Both are accessed by _check_overlap() and _update_values() at the class level.
        idx = self.OBJ_IDXS[self.obj]
        type(self).visual_markers[idx] = marker
        type(self).scales[idx] = self.scale

        # VALUES[idx, 0] was initialized to 0.0 (off) by _add_obj(). Sync the marker color.
        # This replaces the former self._set_value(False) call.
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

    # For this state, we simply store its value and the robot_can_toggle steps.
    def _dump_state(self):
        idx = self.OBJ_IDXS[self.obj]
        return dict(
            value=bool(self.VALUES[idx, 0].item()),
            hand_in_marker_steps=int(self.VALUES[idx, 1].item()),
        )

    def _load_state(self, state):
        # Restore toggle via _set_value so the visual marker color is also updated.
        self._set_value(state["value"])
        # Restore step counter directly into the tensor.
        self.VALUES[self.OBJ_IDXS[self.obj], 1] = float(state["hand_in_marker_steps"])

    def serialize(self, state):
        # [toggle_state, can_toggle_steps] as float32
        return th.tensor([state["value"], state["hand_in_marker_steps"]], dtype=th.float32)

    def deserialize(self, state):
        return dict(value=bool(state[0].item()), hand_in_marker_steps=int(state[1].item())), 2
