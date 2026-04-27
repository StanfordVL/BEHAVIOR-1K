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
    Boolean state representing whether an object has been toggled on.
    """

    # S = number of scenes
    # O = number of toggleable objects

    # th.Tensor (S, O) int
    _robots_can_toggle_steps = None

    # (O_requires_closed,) int64
    # These are indices into flattened views of Open.VALUES and ToggledOn.VALUES.
    _requires_closed_obj_idxes_in_open_values = None
    _requires_closed_obj_idxes_in_this_values = None

    # list[list[GeomPrim]]: visual toggle-button markers, one per tracked object. Shape (S, O).
    # Used in _check_overlap and color updates.
    visual_markers = None

    # Contact masks
    # R_s = number of contact-matrix rows (links on the "who is touching" side) for scene s
    # C_s = number of contact-matrix columns (links on the "what are they touching" side) for scene s
    _finger_query_mask = None  # list[Tensor(1, R_s) | None] — finger row mask per scene
    _toggable_objs_with_mask = None  #  list[Tensor(O, C_s) | None] — toggle-object col masks per scene

    # list[list of links of manipulation robots in scene s], len = S
    _finger_links = []

    # Scratch masks, helper for calculate update, (S, O) bool
    _mask_can_toggle = None

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

        # Build indices for requires_closed logic.
        requires_closed_obj_idxes_in_open_values = []
        requires_closed_obj_idxes_in_this_values = []
        for scene_idx, scene in enumerate(cls.IDX_OBJS):
            for obj_idx, toggle_obj in enumerate(scene):
                if toggle_obj is None:
                    continue

                if not toggle_obj.states[ToggledOn].requires_closed:
                    continue

                # Compute the index of this object in the flattened view of this state's VALUES.
                requires_closed_obj_idxes_in_this_values.append(scene_idx * O + obj_idx)
                # Compute the index of this object in the flattened view of Open.VALUES.
                idx_in_open_object_dim = Open.OBJ_IDXS[toggle_obj.relative_prim_path]
                open_values_object_dim_size = Open.VALUES.shape[1]
                requires_closed_obj_idxes_in_open_values.append(
                    scene_idx * open_values_object_dim_size + idx_in_open_object_dim
                )
        cls._requires_closed_obj_idxes_in_open_values = th.tensor(
            requires_closed_obj_idxes_in_open_values, dtype=th.long, device="cuda"
        )
        cls._requires_closed_obj_idxes_in_this_values = th.tensor(
            requires_closed_obj_idxes_in_this_values, dtype=th.long, device="cuda"
        )

        # Build visual_markers: point to each instance's self.marker set during _initialize().
        cls.visual_markers = [[None] * O for _ in range(S)]
        for scene_idx, scene_row in enumerate(cls.IDX_OBJS):
            for obj_idx, toggle_obj in enumerate(scene_row):
                if toggle_obj is None:
                    continue
                state = toggle_obj.states[ToggledOn]
                cls.visual_markers[scene_idx][obj_idx] = state.marker

        if S == 0 or O == 0:
            # No objects — allocate empty lists/tensors and return early
            cls._finger_query_mask = []
            cls._toggable_objs_with_mask = []
            cls._mask_can_toggle = th.zeros((0, 0), dtype=th.bool, device="cuda")
            return

        # Loop over scenes to build contact query_masks and with_masks to help detect whether finger contacting any togglable objects
        cls._finger_query_mask = []
        cls._toggable_objs_with_mask = []

        for scene_idx, scene in enumerate(og.sim.scenes):
            finger_links = [
                link
                for robot in scene.robots
                if robot.is_manipulation
                for links in robot.finger_links.values()
                for link in links
            ]
            cls._finger_links.append(finger_links)
            if not finger_links:
                cls._finger_query_mask.append(None)
                cls._toggable_objs_with_mask.append(None)
                continue

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

        # Allocate per-step scratch masks — GPU for computation; contact query masks stay CPU
        cls._mask_can_toggle = th.zeros((S, O), dtype=th.bool, device="cuda")

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
        Vectorized per-step update for all tracked ToggledOn instances across all scenes.

        Steps:
        - For objects that are open yet required to be closed to toggle on, VALUE = False, robot_can_toggle_step = 0
        - Find what toggleable objects have a finger nearby. For those who have, check fingers truly overlap the button mesh.
        - Increment robot_can_toggle_steps for objects that can be toggled.
        - Set VALUES to be True for whose robot_can_toggle_steps == m.CAN_TOGGLE_STEPS.

        Args:
            values (th.Tensor): Shape (S, O). Toggle state stored as 0.0/1.0. Mutated in-place.
        """
        S = values.shape[0]

        # Get what toggleable objects are being touched by a finger
        cls._mask_can_toggle.fill_(False)
        for scene_idx, (query_mask, with_mask) in enumerate(zip(cls._finger_query_mask, cls._toggable_objs_with_mask)):
            if query_mask is None or with_mask is None:
                continue
            result = RigidContactAPI.is_in_contact_batch(
                scene_idx=scene_idx,
                query_masks=query_mask,  # (1, R_s)
                with_masks=with_mask,  # (O, C_s)
                ignore_masks=None,
                current_only=False,
            )  # (O,) bool
            cls._mask_can_toggle[scene_idx].copy_(result)

        # For objects that are open yet required to be closed to toggle on, set VALUE = False and robot_can_toggle_step = 0
        flattened_open_values = Open.VALUES.view(-1)
        flattened_this_values = values.view(-1)

        # Get the values of the objects that are open yet required to be closed to toggle on
        requires_closed_obj_open_values = flattened_open_values[cls._requires_closed_obj_idxes_in_open_values]

        # Get the indices of the objects that are open yet required to be closed to toggle on
        requires_closed_obj_idxes_that_are_open = cls._requires_closed_obj_idxes_in_this_values[
            requires_closed_obj_open_values
        ]

        # Set the values of the objects that are open yet required to be closed to toggle on to False and reset the robot_can_toggle_steps to 0
        # and also set the mask_can_toggle to False for these objects - they cannot be toggled on this step either!
        flattened_this_values[requires_closed_obj_idxes_that_are_open] = False
        cls._robots_can_toggle_steps.view(-1)[requires_closed_obj_idxes_that_are_open] = 0
        cls._mask_can_toggle.view(-1)[requires_closed_obj_idxes_that_are_open] = False

        # Find what toggleable objects have a finger nearby. For those who have, check fingers truly overlap the button mesh.
        for scene_idx in range(S):
            for obj_idx in th.where(cls._mask_can_toggle[scene_idx])[0].tolist():
                cls._mask_can_toggle[scene_idx, obj_idx] &= cls._check_overlap(scene_idx, obj_idx)

        # Update robot_can_toggle_steps
        cls._robots_can_toggle_steps[cls._mask_can_toggle] += 1
        cls._robots_can_toggle_steps[~cls._mask_can_toggle] = 0

        # Only objects that have robot_can_toggle_steps == m.CAN_TOGGLE_STEPS can be toggled on
        cls._mask_can_toggle[cls._robots_can_toggle_steps != m.CAN_TOGGLE_STEPS] = False

        # Flip values
        th.logical_xor(values, cls._mask_can_toggle, out=values)

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
