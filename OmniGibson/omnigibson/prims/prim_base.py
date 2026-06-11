import math
import string
from abc import ABC
from collections.abc import Iterable
from typing import Literal

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.utils.python_utils import Recreatable, Serializable
from omnigibson.utils.transform_utils import quat2euler
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.usd_utils import (
    activate_prim_and_children,
    delete_or_deactivate_prim,
    ensure_usd_api,
    get_local_pose,
    get_sdf_value_type_name,
    get_world_pose,
    get_world_pose_with_scale,
    scene_relative_prim_path_to_absolute,
)

# Create module logger
log = create_module_logger(module_name=__name__)


class BasePrim(Serializable, Recreatable, ABC):
    """
    Provides high level functions to deal with a prim and its attributes/properties, including its pose, scale and
    bound material. If there is an Xform prim present at the path, it will use it. Otherwise, a new Xform prim at
    the specified prim path will be created when self.load(...) is called.

    All pose / scale / material getters are available on every instance of this class. The corresponding setters only
    work if the relevant xform properties ("xformOp:translate", "xformOp:orient", "xformOp:scale") already exist on the
    prim. By default these xform properties are added to the prim when it is loaded, unless @_dont_add_xform_props is
    set (e.g. for prims that are not Xforms, or whose properties already exist on the USD).

    Note: the prim will have "xformOp:orient", "xformOp:translate" and "xformOp:scale" only post init,
        unless it is a non-root articulation link.

    Args:
        relative_prim_path (str): Scene-local prim path of the Prim to encapsulate or create.
        name (str): Name for the object. Names need to be unique per scene.
        load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
            loading this prim at runtime. Note that this is only needed if the prim does not already exist at
            @relative_prim_path -- it will be ignored if it already exists. Subclasses should define the exact keys
            expected for their class. For this base prim, the below values can be specified:

            scale (None or float or 3-array): If specified, sets the scale for this object. A single number corresponds
                to uniform scaling along the x,y,z axes, whereas a 3-array specifies per-axis scaling.
            dont_add_xform_props (bool): If set, the xform properties will not be added to this prim when it is loaded.
    """

    def __init__(
        self,
        relative_prim_path,
        name,
        load_config=None,
    ):
        # Other values that will be filled in at runtime
        self._material = None
        self.original_scale = None
        self._cached_scale = None

        self._relative_prim_path = relative_prim_path
        assert relative_prim_path.startswith("/"), f"Relative prim path {relative_prim_path} must start with a '/'!"
        assert all(
            component[0] in string.ascii_letters for component in relative_prim_path[1:].split("/")
        ), f"Each component of relative prim path {relative_prim_path} must start with a letter!"

        self._name = name
        self._load_config = dict() if load_config is None else load_config

        # Other values that will be filled in at runtime
        self._scene = None
        self._scene_assigned = False
        self._applied_visual_material = None
        self._loaded = False  # Whether this prim exists in the stage or not
        self._initialized = False  # Whether this prim has its internal handles / info initialized or not (occurs AFTER and INDEPENDENTLY from loading!)
        self._prim = None
        self._n_duplicates = 0  # Simple counter for keeping track of duplicates for unique name indexing

        # Whether or not xform properties should be added to this prim. This is necessary for every xform prim,
        # but many of them from our dataset USDs already have these properties, and readding them is very slow, so
        # we skip it for those. This can also be used to skip adding xform properties onto prims for arbitrary other
        # reasons (e.g. instanceable prims, or prims that are not Xforms at all such as joints or materials).
        # Subclasses that are not Xforms should set "dont_add_xform_props" to True in @load_config.
        self._dont_add_xform_props = self._load_config.get("dont_add_xform_props", False)
        # Run super init
        super().__init__()

    def _initialize(self):
        """
        Initializes state of this object and sets up any references necessary post-loading. Should be implemented by
        sub-class for extended utility
        """
        pass

    def initialize(self):
        """
        Initializes state of this object and sets up any references necessary post-loading. Subclasses should
        implement / extend the _initialize() method.
        """
        assert (
            not self._initialized
        ), f"Prim {self.name} at prim_path {self.prim_path} can only be initialized once! (It is already initialized)"
        self._initialize()

        self._initialized = True

        # Cache state size (note that we are doing this after initialized is set to True because
        # dump_state asserts that the prim is initialized for some prims).
        self._state_size = len(self.dump_state(serialized=True))

    def load(self, scene):
        """
        Load this prim into omniverse, and return loaded prim reference.

        Returns:
            Usd.Prim: Prim object loaded into the simulator
        """
        # Load the prim if it doesn't exist yet.
        assert (
            not self._loaded
        ), f"Prim {self.name} at prim_path {self.prim_path} can only be loaded once! (It is already loaded)"

        # Assign the scene first.
        self._scene = scene
        self._scene_assigned = True

        # Check if the prim path exists in the stage
        if lazy.isaacsim.core.utils.prims.is_prim_path_valid(prim_path=self.prim_path):
            existing_prim = lazy.isaacsim.core.utils.prims.get_prim_at_path(prim_path=self.prim_path)

            # Note: A prim path can be valid but the prim itself may be inactive.
            # This commonly occurs after transition rules when scene prims get deleted -
            # the prim paths remain valid but the prims become inactive.
            # In such cases, we need to activate the prim to make it usable again.
            if not existing_prim.IsActive():
                log.debug(f"Prim '{self.name}' exists but is inactive/invalid, activating it")
                # Recursively find all prims under it, even those that are deactivated
                activate_prim_and_children(self.prim_path)
            self._prim = existing_prim
        else:
            # Prim path doesn't exist - load it for the first time
            log.debug(f"Prim '{self.name}' doesn't exist, loading")
            self._prim = self._load()

        # Mark the prim as loaded.
        self._loaded = True

        # Run any post-loading logic
        self._post_load()

        return self._prim

    def _post_load(self):
        """
        Any actions that should be taken (e.g.: modifying the object's properties such as scale, visibility, additional
        joints, etc.) that should be taken after loading the raw object into omniverse but BEFORE we initialize the
        object and grab its handles and internal references.
        """
        # Avoid a circular import
        from omnigibson.prims.material_prim import MaterialPrim

        # Make sure all xforms have pose and scaling info
        # These only need to be done if we are creating this prim from scratch AND it is not an instanceable / proxy prim.
        # Pre-created OG objects' prims always have these things set up ahead of time.
        # Note that if this is an instanceable prim, we also don't need write these properties
        if not self._dont_add_xform_props:
            self._set_xform_properties()

        # Cache the original scale from the USD so that when EntityPrim sets the scale for each link (Rigid/ClothPrim),
        # the new scale is with respect to the original scale. BasePrim's scale always matches the scale in the USD.
        original_scale = self.get_attribute("xformOp:scale")
        if original_scale is None:
            original_scale = [1.0, 1.0, 1.0]
        self.original_scale = th.tensor(original_scale, dtype=th.float32)

        # Grab the attached material if it exists
        if self.has_material():
            material_prim_path = self._binding_api.GetDirectBinding().GetMaterialPath().pathString
            material_name = f"{self.name}:material"
            material = MaterialPrim.get_material(scene=self.scene, prim_path=material_prim_path, name=material_name)
            assert material.loaded, f"Material prim path {material_prim_path} doesn't exist on stage."
            material.add_user(self)
            self._material = material

        # Optionally set the scale, which is specified with respect to the original scale
        if "scale" in self._load_config and self._load_config["scale"] is not None:
            self.scale = self._load_config["scale"] * self.original_scale

    def remove(self):
        """
        Removes this prim from omniverse stage.
        """
        if not self._loaded:
            raise ValueError("Cannot remove a prim that was never loaded.")

        # Remove the material prim if one exists
        if self._material is not None:
            self._material.remove_user(self)

        # Remove or deactivate prim if it's possible
        if not delete_or_deactivate_prim(self.prim_path):
            log.warning(f"Prim {self.name} at prim_path {self.prim_path} could not be deleted or deactivated.")

    def _load(self):
        """
        Loads the raw prim into the simulator. Any post-processing should be done in @self._post_load()
        """
        with og.sim.editing_usd():
            return og.sim.stage.DefinePrim(self.prim_path, "Xform")

    @property
    def loaded(self):
        return self._loaded

    @property
    def initialized(self):
        return self._initialized

    @property
    def scene(self):
        """
        Returns:
            Scene or None: Scene object that this prim is loaded into
        """
        assert self._scene_assigned, "Scene has not been assigned to this prim yet!"
        return self._scene

    @property
    def state_size(self):
        # This is the cached value
        return self._state_size

    @property
    def prim_path(self):
        """
        Returns:
            str: prim path in the stage.
        """
        return scene_relative_prim_path_to_absolute(self.scene, self._relative_prim_path)

    @property
    def name(self):
        """
        Returns:
            str: unique name assigned to this prim
        """
        return self._name

    @property
    def prim(self):
        """
        Returns:
            Usd.Prim: USD Prim object that this object holds.
        """
        return self._prim

    @property
    def property_names(self):
        """
        Returns:
            set of str: Set of property names that this prim has (e.g.: visibility, proxyPrim, etc.)
        """
        return set(self._prim.GetPropertyNames())

    @property
    def visible(self):
        """
        Returns:
            bool: true if the prim is visible in stage. false otherwise.
        """
        return (
            lazy.pxr.UsdGeom.Imageable(self.prim).ComputeVisibility(lazy.pxr.Usd.TimeCode.Default())
            != lazy.pxr.UsdGeom.Tokens.invisible
        )

    @visible.setter
    def visible(self, visible):
        """
        Sets the visibility of the prim in stage.

        Args:
            visible (bool): flag to set the visibility of the usd prim in stage.
        """
        with og.sim.editing_usd():
            imageable = lazy.pxr.UsdGeom.Imageable(self.prim)
            if visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()

    def is_valid(self):
        """
        Returns:
            bool: True is the current prim path corresponds to a valid prim in stage. False otherwise.
        """
        return lazy.isaacsim.core.utils.prims.is_prim_path_valid(self.prim_path)

    def is_attribute_valid(self, attr):
        """
        Check if the attribute is valid for this prim.

        Args:
            attr (str): Attribute to check

        Returns:
            bool: True if the attribute is valid for this prim. False otherwise.
        """
        return self._prim.GetAttribute(attr).IsValid()

    def get_attribute(self, attr):
        """
        Get this prim's attribute. Should be a valid attribute under self._prim.GetAttributes()

        Returns:
            any: value of the requested @attribute
        """
        return self._prim.GetAttribute(attr).Get()

    def set_attribute(self, attr, val):
        """
        Set this prim's attribute. Should be a valid attribute under self._prim.GetAttributes()

        Args:
            attr (str): Attribute to set
            val (any): Value to set for the attribute. This should be the valid type for that attribute.
        """
        with og.sim.editing_usd():
            self._prim.GetAttribute(attr).Set(val)

    def create_attribute(self, attr, val):
        """
        Create a new attribute for this prim. Should be a valid attribute under self._prim.GetAttributes()

        Args:
            attr (str): Attribute to create
            val (any): Value to set for the attribute. This should be the valid type for that attribute.
        """
        with og.sim.editing_usd():
            self._prim.CreateAttribute(attr, get_sdf_value_type_name(val))

    def get_property(self, prop):
        """
        Sets property @prop with value @val

        Args:
            prop (str): Name of the property to get. See Raw USD Properties in the GUI for examples of property names

        Returns:
            any: Property value
        """
        self._prim.GetProperty(prop).Get()

    def set_property(self, prop, val):
        """
        Sets property @prop with value @val

        Args:
            prop (str): Name of the property to set. See Raw USD Properties in the GUI for examples of property names
            val (any): Value to set for the property. Should be valid for that property
        """
        self._prim.GetProperty(prop).Set(val)

    def get_custom_data(self):
        """
        Get custom data associated with this prim

        Returns:
            dict: Dictionary of any custom information
        """
        return self._prim.GetCustomData()

    def _set_xform_properties(self):
        current_position, current_orientation = self.get_position_orientation()
        properties_to_remove = [
            "xformOp:rotateX",
            "xformOp:rotateXZY",
            "xformOp:rotateY",
            "xformOp:rotateYXZ",
            "xformOp:rotateYZX",
            "xformOp:rotateZ",
            "xformOp:rotateZYX",
            "xformOp:rotateZXY",
            "xformOp:rotateXYZ",
            "xformOp:transform",
        ]
        prop_names = self.prim.GetPropertyNames()
        xformable = lazy.pxr.UsdGeom.Xformable(self.prim)

        with og.sim.editing_usd():
            xformable.ClearXformOpOrder()
            # TODO: wont be able to delete props for non root links on articulated objects
            for prop_name in prop_names:
                if prop_name in properties_to_remove:
                    self.prim.RemoveProperty(prop_name)
            if "xformOp:scale" not in prop_names:
                xform_op_scale = xformable.AddXformOp(
                    lazy.pxr.UsdGeom.XformOp.TypeScale, lazy.pxr.UsdGeom.XformOp.PrecisionDouble, ""
                )
                xform_op_scale.Set(lazy.pxr.Gf.Vec3d([1.0, 1.0, 1.0]))
            else:
                xform_op_scale = lazy.pxr.UsdGeom.XformOp(self._prim.GetAttribute("xformOp:scale"))

            if "xformOp:translate" not in prop_names:
                xform_op_translate = xformable.AddXformOp(
                    lazy.pxr.UsdGeom.XformOp.TypeTranslate, lazy.pxr.UsdGeom.XformOp.PrecisionDouble, ""
                )
            else:
                xform_op_translate = lazy.pxr.UsdGeom.XformOp(self._prim.GetAttribute("xformOp:translate"))

            if "xformOp:orient" not in prop_names:
                xform_op_rot = xformable.AddXformOp(
                    lazy.pxr.UsdGeom.XformOp.TypeOrient, lazy.pxr.UsdGeom.XformOp.PrecisionDouble, ""
                )
            else:
                xform_op_rot = lazy.pxr.UsdGeom.XformOp(self._prim.GetAttribute("xformOp:orient"))
            xformable.SetXformOpOrder([xform_op_translate, xform_op_rot, xform_op_scale])

        # This creates its own editing_usd context.
        BasePrim.set_position_orientation(self, position=current_position, orientation=current_orientation)
        new_position, new_orientation = self.get_position_orientation()
        r1 = T.quat2mat(current_orientation)
        r2 = T.quat2mat(new_orientation)
        # Make sure setting is done correctly
        assert th.allclose(new_position, current_position, atol=1e-4) and th.allclose(r1, r2, atol=1e-3), (
            f"{self.prim_path}: old_pos: {current_position}, new_pos: {new_position}, "
            f"old_orn: {current_orientation}, new_orn: {new_orientation}"
        )

    @property
    def _collision_filter_api(self):
        return ensure_usd_api(self._prim, lazy.pxr.UsdPhysics.FilteredPairsAPI)

    @property
    def _binding_api(self):
        if self.prim.HasAPI(lazy.pxr.UsdShade.MaterialBindingAPI):
            return lazy.pxr.UsdShade.MaterialBindingAPI(self.prim)

        return None

    def has_material(self):
        """
        Returns:
            bool: True if there is a visual material bound to this prim. False otherwise
        """
        if self._binding_api is None:
            return False
        material_path = self._binding_api.GetDirectBinding().GetMaterialPath().pathString
        return material_path != "" and lazy.isaacsim.core.utils.prims.is_prim_path_valid(material_path)

    def set_position_orientation(
        self, position=None, orientation=None, frame: Literal["world", "parent", "scene"] = "world"
    ):
        """
        Sets prim's pose with respect to the specified frame

        Args:
            position (None or 3-array): if specified, (x,y,z) position in the world frame
                Default is None, which means left unchanged.
            orientation (None or 4-array): if specified, (x,y,z,w) quaternion orientation in the world frame.
                Default is None, which means left unchanged.
            frame (Literal): frame to set the pose with respect to, defaults to "world". parent frame
                set position relative to the object parent. scene frame set position relative to the scene.
        """
        assert frame in ["world", "parent", "scene"], f"Invalid frame '{frame}'. Must be 'world', 'parent', or 'scene'."

        # If no position or no orientation are given, get the current position and orientation of the object
        if position is None or orientation is None:
            current_position, current_orientation = self.get_position_orientation(frame=frame)
        position = current_position if position is None else position
        orientation = current_orientation if orientation is None else orientation

        # Convert to th.Tensor if necessary
        position = th.as_tensor(position, dtype=th.float32)
        orientation = th.as_tensor(orientation, dtype=th.float32)

        # Convert to from scene-relative to world if necessary
        if frame == "scene":
            assert self.scene is not None, "cannot set position and orientation relative to scene without a scene"
            position, orientation = self.scene.convert_scene_relative_pose_to_world(position, orientation)

        # If the current pose is not in parent frame, convert to parent frame since that's what we can set.
        if frame != "parent":
            world_transform = T.pose2mat((position, orientation))
            parent_path = str(lazy.isaacsim.core.utils.prims.get_prim_parent(self._prim).GetPath())
            parent_world_transform = get_world_pose_with_scale(parent_path)

            local_transform = th.linalg.inv_ex(parent_world_transform).inverse @ world_transform
            local_transform[:3, :3] /= th.linalg.norm(
                local_transform[:3, :3], dim=0
            )  # unscale local transform's rotation

            # Check that the local transform consists only of a position, scale and rotation
            product = local_transform[:3, :3] @ local_transform[:3, :3].T
            assert th.allclose(
                product, th.diag(th.diag(product)), atol=1e-3
            ), f"{self._prim.GetPath()} local transform is not orthogonal."

            # Return the local pose
            position, orientation = T.mat2pose(local_transform)

        # Assert validity of the orientation
        assert math.isclose(
            th.norm(orientation).item(), 1, abs_tol=1e-3
        ), f"{self.prim_path} desired orientation {orientation} is not a unit quaternion."

        # Actually set the local pose now.
        properties = self.prim.GetPropertyNames()
        position = lazy.pxr.Gf.Vec3d(*position.tolist())
        if "xformOp:translate" not in properties:
            log.error("Translate property needs to be set for {} before setting its position".format(self.name))
        self.set_attribute("xformOp:translate", position)
        orientation = orientation[[3, 0, 1, 2]].tolist()
        if "xformOp:orient" not in properties:
            log.error("Orient property needs to be set for {} before setting its orientation".format(self.name))
        xform_op = self._prim.GetAttribute("xformOp:orient")
        if xform_op.GetTypeName() == "quatf":
            rotq = lazy.pxr.Gf.Quatf(*orientation)
        else:
            rotq = lazy.pxr.Gf.Quatd(*orientation)
        with og.sim.editing_usd():
            xform_op.Set(rotq)

        og.sim.fabric_hierarchy.update_world_xforms()

    def get_position_orientation(self, frame: Literal["world", "scene", "parent"] = "world", clone=True):
        """
        Gets prim's pose with respect to the specified frame.

        Args:
            frame (Literal): frame to get the pose with respect to. Default to world.
                parent frame: get position relative to the object parent.
                scene frame: get position relative to the scene.
            clone (bool): Whether to clone the internal buffer or not when grabbing data

        Returns:
            2-tuple:
                - th.Tensor: (x,y,z) position in the specified frame
                - th.Tensor: (x,y,z,w) quaternion orientation in the specified frame
        """
        assert frame in ["world", "parent", "scene"], f"Invalid frame '{frame}'. Must be 'world', 'parent', or 'scene'."
        if frame == "world":
            return get_world_pose(self.prim_path)
        elif frame == "scene":
            assert self.scene is not None, "Cannot get position and orientation relative to scene without a scene"
            return self.scene.convert_world_pose_to_scene_relative(*get_world_pose(self.prim_path))
        else:
            return get_local_pose(self.prim_path)

    def set_position(self, position):
        """
        Set this prim's position with respect to the world frame

        Args:
            position (3-array): (x,y,z) global cartesian position to set
        """
        log.warning(
            "set_position is deprecated and will be removed in a future release. Use set_position_orientation(position=position) instead"
        )
        return self.set_position_orientation(position=position)

    def get_position(self):
        """
        Get this prim's position with respect to the world frame

        Returns:
            3-array: (x,y,z) global cartesian position of this prim
        """
        log.warning(
            "get_position is deprecated and will be removed in a future release. Use get_position_orientation()[0] instead."
        )
        return self.get_position_orientation()[0]

    def set_orientation(self, orientation):
        """
        Set this prim's orientation with respect to the world frame

        Args:
            orientation (4-array): (x,y,z,w) global quaternion orientation to set
        """
        log.warning(
            "set_orientation is deprecated and will be removed in a future release. Use set_position_orientation(orientation=orientation) instead"
        )
        self.set_position_orientation(orientation=orientation)

    def get_orientation(self):
        """
        Get this prim's orientation with respect to the world frame

        Returns:
            4-array: (x,y,z,w) global quaternion orientation of this prim
        """
        log.warning(
            "get_orientation is deprecated and will be removed in a future release. Use get_position_orientation()[1] instead"
        )
        return self.get_position_orientation()[1]

    def get_rpy(self):
        """
        Get this prim's orientation with respect to the world frame

        Returns:
            3-array: (roll, pitch, yaw) global euler orientation of this prim
        """
        return quat2euler(self.get_position_orientation()[1])

    def get_xy_orientation(self):
        """
        Get this prim's orientation on the XY plane of the world frame. This is obtained by
        projecting the forward vector onto the XY plane and then computing the angle.
        """
        return T.calculate_xy_plane_angle(self.get_position_orientation()[1])

    def get_local_pose(self):
        """
        Gets prim's pose with respect to the prim's local frame (its parent frame)

        Returns:
            2-tuple:
                - 3-array: (x,y,z) position in the local frame
                - 4-array: (x,y,z,w) quaternion orientation in the local frame
        """
        log.warning(
            'get_local_pose is deprecated and will be removed in a future release. Use get_position_orientation(frame="parent") instead'
        )
        return self.get_position_orientation(frame="parent")

    def set_local_pose(self, position=None, orientation=None):
        """
        Sets prim's pose with respect to the local frame (the prim's parent frame).

        Args:
            position (None or 3-array): if specified, (x,y,z) position in the local frame of the prim
                (with respect to its parent prim). Default is None, which means left unchanged.
            orientation (None or 4-array): if specified, (x,y,z,w) quaternion orientation in the local frame of the prim
                (with respect to its parent prim). Default is None, which means left unchanged.
        """
        log.warning(
            'set_local_pose is deprecated and will be removed in a future release. Use set_position_orientation(position=position, orientation=orientation, frame="parent") instead'
        )
        return self.set_position_orientation(position, orientation, frame="parent")

    @property
    def aabb(self):
        aabb_min, aabb_max = lazy.omni.usd.get_context().compute_path_world_bounding_box(self.prim_path)
        log.warning(
            "Computing AABB of a BasePrim using the USD context is slow and unreliable, especially when fabric is enabled. "
            "This is provided as a convenience for USD editing use cases and should generally not be used for physical objects."
        )
        return th.tensor(aabb_min), th.tensor(aabb_max)

    @property
    def aabb_center(self):
        min_corner, max_corner = self.aabb
        return (min_corner + max_corner) / 2

    @property
    def aabb_extent(self):
        min_corner, max_corner = self.aabb
        return max_corner - min_corner

    def get_world_scale(self):
        """
        Gets prim's scale with respect to the world's frame.

        Returns:
            th.tensor: scale applied to the prim's dimensions in the world frame. shape is (3, ).
        """
        prim_tf = lazy.pxr.UsdGeom.Xformable(self._prim).ComputeLocalToWorldTransform(lazy.pxr.Usd.TimeCode.Default())
        transform = lazy.pxr.Gf.Transform()
        transform.SetMatrix(prim_tf)
        return th.tensor(transform.GetScale())

    @property
    def scaled_transform(self):
        """
        Returns the scaled transform of this prim.
        """
        return get_world_pose_with_scale(self.prim_path)

    def transform_local_points_to_world(self, points):
        return T.transform_points(points, self.scaled_transform)

    @property
    def scale(self):
        """
        Gets prim's scale with respect to the local frame (the parent's frame).

        Returns:
            th.tensor: scale applied to the prim's dimensions in the local frame. shape is (3, ).
        """
        if self._cached_scale is not None:
            return self._cached_scale
        scale = self.get_attribute("xformOp:scale")
        assert scale is not None, "Attribute 'xformOp:scale' is None for prim {}".format(self.name)
        return th.tensor(scale)

    @scale.setter
    def scale(self, scale):
        """
        Sets prim's scale with respect to the local frame (the prim's parent frame).

        Args:
            scale (float or th.tensor): scale to be applied to the prim's dimensions. shape is (3, ).
                                          Defaults to None, which means left unchanged.
        """
        if isinstance(scale, th.Tensor):
            scale = scale
        elif isinstance(scale, Iterable):
            scale = th.tensor(scale, dtype=th.float32)
        else:
            scale = th.ones(3, dtype=th.float32) * scale
        assert th.all(scale > 0), f"Scale {scale} must consist of positive numbers."
        # Invalidate the cached scale
        self._cached_scale = None
        scale = lazy.pxr.Gf.Vec3d(*scale.tolist())
        properties = self.prim.GetPropertyNames()
        if "xformOp:scale" not in properties:
            log.error("Scale property needs to be set for {} before setting its scale".format(self.name))
        self.set_attribute("xformOp:scale", scale)

    @property
    def material(self):
        """
        Returns:
            None or MaterialPrim: The bound material to this prim, if there is one
        """
        return self._material

    @material.setter
    def material(self, material):
        """
        Set the material @material for this prim. This will also bind the material to this prim

        Args:
            material (MaterialPrim): Material to bind to this prim
        """
        binding_api = self._binding_api
        with og.sim.editing_usd():
            binding_api.Bind(
                lazy.pxr.UsdShade.Material(material.prim),
                bindingStrength=lazy.pxr.UsdShade.Tokens.weakerThanDescendants,
            )
        self._material = material

    def add_filtered_collision_pair(self, prim):
        """
        Adds a collision filter pair with another prim

        Args:
            prim (BasePrim): Another prim to filter collisions with
        """
        self_api = self._collision_filter_api
        other_api = prim._collision_filter_api
        with og.sim.editing_usd():
            # Add to both this prim's and the other prim's filtered pair
            self_api.GetFilteredPairsRel().AddTarget(prim.prim_path)
            other_api.GetFilteredPairsRel().AddTarget(self.prim_path)

    def remove_filtered_collision_pair(self, prim):
        """
        Removes a collision filter pair with another prim

        Args:
            prim (BasePrim): Another prim to remove filter collisions with
        """
        self_api = self._collision_filter_api
        other_api = prim._collision_filter_api
        with og.sim.editing_usd():
            self_api.GetFilteredPairsRel().RemoveTarget(prim.prim_path)
            other_api.GetFilteredPairsRel().RemoveTarget(self.prim_path)

    def _dump_state(self):
        pos, ori = self.get_position_orientation()
        return dict(pos=pos, ori=ori)

    def _load_state(self, state):
        pos, orn = state["pos"], state["ori"]
        self.set_position_orientation(pos, orn)

    def serialize(self, state):
        return th.cat([state["pos"], state["ori"]])

    def deserialize(self, state):
        # We deserialize deterministically by knowing the order of values -- pos, ori
        return dict(pos=state[0:3], ori=state[3:7]), 7
