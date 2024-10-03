from functools import cached_property

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.utils.numpy_utils import vtarray_to_torch
from omnigibson.utils.python_utils import assert_valid_key
from omnigibson.utils.usd_utils import PoseAPI, mesh_prim_shape_to_trimesh_mesh


class GeomPrim(XFormPrim):
    """
    Provides high level functions to deal with a geom prim and its attributes / properties.
    If there is an geom prim present at the path, it will use it. By default, a geom prim cannot be directly
    created from scratch.at

    Args:
        relative_prim_path (str): Scene-local prim path of the Prim to encapsulate or create.
        name (str): Name for the object. Names need to be unique per scene.
        load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
            loading this prim at runtime. For this mesh prim, the below values can be specified:
    """

    def __init__(
        self,
        relative_prim_path,
        name,
        load_config=None,
    ):

        # Run super method
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            load_config=load_config,
        )

    def _load(self):
        # This should not be called, because this prim cannot be instantiated from scratch!
        raise NotImplementedError("By default, a geom prim cannot be created from scratch.")

    def _post_load(self):
        # run super first
        super()._post_load()

        # By default, GeomPrim shows up in the rendering.
        self.purpose = "default"

    @property
    def purpose(self):
        """
        Returns:
            str: the purpose used for this geom, one of {"default", "render", "proxy", "guide"}
        """
        return self.get_attribute("purpose")

    @purpose.setter
    def purpose(self, purpose):
        """
        Sets the purpose of this geom

        Args:
            purpose (str): the purpose used for this geom, one of {"default", "render", "proxy", "guide"}
        """
        self.set_attribute("purpose", purpose)

    @property
    def color(self):
        """
        Returns:
            None or 3-array: If set, the default RGB color used for this visual geom
        """
        if self.has_material():
            return self.material.diffuse_color_constant
        else:
            color = self.get_attribute("primvars:displayColor")
            return None if color is None else th.tensor(color)[0]

    @color.setter
    def color(self, rgb):
        """
        Sets the RGB color of this visual mesh

        Args:
            3-array: The default RGB color used for this visual geom
        """
        rgb = th.as_tensor(rgb)
        if self.has_material():
            self.material.diffuse_color_constant = rgb
        else:
            self.set_attribute("primvars:displayColor", rgb.cpu().numpy())

    @property
    def opacity(self):
        """
        Returns:
            None or float: If set, the default opacity used for this visual geom
        """
        if self.has_material():
            return self.material.opacity_constant
        else:
            opacity = self.get_attribute("primvars:displayOpacity")
            return None if opacity is None else th.tensor(opacity)[0]

    @opacity.setter
    def opacity(self, opacity):
        """
        Sets the opacity of this visual mesh

        Args:
            opacity: The default opacity used for this visual geom
        """
        if self.has_material():
            self.material.opacity_constant = opacity
        else:
            self.set_attribute("primvars:displayOpacity", th.tensor([opacity]))

    @property
    def points(self):
        """
        Returns:
            th.tensor: Local poses of all points
        """
        # If the geom is a mesh we can directly return its points.
        mesh = self.prim
        mesh_type = mesh.GetPrimTypeInfo().GetTypeName()
        if mesh_type == "Mesh":
            # If the geom is a mesh we can directly return its points.
            return vtarray_to_torch(self.prim.GetAttribute("points").Get(), dtype=th.float32)
        else:
            # Return the vertices of the trimesh
            return th.tensor(mesh_prim_shape_to_trimesh_mesh(mesh).vertices, dtype=th.float32)

    @property
    def points_in_parent_frame(self):
        points = self.points
        if points is None:
            return None
        position, orientation = self.get_position_orientation(frame="parent")
        scale = self.scale
        points_scaled = points * scale
        points_rotated = (T.quat2mat(orientation) @ points_scaled.T).T
        points_transformed = points_rotated + position
        return points_transformed

    @property
    def aabb(self):
        world_pose_w_scale = PoseAPI.get_world_pose_with_scale(self.prim_path)

        # transform self.points into world frame
        points = self.points
        points_homogeneous = th.cat((points, th.ones((points.shape[0], 1))), dim=1)
        points_transformed = (points_homogeneous @ world_pose_w_scale.T)[:, :3]

        aabb_lo = th.min(points_transformed, dim=0).values
        aabb_hi = th.max(points_transformed, dim=0).values
        return aabb_lo, aabb_hi

    @property
    def aabb_extent(self):
        """
        Bounding box extent of this geom prim

        Returns:
            3-array: (x,y,z) bounding box
        """
        min_corner, max_corner = self.aabb
        return max_corner - min_corner

    @property
    def aabb_center(self):
        """
        Bounding box center of this geom prim

        Returns:
            3-array: (x,y,z) bounding box center
        """
        min_corner, max_corner = self.aabb
        return (max_corner + min_corner) / 2.0

    @cached_property
    def extent(self):
        """
        Returns:
            th.tensor: The unscaled 3d extent of the mesh in its local frame.
        """
        points = self.points
        return th.max(points, dim=0).values - th.min(points, dim=0).values


class CollisionGeomPrim(GeomPrim):

    def __init__(
        self,
        relative_prim_path,
        name,
        load_config=None,
    ):
        # Store values created at runtime
        self._collision_api = None
        self._mesh_collision_api = None
        self._physx_collision_api = None
        self._applied_physics_material = None

        # Run super method
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            load_config=load_config,
        )

    def _post_load(self):
        # run super first
        super()._post_load()

        # By default, CollisionGeomPrim does not show up in the rendering.
        self.purpose = "guide"

        # Create API references
        self._collision_api = (
            lazy.pxr.UsdPhysics.CollisionAPI(self._prim)
            if self._prim.HasAPI(lazy.pxr.UsdPhysics.CollisionAPI)
            else lazy.pxr.UsdPhysics.CollisionAPI.Apply(self._prim)
        )
        self._physx_collision_api = (
            lazy.pxr.PhysxSchema.PhysxCollisionAPI(self._prim)
            if self._prim.HasAPI(lazy.pxr.PhysxSchema.PhysxCollisionAPI)
            else lazy.pxr.PhysxSchema.PhysxCollisionAPI.Apply(self._prim)
        )

        # Optionally add mesh collision API if this is a mesh
        if self._prim.GetPrimTypeInfo().GetTypeName() == "Mesh":
            self._mesh_collision_api = (
                lazy.pxr.UsdPhysics.MeshCollisionAPI(self._prim)
                if self._prim.HasAPI(lazy.pxr.UsdPhysics.MeshCollisionAPI)
                else lazy.pxr.UsdPhysics.MeshCollisionAPI.Apply(self._prim)
            )
            # Set the approximation to be convex hull by default
            self.set_collision_approximation(approximation_type="convexHull")

    @property
    def collision_enabled(self):
        """
        Returns:
            bool: Whether collisions are enabled for this collision mesh
        """
        return self.get_attribute("physics:collisionEnabled")

    @collision_enabled.setter
    def collision_enabled(self, enabled):
        """
        Sets whether collisions are enabled for this mesh

        Args:
            enabled (bool): Whether collisions should be enabled for this mesh
        """
        # Currently, trying to toggle while simulator is playing while using GPU dynamics results in a crash, so we
        # assert that the sim is stopped here
        if self._initialized and gm.USE_GPU_DYNAMICS:
            assert og.sim.is_stopped(), "Cannot toggle collisions while using GPU dynamics unless simulator is stopped!"
        self.set_attribute("physics:collisionEnabled", enabled)

    # TODO: Maybe this should all be added to RigidPrim instead?
    def set_contact_offset(self, offset):
        """
        Args:
            offset (float): Contact offset of a collision shape. Allowed range [maximum(0, rest_offset), 0].
                            Default value is -inf, means default is picked by simulation based on the shape extent.
        """
        self._physx_collision_api.GetContactOffsetAttr().Set(offset)
        return

    def get_contact_offset(self):
        """
        Returns:
            float: contact offset of the collision shape.
        """
        return self._physx_collision_api.GetContactOffsetAttr().Get()

    def set_rest_offset(self, offset):
        """
        Args:
            offset (float): Rest offset of a collision shape. Allowed range [-max_float, contact_offset.
                            Default value is -inf, means default is picked by simulatiion. For rigid bodies its zero.
        """
        self._physx_collision_api.GetRestOffsetAttr().Set(offset)
        return

    def get_rest_offset(self):
        """
        Returns:
            float: rest offset of the collision shape.
        """
        return self._physx_collision_api.GetRestOffsetAttr().Get()

    def set_torsional_patch_radius(self, radius):
        """
        Args:
            radius (float): radius of the contact patch used to apply torsional friction. Allowed range [0, max_float].
        """
        self._physx_collision_api.GetTorsionalPatchRadiusAttr().Set(radius)
        return

    def get_torsional_patch_radius(self):
        """
        Returns:
            float: radius of the contact patch used to apply torsional friction. Allowed range [0, max_float].
        """
        return self._physx_collision_api.GetTorsionalPatchRadiusAttr().Get()

    def set_min_torsional_patch_radius(self, radius):
        """
        Args:
            radius (float): minimum radius of the contact patch used to apply torsional friction. Allowed range [0, max_float].
        """
        self._physx_collision_api.GetMinTorsionalPatchRadiusAttr().Set(radius)
        return

    def get_min_torsional_patch_radius(self):
        """
        Returns:
            float: minimum radius of the contact patch used to apply torsional friction. Allowed range [0, max_float].
        """
        return self._physx_collision_api.GetMinTorsionalPatchRadiusAttr().Get()

    def set_collision_approximation(self, approximation_type):
        """
        Args:
            approximation_type (str): approximation used for collision.
                Can be one of: {"none", "convexHull", "convexDecomposition", "meshSimplification", "sdf",
                    "boundingSphere", "boundingCube"}
                If None, the approximation will use the underlying triangle mesh.
        """
        assert self._mesh_collision_api is not None, "collision_approximation only applicable for meshes!"
        assert_valid_key(
            key=approximation_type,
            valid_keys={
                "none",
                "convexHull",
                "convexDecomposition",
                "meshSimplification",
                "sdf",
                "boundingSphere",
                "boundingCube",
            },
            name="collision approximation type",
        )

        # Make sure to add the appropriate API if we're setting certain values
        if approximation_type == "convexHull" and not self._prim.HasAPI(
            lazy.pxr.PhysxSchema.PhysxConvexHullCollisionAPI
        ):
            lazy.pxr.PhysxSchema.PhysxConvexHullCollisionAPI.Apply(self._prim)
        elif approximation_type == "convexDecomposition" and not self._prim.HasAPI(
            lazy.pxr.PhysxSchema.PhysxConvexDecompositionCollisionAPI
        ):
            lazy.pxr.PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(self._prim)
        elif approximation_type == "meshSimplification" and not self._prim.HasAPI(
            lazy.pxr.PhysxSchema.PhysxTriangleMeshSimplificationCollisionAPI
        ):
            lazy.pxr.PhysxSchema.PhysxTriangleMeshSimplificationCollisionAPI.Apply(self._prim)
        elif approximation_type == "sdf" and not self._prim.HasAPI(lazy.pxr.PhysxSchema.PhysxSDFMeshCollisionAPI):
            lazy.pxr.PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(self._prim)
        elif approximation_type == "none" and not self._prim.HasAPI(lazy.pxr.PhysxSchema.PhysxTriangleMeshCollisionAPI):
            lazy.pxr.PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(self._prim)

        if approximation_type == "convexHull":
            pch_api = lazy.pxr.PhysxSchema.PhysxConvexHullCollisionAPI(self._prim)
            # Also make sure the maximum vertex count is 60 (max number compatible with GPU)
            # https://docs.omniverse.nvidia.com/app_create/prod_extensions/ext_physics/rigid-bodies.html#collision-settings
            if pch_api.GetHullVertexLimitAttr().Get() is None:
                pch_api.CreateHullVertexLimitAttr()
            pch_api.GetHullVertexLimitAttr().Set(60)

        self._mesh_collision_api.GetApproximationAttr().Set(approximation_type)

    def get_collision_approximation(self):
        """
        Returns:
            str: approximation used for collision, could be "none", "convexHull" or "convexDecomposition"
        """
        assert self._mesh_collision_api is not None, "collision_approximation only applicable for meshes!"
        return self._mesh_collision_api.GetApproximationAttr().Get()

    def apply_physics_material(self, physics_material, weaker_than_descendants=False):
        """
        Used to apply physics material to the held prim and optionally its descendants.

        Args:
            physics_material (PhysicsMaterial): physics material to be applied to the held prim. This where you want to
                                                define friction, restitution..etc. Note: if a physics material is not
                                                defined, the defaults will be used from PhysX.
            weaker_than_descendants (bool, optional): True if the material shouldn't override the descendants
                                                      materials, otherwise False. Defaults to False.
        """
        if weaker_than_descendants:
            self._binding_api.Bind(
                physics_material.material,
                bindingStrength=lazy.pxr.UsdShade.Tokens.weakerThanDescendants,
                materialPurpose="physics",
            )
        else:
            self._binding_api.Bind(
                physics_material.material,
                bindingStrength=lazy.pxr.UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )
        self._applied_physics_material = physics_material
        return

    def get_applied_physics_material(self):
        """
        Returns the current applied physics material in case it was applied using apply_physics_material or not.

        Returns:
            PhysicsMaterial: the current applied physics material.
        """
        if self._applied_physics_material is not None:
            return self._applied_physics_material
        else:
            physics_binding = self._binding_api.GetDirectBinding(materialPurpose="physics")
            path = physics_binding.GetMaterialPath()
            if path == "":
                return None
            else:
                self._applied_physics_material = lazy.omni.isaac.core.materials.PhysicsMaterial(prim_path=path)
                return self._applied_physics_material


class VisualGeomPrim(GeomPrim):
    pass


class CollisionVisualGeomPrim(CollisionGeomPrim, VisualGeomPrim):

    def _post_load(self):
        # run super first
        super()._post_load()

        # The purpose should be default, not guide as set by CollisionGeomPrim
        # this is to make sure the geom is visualized, even though it's also collidable
        self.purpose = "default"
