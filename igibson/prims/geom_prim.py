# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from collections import Iterable, OrderedDict
from typing import Optional, Tuple
from pxr import Gf, Usd, Sdf, UsdGeom, UsdShade, UsdPhysics, PhysxSchema
from omni.isaac.core.materials import PreviewSurface, OmniGlass, OmniPBR, VisualMaterial
from omni.isaac.core.utils.rotations import gf_quat_to_np_array
from omni.isaac.core.utils.transformations import tf_matrix_from_pose
from omni.isaac.core.utils.prims import (
    get_prim_at_path,
    move_prim,
    query_parent_path,
    is_prim_path_valid,
    define_prim,
    get_prim_parent,
    get_prim_object_type,
)
import numpy as np
import carb
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.materials import PhysicsMaterial
from igibson.prims.prim_base import BasePrim


class GeomPrim(BasePrim):
    """
    Provides high level functions to deal with a geom prim and its attributes / properties.
    If there is an geom prim present at the path, it will use it. By default, a geom prim cannot be directly
    created from scratch.at

        Args:
            prim_path (str): prim path of the Prim to encapsulate or create.
            name (str): Name for the object. Names need to be unique per scene.
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime. For this mesh prim, the below values can be specified:
    """

    def __init__(
        self,
        prim_path,
        name,
        load_config=None,
    ):

        # Run super method
        super().__init__(
            prim_path=prim_path,
            name=name,
            load_config=load_config,
        )

    def _load(self, simulator=None):
        # This should not be called, because this prim cannot be instantiated from scratch!
        raise NotImplementedError("By default, a geom prim cannot be created from scratch.")

    def _dump_state(self):
        # No state to dump
        return OrderedDict()

    def _load_state(self, state):
        # No state to load
        pass

    def _serialize(self, state):
        # No state to serialize
        return np.array([])

    def _deserialize(self, state):
        # No state to deserialize
        return OrderedDict()

    def duplicate(self, simulator, prim_path):
        # Cannot directly duplicate a mesh prim
        raise NotImplementedError("Cannot directly duplicate a geom prim!")


class CollisionGeomPrim(GeomPrim):

    def __init__(
        self,
        prim_path,
        name,
        load_config=None,
    ):
        # Store values created at runtime
        self._collision_api = None
        self._mesh_collision_api = None
        self._physx_collision_api = None

        # Run super method
        super().__init__(
            prim_path=prim_path,
            name=name,
            load_config=load_config,
        )

    def _post_load(self):
        # run super first
        super()._post_load()

        # Create API references
        self._collision_api = UsdPhysics.CollisionAPI(self._prim) if \
            self._prim.HasAPI(UsdPhysics.CollisionAPI) else UsdPhysics.CollisionAPI.Apply(self._prim)
        self._physx_collision_api = PhysxSchema.PhysxCollisionAPI(self._prim) if \
            self._prim.HasAPI(PhysxSchema.PhysxCollisionAPI) else PhysxSchema.PhysxCollisionAPI.Apply(self._prim)

        # Optionally add mesh collision API if this is a mesh
        if self._prim.GetPrimTypeInfo().GetTypeName() == "Mesh":
            self._mesh_collision_api = UsdPhysics.MeshCollisionAPI(self._prim) if \
                self._prim.HasAPI(UsdPhysics.MeshCollisionAPI) else UsdPhysics.MeshCollisionAPI.Apply(self._prim)

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
            approximation_type (str): approximation used for collision, could be "none", "convexHull" or "convexDecomposition"
        """
        assert self._mesh_collision_api is not None, "collision_approximation only applicable for meshes!"
        self._mesh_collision_api.GetApproximationAttr().Set(approximation_type)
        return

    def get_collision_approximation(self):
        """
        Returns:
            str: approximation used for collision, could be "none", "convexHull" or "convexDecomposition"
        """
        assert self._mesh_collision_api is not None, "collision_approximation only applicable for meshes!"
        return self._mesh_collision_api.GetApproximationAttr().Get()

    def apply_physics_material(self, physics_material, weaker_than_descendants=False):
        """Used to apply physics material to the held prim and optionally its descendants.

        Args:
            physics_material (PhysicsMaterial): physics material to be applied to the held prim. This where you want to
                                                define friction, restitution..etc. Note: if a physics material is not
                                                defined, the defaults will be used from PhysX.
            weaker_than_descendants (bool, optional): True if the material shouldn't override the descendants
                                                      materials, otherwise False. Defaults to False.
        """
        if self._binding_api is None:
            if self._prim.HasAPI(UsdShade.MaterialBindingAPI):
                self._binding_api = UsdShade.MaterialBindingAPI(self.prim)
            else:
                self._binding_api = UsdShade.MaterialBindingAPI.Apply(self.prim)
        if weaker_than_descendants:
            self._binding_api.Bind(
                physics_material.material,
                bindingStrength=UsdShade.Tokens.weakerThanDescendants,
                materialPurpose="physics",
            )
        else:
            self._binding_api.Bind(
                physics_material.material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )
        self._applied_physics_material = physics_material
        return

    def get_applied_physics_material(self):
        """Returns the current applied physics material in case it was applied using apply_physics_material or not.

        Returns:
            PhysicsMaterial: the current applied physics material.
        """
        if self._binding_api is None:
            if self._prim.HasAPI(UsdShade.MaterialBindingAPI):
                self._binding_api = UsdShade.MaterialBindingAPI(self.prim)
            else:
                self._binding_api = UsdShade.MaterialBindingAPI.Apply(self.prim)
        if self._applied_physics_material is not None:
            return self._applied_physics_material
        else:
            physics_binding = self._binding_api.GetDirectBinding(materialPurpose="physics")
            path = physics_binding.GetMaterialPath()
            if path == "":
                return None
            else:
                self._applied_physics_material = PhysicsMaterial(prim_path=path)
                return self._applied_physics_material


class VisualGeomPrim(GeomPrim):

    def __init__(
            self,
            prim_path,
            name,
            load_config=None,
    ):

        # Run super method
        super().__init__(
            prim_path=prim_path,
            name=name,
            load_config=load_config,
        )

    @property
    def color(self):
        """
        Returns:
            None or 3-array: If set, the default RGB color used for this visual geom
        """
        color = self.get_attribute("primvars:displayColor")
        return None if color is None else np.array(color)[0]

    @color.setter
    def color(self, rgb):
        """
        Sets the RGB color of this visual mesh

        Args:
            3-array: The default RGB color used for this visual geom
        """
        self.set_attribute("primvars:displayColor", np.array(rgb))

    @property
    def opacity(self):
        """
        Returns:
            None or float: If set, the default opacity used for this visual geom
        """
        opacity = self.get_attribute("primvars:displayOpacity")
        return None if opacity is None else np.array(opacity)[0]

    @opacity.setter
    def opacity(self, opacity):
        """
        Sets the opacity of this visual mesh

        Args:
            opacity: The default opacity used for this visual geom
        """
        self.set_attribute("primvars:displayOpacity", np.array([opacity]))
