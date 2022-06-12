import logging
import numpy as np
from igibson.objects.stateful_object import StatefulObject
from igibson.utils.python_utils import assert_valid_key

from pxr import Gf, Usd, Sdf, UsdGeom, UsdPhysics, PhysxSchema, UsdShade
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from igibson.utils.constants import PrimType
from igibson.utils.usd_utils import create_mesh_prim_with_default_xform
from igibson.utils.types import PRIMITIVE_MESH_TYPES
import omni
import carb



VALID_RADIUS_OBJECTS = {"Cone", "Cylinder", "Disk", "Sphere"}
VALID_HEIGHT_OBJECTS = {"Cone", "Cylinder"}
VALID_SIZE_OBJECTS = {"Cube", "Torus"}


class PrimitiveObject(StatefulObject):
    """
    PrimitiveObjects are objects defined by a single geom, e.g: sphere, mesh, cube, etc.
    """

    def __init__(
        self,
        prim_path,
        primitive_type,
        name=None,
        category="object",
        class_id=None,
        uuid=None,
        scale=None,
        rendering_params=None,
        visible=True,
        fixed_base=False,
        visual_only=False,
        self_collisions=False,
        prim_type=PrimType.RIGID,
        load_config=None,
        abilities=None,
        rgba=(0.5, 0.5, 0.5, 1.0),
        radius=None,
        height=None,
        size=None,
        **kwargs,
    ):
        """
        @param prim_path: str, global path in the stage to this object
        @param primitive_type: str, type of primitive object to create. Should be one of:
            {"Cone", "Cube", "Cylinder", "Disk", "Plane", "Sphere", "Torus"}
        @param name: Name for the object. Names need to be unique per scene. If no name is set, a name will be generated
            at the time the object is added to the scene, using the object's category.
        @param category: Category for the object. Defaults to "object".
        @param class_id: What class ID the object should be assigned in semantic segmentation rendering mode.
        @param uuid: Unique unsigned-integer identifier to assign to this object (max 8-numbers).
            If None is specified, then it will be auto-generated
        @param scale: float or 3-array, sets the scale for this object. A single number corresponds to uniform scaling
            along the x,y,z axes, whereas a 3-array specifies per-axis scaling.
        @param rendering_params: Any relevant rendering settings for this object.
        @param visible: bool, whether to render this object or not in the stage
        @param fixed_base: bool, whether to fix the base of this object or not
        @param visual_only: whether this object should be a visual-only (i.e.: not affected by collisions / gravity)
            object or not
        self_collisions (bool): Whether to enable self collisions for this object
        prim_type (PrimType): Which type of prim the object is, Valid options are: {PrimType.RIGID, PrimType.CLOTH}
        load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
            loading this prim at runtime.

            Can specify:
                scale (None or float or 3-array): If specified, sets the scale for this object. A single number
                    corresponds
                    to uniform scaling along the x,y,z axes, whereas a 3-array specifies per-axis scaling.
                mass (None or float): If specified, mass of this body in kg

        @param abilities: dict in the form of {ability: {param: value}} containing
            object abilities and parameters.
        rgba (4-array): (R, G, B, A) values to set for this object
        radius (None or float): If specified, sets the radius for this object. This value is scaled by @scale
            Note: Should only be specified if the @primitive_type is one of {"Cone", "Cylinder", "Disk", "Sphere"}
        height (None or float): If specified, sets the height for this object. This value is scaled by @scale
            Note: Should only be specified if the @primitive_type is one of {"Cone", "Cylinder"}
        size (None or float): If specified, sets the size for this object. This value is scaled by @scale
            Note: Should only be specified if the @primitive_type is one of {"Cube", "Torus"}
        kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
            for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
        # Compose load config and add rgba values
        load_config = dict() if load_config is None else load_config
        load_config["color"] = np.array(rgba[:3])
        load_config["opacity"] = rgba[3]
        load_config["radius"] = 1.0 if radius is None else radius
        load_config["height"] = 1.0 if height is None else height
        load_config["size"] = 1.0 if size is None else size

        # Initialize other internal variables
        self._vis_geom = None
        self._col_geom = None

        # Make sure primitive type is valid
        assert_valid_key(key=primitive_type, valid_keys=PRIMITIVE_MESH_TYPES, name="primitive mesh type")
        self._primitive_type = primitive_type

        super().__init__(
            prim_path=prim_path,
            name=name,
            category=category,
            class_id=class_id,
            uuid=uuid,
            scale=scale,
            rendering_params=rendering_params,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            self_collisions=self_collisions,
            prim_type=prim_type,
            load_config=load_config,
            abilities=abilities,
            **kwargs,
        )

    def _load(self, simulator=None):
        """
        Load the object into pybullet and set it to the correct pose
        """
        logging.info(f"Loading the following primitive: {self._primitive_type}")

        # Define an Xform at the specified path
        stage = simulator.stage
        prim = stage.DefinePrim(self._prim_path, "Xform")

        if self._prim_type == PrimType.RIGID:
            # Define a nested mesh corresponding to the root link for this prim
            base_link = stage.DefinePrim(f"{self._prim_path}/base_link", "Xform")
            visual_mesh_path = f"{self._prim_path}/base_link/visual"
            create_mesh_prim_with_default_xform(self._primitive_type, visual_mesh_path, stage=stage)
            collision_mesh_path = f"{self._prim_path}/base_link/collision"
            create_mesh_prim_with_default_xform(self._primitive_type, collision_mesh_path, stage=stage)

            self._vis_geom = UsdGeom.Mesh.Define(stage, visual_mesh_path)
            self._col_geom = UsdGeom.Mesh.Define(stage, collision_mesh_path)

            # Add collision API to collision geom
            UsdPhysics.CollisionAPI.Apply(self._col_geom.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(self._col_geom.GetPrim())
            PhysxSchema.PhysxCollisionAPI.Apply(self._col_geom.GetPrim())

        elif self._prim_type == PrimType.CLOTH:
            # For Cloth, the base link itself is a cloth mesh
            visual_mesh_path = f"{self._prim_path}/base_link"
            # TODO (eric): configure u_patches and v_patches
            create_mesh_prim_with_default_xform(self._primitive_type, visual_mesh_path, stage=stage,
                                                u_patches=None, v_patches=None)
            self._vis_prim = UsdGeom.Mesh.Define(stage, visual_mesh_path)
            self._col_prim = None


        return prim

    def _post_load(self):
        # Run super first
        super()._post_load()

        if self._prim_type == PrimType.RIGID:
            visual_geom_prim = list(self.links["base_link"].visual_meshes.values())[0]
        elif self._prim_type == PrimType.CLOTH:
            visual_geom_prim = self.links["base_link"]

        visual_geom_prim.color = self._load_config["color"]
        visual_geom_prim.opacity = self._load_config["opacity"]

        # Possibly set scalings
        if self._primitive_type in VALID_RADIUS_OBJECTS:
            self.radius = self._load_config["radius"]
        if self._primitive_type in VALID_HEIGHT_OBJECTS:
            self.height = self._load_config["height"]
        if self._primitive_type in VALID_SIZE_OBJECTS:
            self.size = self._load_config["size"]

    @property
    def radius(self):
        """
        Gets this object's radius, if it exists.

        Note: Can only be called if the primitive type is one of {"Cone", "Cylinder", "Disk", "Sphere"}

        Returns:
            float: radius for this object
        """
        assert_valid_key(key=self._primitive_type, valid_keys=VALID_RADIUS_OBJECTS, name="primitive object with radius")
        # Scale 1 1 1 --> side length 2, so radius is 1/2 side = scale
        return self.scale[0]

    @radius.setter
    def radius(self, radius):
        """
        Sets this object's radius

        Note: Can only be called if the primitive type is one of {"Cone", "Cylinder", "Disk", "Sphere"}

        Args:
            radius (float): radius to set
        """
        assert_valid_key(key=self._primitive_type, valid_keys=VALID_RADIUS_OBJECTS, name="primitive object with radius")
        original_scale = self.scale
        if self._primitive_type == "Sphere":
            original_scale[:] = radius
        else:
            original_scale[:2] = radius
        self.scale = original_scale

    @property
    def height(self):
        """
        Gets this object's height, if it exists.

        Note: Can only be called if the primitive type is one of {"Cone", "Cylinder"}

        Returns:
            float: height for this object
        """
        assert_valid_key(key=self._primitive_type, valid_keys=VALID_HEIGHT_OBJECTS, name="primitive object with height")

        # TODO (eric): currently scale [1, 1, 1] will have height 2.0
        return self.scale[2] * 2.0

    @height.setter
    def height(self, height):
        """
        Sets this object's height

        Note: Can only be called if the primitive type is one of {"Cone", "Cylinder"}

        Args:
            height (float): height to set
        """
        assert_valid_key(key=self._primitive_type, valid_keys=VALID_HEIGHT_OBJECTS, name="primitive object with height")
        original_scale = self.scale

        # TODO (eric): currently scale [1, 1, 1] will have height 2.0
        original_scale[2] = height / 2.0
        self.scale = original_scale

    @property
    def size(self):
        """
        Gets this object's size, if it exists.

        Note: Can only be called if the primitive type is one of {"Cube", "Torus"}

        Returns:
            float: size for this object
        """
        assert_valid_key(key=self._primitive_type, valid_keys=VALID_SIZE_OBJECTS, name="primitive object with size")

        # TODO (eric): currently scale [1, 1, 1] will have size 2.0
        return self.scale[0] * 2.0

    @size.setter
    def size(self, size):
        """
        Sets this object's size

        Note: Can only be called if the primitive type is one of {"Cube", "Torus"}

        Args:
            size (float): size to set
        """
        assert_valid_key(key=self._primitive_type, valid_keys=VALID_SIZE_OBJECTS, name="primitive object with size")
        original_scale = self.scale

        # TODO (eric): currently scale [1, 1, 1] will have size 2.0
        original_scale[:] = size / 2.0
        self.scale = original_scale

    def _create_prim_with_same_kwargs(self, prim_path, name, load_config):
        # Add additional kwargs (fit_avg_dim_volume and bounding_box are already captured in load_config)
        return self.__class__(
            prim_path=prim_path,
            primitive_type=self._primitive_type,
            name=name,
            category=self.category,
            class_id=self.class_id,
            scale=self.scale,
            rendering_params=self.rendering_params,
            visible=self.visible,
            fixed_base=self.fixed_base,
            prim_type=self._prim_type,
            load_config=load_config,
            abilities=self._abilities,
            visual_only=self._visual_only,
        )
