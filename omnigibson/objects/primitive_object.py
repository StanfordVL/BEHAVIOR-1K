import logging
import numpy as np
from omnigibson.objects.stateful_object import StatefulObject
from omnigibson.utils.python_utils import assert_valid_key

from pxr import Gf, Usd, Sdf, Vt, UsdGeom, UsdPhysics, PhysxSchema, UsdShade
from omni.isaac.core.utils.prims import get_prim_at_path
from omnigibson.utils.constants import PrimType, PRIMITIVE_MESH_TYPES
from omnigibson.utils.usd_utils import create_primitive_mesh
from omnigibson.utils.render_utils import create_pbr_material
from omnigibson.utils.physx_utils import bind_material
import omni
import carb


# Define valid objects that can be created
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
        load_config["radius"] = radius
        load_config["height"] = height
        load_config["size"] = size

        # Initialize other internal variables
        self._vis_geom = None
        self._col_geom = None
        self._extents = np.ones(3)            # (x,y,z extents)

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
            self._vis_geom = create_primitive_mesh(prim_path=f"{self._prim_path}/base_link/visual", primitive_type=self._primitive_type)
            self._col_geom = create_primitive_mesh(prim_path=f"{self._prim_path}/base_link/collision", primitive_type=self._primitive_type)

            # Add collision API to collision geom
            UsdPhysics.CollisionAPI.Apply(self._col_geom.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(self._col_geom.GetPrim())
            PhysxSchema.PhysxCollisionAPI.Apply(self._col_geom.GetPrim())

        elif self._prim_type == PrimType.CLOTH:
            # For Cloth, the base link itself is a cloth mesh
            # TODO (eric): configure u_patches and v_patches
            self._vis_geom = create_primitive_mesh(
                prim_path=f"{self._prim_path}/base_link",
                primitive_type=self._primitive_type,
                u_patches=None,
                v_patches=None,
            )
            self._col_geom = None

        # Create a material for this object for the base link
        stage.DefinePrim(f"{self._prim_path}/Looks", "Scope")
        mat_path = f"{self._prim_path}/Looks/default"
        mat = create_pbr_material(prim_path=mat_path)
        bind_material(prim_path=self._vis_geom.GetPrim().GetPrimPath().pathString, material_path=mat_path)

        return prim

    def _post_load(self):
        # Run super first
        super()._post_load()

        if self._prim_type == PrimType.RIGID:
            visual_geom_prim = list(self.links["base_link"].visual_meshes.values())[0]
        elif self._prim_type == PrimType.CLOTH:
            visual_geom_prim = self.links["base_link"]
        else:
            raise ValueError("Prim type must either be PrimType.RIGID or PrimType.CLOTH for loading a primitive object")

        visual_geom_prim.color = self._load_config["color"]
        visual_geom_prim.opacity = self._load_config["opacity"]

        # Update collision approximation
        self.root_link.collision_meshes["collision"].set_collision_approximation("convexHull")

        # Possibly set scalings (only if the scale value is not set)
        if self._load_config["scale"] is not None:
            logging.warning("Custom scale specified for primitive object, so ignoring radius, height, and size arguments!")
        else:
            if self._load_config["radius"] is not None:
                self.radius = self._load_config["radius"]
            if self._load_config["height"] is not None:
                self.height = self._load_config["height"]
            if self._load_config["size"] is not None:
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
        return self._extents[0] / 2.0

    @radius.setter
    def radius(self, radius):
        """
        Sets this object's radius

        Note: Can only be called if the primitive type is one of {"Cone", "Cylinder", "Disk", "Sphere"}

        Args:
            radius (float): radius to set
        """
        assert_valid_key(key=self._primitive_type, valid_keys=VALID_RADIUS_OBJECTS, name="primitive object with radius")
        original_extent = self._extents
        attr_pairs = []
        for geom in self._vis_geom, self._col_geom:
            if geom is not None:
                for attr in (geom.GetPointsAttr(), geom.GetNormalsAttr()):
                    vals = np.array(attr.Get()).astype(np.float64)
                    attr_pairs.append([attr, vals])

        # Calculate how much to scale extents by and then modify the points / normals accordingly
        scaling_factor = 2.0 * radius / original_extent[0]
        for attr, vals in attr_pairs:
            # If this is a sphere, modify all 3 axes
            if self._primitive_type == "Sphere":
                vals = vals * scaling_factor
            # Otherwise, just modify the first two dimensions
            else:
                vals[:, :2] = vals[:, :2] * scaling_factor
            # Set the value
            attr.Set(Vt.Vec3fArray([Gf.Vec3f(*v) for v in vals]))

        # Update the extents variable
        self._extents = np.ones(3) * radius * 2.0 if self._primitive_type == "Sphere" else \
            np.array([radius * 2.0, radius * 2.0, self._extents[2]])

    @property
    def height(self):
        """
        Gets this object's height, if it exists.

        Note: Can only be called if the primitive type is one of {"Cone", "Cylinder"}

        Returns:
            float: height for this object
        """
        assert_valid_key(key=self._primitive_type, valid_keys=VALID_HEIGHT_OBJECTS, name="primitive object with height")
        return self._extents[2]

    @height.setter
    def height(self, height):
        """
        Sets this object's height

        Note: Can only be called if the primitive type is one of {"Cone", "Cylinder"}

        Args:
            height (float): height to set
        """
        assert_valid_key(key=self._primitive_type, valid_keys=VALID_HEIGHT_OBJECTS, name="primitive object with height")
        original_extent = self._extents

        # Calculate the correct scaling factor and scale the points and normals appropriately
        scaling_factor = height / original_extent[2]
        for geom in self._vis_geom, self._col_geom:
            if geom is not None:
                for attr in (geom.GetPointsAttr(), geom.GetNormalsAttr()):
                    vals = np.array(attr.Get()).astype(np.float64)
                    # Scale the z axis by the scaling factor
                    vals[:, 2] = vals[:, 2] * scaling_factor
                    attr.Set(Vt.Vec3fArray([Gf.Vec3f(*v) for v in vals]))

        # Update the extents variable
        self._extents[2] = height

    @property
    def size(self):
        """
        Gets this object's size, if it exists.

        Note: Can only be called if the primitive type is one of {"Cube", "Torus"}

        Returns:
            float: size for this object
        """
        assert_valid_key(key=self._primitive_type, valid_keys=VALID_SIZE_OBJECTS, name="primitive object with size")
        return self._extents[0]

    @size.setter
    def size(self, size):
        """
        Sets this object's size

        Note: Can only be called if the primitive type is one of {"Cube", "Torus"}

        Args:
            size (float): size to set
        """
        assert_valid_key(key=self._primitive_type, valid_keys=VALID_SIZE_OBJECTS, name="primitive object with size")

        original_extent = self._extents

        # Calculate the correct scaling factor and scale the points and normals appropriately
        scaling_factor = size / original_extent[0]
        for geom in self._vis_geom, self._col_geom:
            if geom is not None:
                for attr in (geom.GetPointsAttr(), geom.GetNormalsAttr()):
                    # Scale all three axes by the scaling factor
                    vals = np.array(attr.Get()).astype(np.float64) * scaling_factor
                    attr.Set(Vt.Vec3fArray([Gf.Vec3f(*v) for v in vals]))

        # Update the extents variable
        self._extents = np.ones(3) * size

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

    def _dump_state(self):
        state = super()._dump_state()
        # state["extents"] = self._extents
        state["radius"] = self.radius if self._primitive_type in VALID_RADIUS_OBJECTS else -1
        state["height"] = self.height if self._primitive_type in VALID_HEIGHT_OBJECTS else -1
        state["size"] = self.size if self._primitive_type in VALID_SIZE_OBJECTS else -1
        return state

    def _load_state(self, state):
        super()._load_state(state=state)
        # self._extents = np.array(state["extents"])
        if self._primitive_type in VALID_RADIUS_OBJECTS:
            self.radius = state["radius"]
        if self._primitive_type in VALID_HEIGHT_OBJECTS:
            self.height = state["height"]
        if self._primitive_type in VALID_SIZE_OBJECTS:
            self.size = state["size"]

    def _deserialize(self, state):
        state_dict, idx = super()._deserialize(state=state)
        # state_dict["extents"] = state[idx: idx + 3]
        state_dict["radius"] = state[idx]
        state_dict["height"] = state[idx + 1]
        state_dict["size"] = state[idx + 2]
        return state_dict, idx + 3

    def _serialize(self, state):
        # Run super first
        state_flat = super()._serialize(state=state)

        return np.concatenate([
            state_flat,
            np.array([state["radius"], state["height"], state["size"]]),
        ]).astype(float)
