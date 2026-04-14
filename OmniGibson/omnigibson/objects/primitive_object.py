import os
import tempfile

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.objects.usd_object import USDObject
from omnigibson.utils.constants import PRIMITIVE_MESH_TYPES, PrimType
from omnigibson.utils.physx_utils import bind_material
from omnigibson.utils.python_utils import assert_valid_key
from omnigibson.utils.render_utils import create_pbr_material
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.usd_utils import create_primitive_mesh, create_usd_stage, ensure_usd_api

# Create module logger
log = create_module_logger(module_name=__name__)


# Define valid objects that can be created
VALID_RADIUS_OBJECTS = {"Cone", "Cylinder", "Disk", "Sphere"}
VALID_HEIGHT_OBJECTS = {"Cone", "Cylinder"}
VALID_SIZE_OBJECTS = {"Cube", "Torus"}


class PrimitiveObject(USDObject):
    """
    PrimitiveObjects are objects defined by a single geom, e.g: sphere, mesh, cube, etc.
    """

    def __init__(
        self,
        name,
        primitive_type,
        relative_prim_path=None,
        category="object",
        scale=None,
        visible=True,
        fixed_base=False,
        visual_only=False,
        kinematic_only=None,
        self_collisions=False,
        prim_type=PrimType.RIGID,
        link_physics_materials=None,
        load_config=None,
        abilities=None,
        include_default_states=True,
        rgba=(0.5, 0.5, 0.5, 1.0),
        radius=None,
        height=None,
        size=None,
        **kwargs,
    ):
        """
        Args:
            name (str): Name for the object. Names need to be unique per scene
            primitive_type (str): type of primitive object to create. Should be one of:
                {"Cone", "Cube", "Cylinder", "Disk", "Plane", "Sphere", "Torus"}
            relative_prim_path (None or str): The path relative to its scene prim for this object. If not specified, it defaults to /<name>.
            category (str): Category for the object. Defaults to "object".
            scale (None or float or 3-array): if specified, sets either the uniform (float) or x,y,z (3-array) scale
                for this object. A single number corresponds to uniform scaling along the x,y,z axes, whereas a
                3-array specifies per-axis scaling.
            visible (bool): whether to render this object or not in the stage
            fixed_base (bool): whether to fix the base of this object or not
            visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
            kinematic_only (None or bool): Whether this object should be kinematic only (and not get affected by any
                collisions). If None, then this value will be set to True if @fixed_base is True and some other criteria
                are satisfied (see usd_object.py post_load function), else False.
            self_collisions (bool): Whether to enable self collisions for this object
            prim_type (PrimType): Which type of prim the object is, Valid options are: {PrimType.RIGID, PrimType.CLOTH}
            link_physics_materials (None or dict): If specified, dictionary mapping link name to kwargs used to generate
                a specific physical material for that link's collision meshes, where the kwargs are arguments directly
                passed into the isaacsim.core.api.materials.physics_material.PhysicsMaterial constructor, e.g.: "static_friction",
                "dynamic_friction", and "restitution"
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime.
            abilities (None or dict): If specified, manually adds specific object states to this object. It should be
                a dict in the form of {ability: {param: value}} containing object abilities and parameters to pass to
                the object state instance constructor.rgba (4-array): (R, G, B, A) values to set for this object
            include_default_states (bool): whether to include the default object states from @get_default_states
            radius (None or float): If specified, sets the radius for this object. This value is scaled by @scale
                Note: Should only be specified if the @primitive_type is one of {"Cone", "Cylinder", "Disk", "Sphere"}
            height (None or float): If specified, sets the height for this object. This value is scaled by @scale
                Note: Should only be specified if the @primitive_type is one of {"Cone", "Cylinder"}
            size (None or float): If specified, sets the size for this object. This value is scaled by @scale
                Note: Should only be specified if the @primitive_type is one of {"Cube", "Torus"}
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject).
        """
        # Compose load config and add rgba values
        load_config = dict() if load_config is None else load_config
        load_config["color"] = rgba[:3]
        load_config["opacity"] = rgba[3]
        load_config["radius"] = radius
        load_config["height"] = height
        load_config["size"] = size

        # Initialize other internal variables
        self._vis_geom = None
        self._col_geom = None
        self._extents = th.ones(3)  # (x,y,z extents)

        # Make sure primitive type is valid
        assert_valid_key(key=primitive_type, valid_keys=PRIMITIVE_MESH_TYPES, name="primitive mesh type")
        self._primitive_type = primitive_type

        # Build the USD for this primitive upfront and pass it to USDObject
        usd_path = self._build_usd(name=name, primitive_type=primitive_type)

        super().__init__(
            usd_path=usd_path,
            relative_prim_path=relative_prim_path,
            name=name,
            category=category,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            kinematic_only=kinematic_only,
            self_collisions=self_collisions,
            prim_type=prim_type,
            include_default_states=include_default_states,
            link_physics_materials=link_physics_materials,
            load_config=load_config,
            abilities=abilities,
            **kwargs,
        )

    @staticmethod
    def _build_usd(name, primitive_type):
        """
        Build a temporary USD containing the primitive mesh structure and return its path.
        Material creation is deferred to _post_load().
        """
        tempdir_path = tempfile.mkdtemp(name, dir=og.tempdir)
        usd_path = os.path.join(tempdir_path, f"{name}.usd")
        side_stage = create_usd_stage(usd_path)
        root = side_stage.DefinePrim("/object", "Xform")
        side_stage.SetDefaultPrim(root)
        side_stage.DefinePrim("/object/base_link", "Xform")

        create_primitive_mesh(prim_path="/object/base_link/visuals", primitive_type=primitive_type, stage=side_stage)
        col_geom = create_primitive_mesh(
            prim_path="/object/base_link/collisions", primitive_type=primitive_type, stage=side_stage
        )
        ensure_usd_api(col_geom.GetPrim(), lazy.pxr.UsdPhysics.CollisionAPI)
        ensure_usd_api(col_geom.GetPrim(), lazy.pxr.UsdPhysics.MeshCollisionAPI)
        ensure_usd_api(col_geom.GetPrim(), lazy.pxr.PhysxSchema.PhysxCollisionAPI)

        side_stage.Save()
        del side_stage
        return usd_path

    def _post_load(self):
        self._vis_geom = lazy.pxr.UsdGeom.Mesh(og.sim.stage.GetPrimAtPath(f"{self.prim_path}/base_link/visuals"))
        self._col_geom = lazy.pxr.UsdGeom.Mesh(og.sim.stage.GetPrimAtPath(f"{self.prim_path}/base_link/collisions"))

        # Create a material and bind it to the visual geom.
        # This is done here rather than in _prepare_to_load() because create_pbr_material
        # and bind_material both go through omni.kit.commands, which operates on the
        # active stage.
        with og.sim.editing_usd():
            og.sim.stage.DefinePrim(f"{self.prim_path}/Looks", "Scope")
        mat_path = f"{self.prim_path}/Looks/default"
        create_pbr_material(prim_path=mat_path)
        bind_material(prim_path=self._vis_geom.GetPrim().GetPrimPath().pathString, material_path=mat_path)

        # Possibly set scalings (only if the scale value is not set)
        if self._load_config["scale"] is not None:
            log.warning("Custom scale specified for primitive object, so ignoring radius, height, and size arguments!")
        else:
            if self._load_config["radius"] is not None:
                self.radius = self._load_config["radius"]
            if self._load_config["height"] is not None:
                self.height = self._load_config["height"]
            if self._load_config["size"] is not None:
                self.size = self._load_config["size"]

        # This step might will perform cloth remeshing if self._prim_type == PrimType.CLOTH.
        # Therefore, we need to apply size, radius, and height before this to scale the points properly.
        super()._post_load()

        # Cloth primitive does not have collision meshes
        if self._prim_type != PrimType.CLOTH:
            # Set the collision approximation appropriately
            if self._primitive_type == "Sphere":
                col_approximation = "boundingSphere"
            elif self._primitive_type == "Cube":
                col_approximation = "boundingCube"
            else:
                col_approximation = "convexHull"
            self.root_link.set_collision_approximation(col_approximation)

    def _initialize(self):
        # Run super first
        super()._initialize()

        # Set color and opacity
        if self._prim_type == PrimType.RIGID:
            visual_geom_prim = list(self.root_link.visual_meshes.values())[0]
        elif self._prim_type == PrimType.CLOTH:
            visual_geom_prim = self.root_link
        else:
            raise ValueError("Prim type must either be PrimType.RIGID or PrimType.CLOTH for loading a primitive object")

        visual_geom_prim.color = self._load_config["color"]
        visual_geom_prim.opacity = (
            self._load_config["opacity"].item()
            if isinstance(self._load_config["opacity"], th.Tensor)
            else self._load_config["opacity"]
        )

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
        # Update the extents variable
        original_extent = self._extents.clone()
        self._extents = (
            th.ones(3) * radius * 2.0
            if self._primitive_type == "Sphere"
            else th.tensor([radius * 2.0, radius * 2.0, self._extents[2]])
        )
        attr_pairs = []
        with og.sim.editing_usd():
            for geom in self._vis_geom, self._col_geom:
                if geom is not None:
                    for attr in (geom.GetPointsAttr(), geom.GetNormalsAttr()):
                        vals = th.tensor(attr.Get()).double()
                        attr_pairs.append([attr, vals])
                    geom.GetExtentAttr().Set(
                        lazy.pxr.Vt.Vec3fArray(
                            [
                                lazy.pxr.Gf.Vec3f(*(-self._extents / 2.0).tolist()),
                                lazy.pxr.Gf.Vec3f(*(self._extents / 2.0).tolist()),
                            ]
                        )
                    )

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
                attr.Set(lazy.pxr.Vt.Vec3fArray([lazy.pxr.Gf.Vec3f(*v.tolist()) for v in vals]))

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
        # Update the extents variable
        original_extent = self._extents.clone()
        self._extents[2] = height

        # Calculate the correct scaling factor and scale the points and normals appropriately
        scaling_factor = height / original_extent[2]
        with og.sim.editing_usd():
            for geom in self._vis_geom, self._col_geom:
                if geom is not None:
                    for attr in (geom.GetPointsAttr(), geom.GetNormalsAttr()):
                        vals = th.tensor(attr.Get()).double()
                        # Scale the z axis by the scaling factor
                        vals[:, 2] = vals[:, 2] * scaling_factor
                        attr.Set(lazy.pxr.Vt.Vec3fArray([lazy.pxr.Gf.Vec3f(*v) for v in vals.tolist()]))
                    geom.GetExtentAttr().Set(
                        lazy.pxr.Vt.Vec3fArray(
                            [
                                lazy.pxr.Gf.Vec3f(*(-self._extents / 2.0).tolist()),
                                lazy.pxr.Gf.Vec3f(*(self._extents / 2.0).tolist()),
                            ]
                        )
                    )

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

        # Update the extents variable
        original_extent = self._extents.clone()
        self._extents = th.ones(3) * size

        # Calculate the correct scaling factor and scale the points and normals appropriately
        scaling_factor = size / original_extent[0]
        with og.sim.editing_usd():
            for geom in self._vis_geom, self._col_geom:
                if geom is not None:
                    for attr in (geom.GetPointsAttr(), geom.GetNormalsAttr()):
                        # Scale all three axes by the scaling factor
                        vals = th.tensor(attr.Get()).double() * scaling_factor
                        attr.Set(lazy.pxr.Vt.Vec3fArray([lazy.pxr.Gf.Vec3f(*v.tolist()) for v in vals]))
                    geom.GetExtentAttr().Set(
                        lazy.pxr.Vt.Vec3fArray(
                            [
                                lazy.pxr.Gf.Vec3f(*(-self._extents / 2.0).tolist()),
                                lazy.pxr.Gf.Vec3f(*(self._extents / 2.0).tolist()),
                            ]
                        )
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
        # self._extents = th.tensor(state["extents"])
        if self._primitive_type in VALID_RADIUS_OBJECTS:
            self.radius = state["radius"]
        if self._primitive_type in VALID_HEIGHT_OBJECTS:
            self.height = state["height"]
        if self._primitive_type in VALID_SIZE_OBJECTS:
            self.size = state["size"]

    def deserialize(self, state):
        state_dict, idx = super().deserialize(state=state)
        # state_dict["extents"] = state[idx: idx + 3]
        state_dict["radius"] = state[idx]
        state_dict["height"] = state[idx + 1]
        state_dict["size"] = state[idx + 2]
        return state_dict, idx + 3

    def serialize(self, state):
        # Run super first
        state_flat = super().serialize(state=state)

        return th.cat(
            [
                state_flat,
                th.tensor([state["radius"], state["height"], state["size"]]),
            ]
        )
