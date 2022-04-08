import logging
import numpy as np
from igibson.objects.stateful_object import StatefulObject

from pxr import Gf, Usd, Sdf, UsdGeom, UsdPhysics, PhysxSchema, UsdShade
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage


PRIMITIVE_OBJECTS = {
    "Cube",
    "Sphere",
    "Cylinder",
    "Capsule",
    "Cone",
}


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
        scale=None,
        rendering_params=None,
        visible=True,
        fixed_base=False,
        visual_only=False,
        self_collisions=False,
        load_config=None,
        abilities=None,
        rgba=(0.5, 0.5, 0.5, 1.0),
        **kwargs,
    ):
        """
        @param prim_path: str, global path in the stage to this object
        @param primitive_type: str, type of primitive object to create. Should be one of:
            {Cube, Sphere, Cylinder, Capsule, Cone}
        @param name: Name for the object. Names need to be unique per scene. If no name is set, a name will be generated
            at the time the object is added to the scene, using the object's category.
        @param category: Category for the object. Defaults to "object".
        @param class_id: What class ID the object should be assigned in semantic segmentation rendering mode.
        @param scale: float or 3-array, sets the scale for this object. A single number corresponds to uniform scaling
            along the x,y,z axes, whereas a 3-array specifies per-axis scaling.
        @param rendering_params: Any relevant rendering settings for this object.
        @param visible: bool, whether to render this object or not in the stage
        @param fixed_base: bool, whether to fix the base of this object or not
        @param visual_only: whether this object should be a visual-only (i.e.: not affected by collisions / gravity)
            object or not
        self_collisions (bool): Whether to enable self collisions for this object
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
        kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
            for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
        # Compose load config and add rgba values
        load_config = dict() if load_config is None else load_config
        load_config["color"] = np.array(rgba[:3])
        load_config["opacity"] = rgba[3]

        super().__init__(
            prim_path=prim_path,
            name=name,
            category=category,
            class_id=class_id,
            scale=scale,
            rendering_params=rendering_params,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            self_collisions=self_collisions,
            load_config=load_config,
            abilities=abilities,
            **kwargs,
        )
        # Make sure primitive type is valid
        assert primitive_type in PRIMITIVE_OBJECTS, f"Invalid primitive object requested! Valid options are: " \
                                                    f"{PRIMITIVE_OBJECTS}, got: {primitive_type}"
        self._primitive_type = primitive_type

    def _load(self, simulator=None):
        """
        Load the object into pybullet and set it to the correct pose
        """
        logging.info(f"Loading the following primitive: {self._primitive_type}")

        # Define an Xform at the specified path
        stage = get_current_stage()
        prim = stage.DefinePrim(self._prim_path, "Xform")

        # Define a nested mesh corresponding to the root link for this prim
        base_link = stage.DefinePrim(f"{self._prim_path}/base_link", "Xform")

        # Define (finally!) nested meshes corresponding to visual / collision mesh
        vis_prim = UsdGeom.__dict__[self._primitive_type].Define(stage, f"{self._prim_path}/base_link/visual").GetPrim()
        col_prim = UsdGeom.__dict__[self._primitive_type].Define(stage, f"{self._prim_path}/base_link/collision").GetPrim()

        # Add collision API to collision geom
        UsdPhysics.CollisionAPI.Apply(col_prim)
        UsdPhysics.MeshCollisionAPI.Apply(col_prim)
        PhysxSchema.PhysxCollisionAPI.Apply(col_prim)

        return prim

    def _post_load(self, simulator=None):
        # Run super first
        super()._post_load(simulator=simulator)

        # Set color and opacity
        for mesh in self._links["base_link"].visual_meshes.values():
            mesh.color = self._load_config["color"]
            mesh.opacity = self._load_config["opacity"]

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
            load_config=load_config,
            abilities=self._abilities,
            visual_only=self._visual_only,
        )
