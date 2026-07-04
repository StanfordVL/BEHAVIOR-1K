"""Newton-native primitive object descriptor."""

from __future__ import annotations

import tempfile
from pathlib import Path

from omnigibson.objects.usd_object import REGISTERED_OBJECTS, USDObject


PRIMITIVE_MESH_TYPES = {"Cone", "Cube", "Cylinder", "Disk", "Plane", "Sphere", "Torus"}


class PrimitiveObject(USDObject):
    """Object descriptor for simple generated USD primitives."""

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
        prim_type=None,
        link_physics_materials=None,
        load_config=None,
        abilities=None,
        include_default_states=True,
        rgba=(0.5, 0.5, 0.5, 1.0),
        radius=None,
        height=None,
        size=None,
        position=None,
        orientation=None,
        **kwargs,
    ):
        if primitive_type not in PRIMITIVE_MESH_TYPES:
            raise ValueError(
                f"Invalid primitive_type {primitive_type!r}; expected one of {sorted(PRIMITIVE_MESH_TYPES)}."
            )
        self.primitive_type = primitive_type
        self.rgba = tuple(float(v) for v in rgba)
        self.radius = radius
        self.height = height
        self.size = size

        load_config = {} if load_config is None else dict(load_config)
        load_config.update(
            {"color": self.rgba[:3], "opacity": self.rgba[3], "radius": radius, "height": height, "size": size}
        )

        super().__init__(
            name=name,
            usd_path=_build_primitive_usd(name=name, primitive_type=primitive_type),
            relative_prim_path=relative_prim_path,
            category=category,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            kinematic_only=kinematic_only,
            self_collisions=self_collisions,
            prim_type=prim_type,
            link_physics_materials=link_physics_materials,
            load_config=load_config,
            abilities=abilities,
            include_default_states=include_default_states,
            position=position,
            orientation=orientation,
            **kwargs,
        )
        self.object_type = "PrimitiveObject"


def _build_primitive_usd(name, primitive_type):
    from pxr import Gf, Usd, UsdPhysics

    temp_dir = Path(tempfile.mkdtemp(prefix=f"og-newton-primitive-{name}-"))
    usd_path = temp_dir / f"{name}.usd"
    stage = Usd.Stage.CreateNew(str(usd_path))
    root = stage.DefinePrim("/object", "Xform")
    stage.SetDefaultPrim(root)
    link = stage.DefinePrim("/object/base_link", "Xform")
    UsdPhysics.RigidBodyAPI.Apply(link)
    mass_api = UsdPhysics.MassAPI.Apply(link)
    mass_api.GetMassAttr().Set(1.0)
    mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(0.1, 0.1, 0.1))

    geom_path = "/object/base_link/visuals"
    collision_path = "/object/base_link/collisions"
    _define_geom(stage, geom_path, primitive_type)
    collision = _define_geom(stage, collision_path, primitive_type)
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
    stage.Save()
    del stage
    return usd_path


def _define_geom(stage, path, primitive_type):
    from pxr import UsdGeom

    if primitive_type == "Sphere":
        return UsdGeom.Sphere.Define(stage, path)
    if primitive_type == "Cylinder":
        return UsdGeom.Cylinder.Define(stage, path)
    if primitive_type == "Cone":
        return UsdGeom.Cone.Define(stage, path)
    if primitive_type == "Disk":
        return UsdGeom.Cylinder.Define(stage, path)
    if primitive_type == "Plane":
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr([(-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0)])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        return mesh
    return UsdGeom.Cube.Define(stage, path)


REGISTERED_OBJECTS["PrimitiveObject"] = PrimitiveObject
