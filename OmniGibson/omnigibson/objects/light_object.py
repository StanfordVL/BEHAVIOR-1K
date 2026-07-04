"""Newton-native light object descriptor."""

from __future__ import annotations

from omnigibson.objects.usd_object import REGISTERED_OBJECTS, USDObject


class LightObject(USDObject):
    """Scene light declaration with the original object constructor shape."""

    LIGHT_TYPES = {"Cylinder", "Disk", "Distant", "Dome", "Geometry", "Rect", "Sphere"}

    def __init__(
        self,
        name,
        light_type="Sphere",
        relative_prim_path=None,
        category="light",
        scale=None,
        link_physics_materials=None,
        load_config=None,
        abilities=None,
        include_default_states=True,
        radius=1.0,
        intensity=50000.0,
        position=None,
        orientation=None,
        **kwargs,
    ):
        if light_type not in self.LIGHT_TYPES:
            raise ValueError(f"Invalid light_type {light_type!r}; expected one of {sorted(self.LIGHT_TYPES)}.")
        self.light_type = light_type
        self.radius = radius
        self.intensity = intensity

        load_config = {} if load_config is None else dict(load_config)
        load_config.update({"radius": radius, "intensity": intensity})

        super().__init__(
            name=name,
            usd_path=None,
            relative_prim_path=relative_prim_path,
            category=category,
            scale=scale,
            visible=True,
            fixed_base=True,
            visual_only=True,
            kinematic_only=True,
            self_collisions=False,
            prim_type=None,
            link_physics_materials=link_physics_materials,
            load_config=load_config,
            abilities=abilities,
            include_default_states=include_default_states,
            position=position or (0.0, 0.0, 2.0),
            orientation=orientation,
            **kwargs,
        )
        self.object_type = "LightObject"

    @property
    def kind(self):
        return "light"


REGISTERED_OBJECTS["LightObject"] = LightObject
