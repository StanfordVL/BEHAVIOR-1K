import omnigibson as og
import omnigibson.lazy as lazy


ROOM_LIGHT_CATEGORIES = {
    "downlight",
    "room_light",
    "wall_mounted_light",
}
SELF_LIGHT_CATEGORIES = {
    "floor_lamp",
    "table_lamp",
}
LIGHT_CONTROL_CATEGORIES = {
    "electric_switch",
    *SELF_LIGHT_CATEGORIES,
}


def _iter_descendant_prims(prim):
    for child in prim.GetChildren():
        yield child
        yield from _iter_descendant_prims(child)


def _set_light_prims_visible(obj, visible):
    with og.sim.editing_usd():
        for prim in _iter_descendant_prims(obj.prim):
            if "Light" not in prim.GetTypeName():
                continue
            imageable = lazy.pxr.UsdGeom.Imageable(prim)
            if visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()


def _sync_room_lights_for_switch(scene, switch, visible):
    rooms = switch.in_rooms or []
    for room in rooms:
        for obj in scene.object_registry("in_rooms", room, default_val=[]):
            if getattr(obj, "category", None) in ROOM_LIGHT_CATEGORIES:
                obj.visible = visible


def sync_light_visibility_from_toggles(scene):
    from omnigibson.object_states import ToggledOn

    for obj in scene.objects:
        category = getattr(obj, "category", None)
        states = getattr(obj, "states", None)
        if category not in LIGHT_CONTROL_CATEGORIES or states is None or ToggledOn not in states:
            continue

        visible = states[ToggledOn].value
        if category == "electric_switch":
            _sync_room_lights_for_switch(scene, obj, visible)
        else:
            _set_light_prims_visible(obj, visible)


def set_light_visibility_from_default(scene, visible=True):
    for obj in scene.objects:
        category = getattr(obj, "category", None)
        if category in ROOM_LIGHT_CATEGORIES:
            obj.visible = visible
        elif category in SELF_LIGHT_CATEGORIES:
            _set_light_prims_visible(obj, visible)


def set_light_control_toggles(objects, value):
    from omnigibson.object_states import ToggledOn

    for obj in objects:
        category = getattr(obj, "category", None)
        states = getattr(obj, "states", None)
        if obj is None or category not in LIGHT_CONTROL_CATEGORIES or states is None or ToggledOn not in states:
            continue
        states[ToggledOn].set_value(value)


class LightToggleSynchronizer:
    def __init__(self, scene):
        self.scene = scene
        self._toggle_values = {}
        self._target_visibility = {}

    def reset_from_current_state(self):
        self._toggle_values = self._get_toggle_values()
        self._target_visibility = {}
        set_light_visibility_from_default(self.scene, visible=True)
        for obj in self.scene.objects:
            category = getattr(obj, "category", None)
            if category in ROOM_LIGHT_CATEGORIES or category in SELF_LIGHT_CATEGORIES:
                self._target_visibility[obj.name] = True

    def sync_from_current_state(self):
        if not self._toggle_values:
            self.reset_from_current_state()
            return

        current_toggle_values = self._get_toggle_values()
        for obj_name, value in current_toggle_values.items():
            if self._toggle_values.get(obj_name) != value:
                self._toggle_targets(obj_name)
        self._toggle_values = current_toggle_values

    def _get_toggle_values(self):
        from omnigibson.object_states import ToggledOn

        values = {}
        for obj in self.scene.objects:
            category = getattr(obj, "category", None)
            states = getattr(obj, "states", None)
            if category not in LIGHT_CONTROL_CATEGORIES or states is None or ToggledOn not in states:
                continue
            values[obj.name] = states[ToggledOn].value
        return values

    def _toggle_targets(self, obj_name):
        obj = self.scene.object_registry("name", obj_name, None)
        if obj is None:
            return

        category = getattr(obj, "category", None)
        if category == "electric_switch":
            for room in obj.in_rooms or []:
                for light in self.scene.object_registry("in_rooms", room, default_val=[]):
                    if getattr(light, "category", None) in ROOM_LIGHT_CATEGORIES:
                        self._toggle_light(light)
        elif category in SELF_LIGHT_CATEGORIES:
            self._toggle_light(obj)

    def _toggle_light(self, obj):
        visible = not self._target_visibility.get(obj.name, True)
        self._target_visibility[obj.name] = visible
        if getattr(obj, "category", None) in ROOM_LIGHT_CATEGORIES:
            obj.visible = visible
        else:
            _set_light_prims_visible(obj, visible)
