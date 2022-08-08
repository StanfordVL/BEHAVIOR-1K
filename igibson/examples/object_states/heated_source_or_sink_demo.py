import numpy as np
from igibson import object_states, ig_dataset_path
from igibson.objects.dataset_object import DatasetObject
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from omni.isaac.core.utils.viewports import set_camera_view

# Import scene.
sim = Simulator()
scene = EmptyScene(floor_plane_visible=True)
sim.import_scene(scene)

sim._set_viewer_camera("/OmniverseKit_Persp")
set_camera_view(eye=[3, -4, 4], target=[0, 1, 0])

# Import object.
obj_category = "stove"
obj_model = "101908"
name = "stove"

model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
usd_path = f"{model_root_path}/usd/{obj_model}.usd"

stove = DatasetObject(
    prim_path=f"/World/{name}",
    usd_path=usd_path,
    category=obj_category,
    name=f"{name}",
    abilities={"heatSource": {"requires_toggled_on": True}, "toggleable": {},},
)

assert object_states.HeatSourceOrSink in stove.states
assert object_states.ToggledOn in stove.states

sim.import_object(stove, auto_initialize=True)

sim.stop()
stove.set_position(np.array([0, 0, 0.4]))
sim.play()

# Hide visibility of all toggle markers.
for link in stove.states[object_states.ToggledOn].visual_marker_on.links.values():
    link.visible = False
for link in stove.states[object_states.ToggledOn].visual_marker_off.links.values():
    link.visible = False

for _ in range(200):
    sim.step()

# Heat source is off.
heat_source_state, heat_source_position = stove.states[object_states.HeatSourceOrSink].get_value()
assert not heat_source_state

# Toggle on stove.
stove.states[object_states.ToggledOn].set_value(True)
assert stove.states[object_states.ToggledOn].get_value()

# Need to take a step to update the state.
sim.step()

# Heat source is on.
heat_source_state, heat_source_position = stove.states[object_states.HeatSourceOrSink].get_value()
assert heat_source_state
for _ in range(500):
    sim.step()

# Toggle off stove.
stove.states[object_states.ToggledOn].set_value(False)
assert not stove.states[object_states.ToggledOn].get_value()
for _ in range(200):
    sim.step()

# Move stove.
stove.set_linear_velocity(velocity=np.array([-5.0, 0, 0.0]))
for i in range(200):
    sim.step()

# Toggle on stove again.
stove.states[object_states.ToggledOn].set_value(True)
assert stove.states[object_states.ToggledOn].get_value()
for i in range(500):
    sim.step()