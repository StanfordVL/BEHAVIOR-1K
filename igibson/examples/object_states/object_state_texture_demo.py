from igibson import object_states, ig_dataset_path
from igibson.objects.dataset_object import DatasetObject
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator_omni import Simulator
from igibson.object_states.temperature import DEFAULT_TEMPERATURE

# Import scene.
sim = Simulator()
scene = EmptyScene(floor_plane_visible=True)
sim.import_scene(scene)

# Import object.
obj_category = "bottom_cabinet"
obj_model = "45087"
name = "bottom_cabinet"

model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
usd_path = f"{model_root_path}/usd/{obj_model}.usd"

obj = DatasetObject(
    prim_path=f"/World/{name}",
    usd_path=usd_path,
    category=obj_category,
    name=name,
    abilities={"freezable": {}, "cookable": {}, "burnable": {}, "soakable": {}, "toggleable": {}},
)

assert object_states.Frozen in obj.states
assert object_states.Cooked in obj.states
assert object_states.Burnt in obj.states
assert object_states.Soaked in obj.states
assert object_states.ToggledOn in obj.states

sim.import_object(obj, auto_initialize=True)
sim.play()
sim.step()

def report_states():
    # Make sure the state is updated.
    for _ in range(5):
        sim.step()
    print("=" * 20)
    print("temperature:", obj.states[object_states.Temperature].get_value())
    print("obj is frozen:", obj.states[object_states.Frozen].get_value())
    print("obj is cooked:", obj.states[object_states.Cooked].get_value())
    print("obj is burnt:", obj.states[object_states.Burnt].get_value())
    print("obj is soaked:", obj.states[object_states.Soaked].get_value())
    print("obj is toggledon:", obj.states[object_states.ToggledOn].get_value())
    print("obj textures:", obj.get_textures())

# Default.
report_states()

# Take a look at the scene.
for _ in range(2000):
    sim.step()

# Frozen.
obj.states[object_states.Temperature].set_value(-50)
report_states()

# Cooked.
obj.states[object_states.Temperature].set_value(100)
report_states()

# Burnt.
obj.states[object_states.Temperature].set_value(250)
report_states()

# Set back to default temperature
obj.states[object_states.Temperature].set_value(DEFAULT_TEMPERATURE)
obj.states[object_states.MaxTemperature].set_value(DEFAULT_TEMPERATURE)
report_states()

# Soaked.
obj.states[object_states.Soaked].set_value(True)
report_states()
obj.states[object_states.Soaked].set_value(False)
report_states()

# ToggledOn.
obj.states[object_states.ToggledOn].set_value(True)
report_states()
obj.states[object_states.ToggledOn].set_value(False)
report_states()