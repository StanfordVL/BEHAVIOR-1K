from igibson import object_states, ig_dataset_path
from igibson.objects.dataset_object import DatasetObject
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator_omni import Simulator

# Import scene.
sim = Simulator()
scene = EmptyScene()
sim.import_scene(scene)

# Import object.
obj_category = "apple"
obj_model = "00_0"
name = "apple"

model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
usd_path = f"{model_root_path}/usd/{obj_model}.usd"

obj = DatasetObject(
    prim_path=f"/World/{name}",
    usd_path=usd_path,
    category=obj_category,
    name=name,
    abilities={"heatable": {}},
)

assert object_states.Heated in obj.states

sim.import_object(obj, auto_initialize=True)
sim.play()
sim.step()

for _ in range(2000):
    sim.step()

def report_states():
    # Make sure the state is updated.
    for _ in range(5):
        sim.step()
    print("=" * 20)
    print("temperature:", obj.states[object_states.Temperature].get_value())
    print("obj is heated:", obj.states[object_states.Heated].get_value())

# Default.
report_states()

# Heated.
obj.states[object_states.Temperature].set_value(60)
report_states()

# Take a look at the steam effect.
for _ in range(1000):
    sim.step()

# Not heated.
obj.states[object_states.Temperature].set_value(20)
report_states()

# Steam should disappear.
for _ in range(1000):
    sim.step()