from igibson import object_states, ig_dataset_path
from igibson.objects.dataset_object import DatasetObject
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator_omni import Simulator

# Import scene.
sim = Simulator()
scene = EmptyScene()
sim.import_scene(scene)

# Import object.
obj_category = "sink"
obj_model = "sink_1"
name = "sink"

model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
usd_path = f"{model_root_path}/usd/{obj_model}.usd"

obj = DatasetObject(
    prim_path=f"/World/{name}",
    usd_path=usd_path,
    category=obj_category,
    name=name,
    abilities={"freezable": {}, "cookable": {}, "burnable": {}},
)

assert object_states.Frozen in obj.states
assert object_states.Cooked in obj.states
assert object_states.Burnt in obj.states

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
    print("obj textures:", obj.get_textures())


# Default.
report_states()

# Frozen.
obj.states[object_states.Temperature].set_value(-50)
report_states()

# Take a look at the scene.
for _ in range(2000):
    sim.step()
    
# Cooked.
obj.states[object_states.Temperature].set_value(100)
report_states()

# Burnt.
obj.states[object_states.Temperature].set_value(250)
report_states()
