import numpy as np
from igibson import object_states, ig_dataset_path
from igibson.objects.dataset_object import DatasetObject
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator_omni import Simulator

# Import scene.
sim = Simulator()
scene = EmptyScene()
sim.import_scene(scene)

# Import object.
obj_category = "bowl"
obj_model = "68_0"
name1 = "bowl1"
name2 = "bowl2"
name3 = "bowl3"

model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
usd_path = f"{model_root_path}/usd/{obj_model}.usd"

obj1 = DatasetObject(
    prim_path=f"/World/{name1}",
    usd_path=usd_path,
    category=obj_category,
    name=name1,
    scale=0.5,
    abilities={"heatable": {}},
)
obj2 = DatasetObject(
    prim_path=f"/World/{name2}", usd_path=usd_path, category=obj_category, name=name2, abilities={"heatable": {}},
)
obj3 = DatasetObject(
    prim_path=f"/World/{name3}",
    usd_path=usd_path,
    category=obj_category,
    name=name3,
    scale=2.0,
    abilities={"heatable": {}},
)

assert object_states.Heated in obj1.states
assert object_states.Heated in obj2.states
assert object_states.Heated in obj3.states

sim.import_object(obj1, auto_initialize=True)
sim.import_object(obj2, auto_initialize=True)
sim.import_object(obj3, auto_initialize=True)

sim.stop()
obj1.set_position(np.array([-0.6, 0, 0]))
obj2.set_position(np.array([0, 0, 0]))
obj3.set_position(np.array([0.8, 0, 0]))
sim.play()
sim.step()


def report_states(obj):
    # Make sure the state is updated.
    for _ in range(5):
        sim.step()
    print("=" * 20)
    print("object:", obj.name)
    print("temperature:", obj.states[object_states.Temperature].get_value())
    print("obj is heated:", obj.states[object_states.Heated].get_value())


# Default.
report_states(obj1)
report_states(obj2)
report_states(obj3)

for _ in range(2000):
    sim.step()

# Heated.
obj1.states[object_states.Temperature].set_value(50)
obj2.states[object_states.Temperature].set_value(50)
obj3.states[object_states.Temperature].set_value(50)
report_states(obj1)
report_states(obj2)
report_states(obj3)

# Take a look at the steam effect.
# After a while, objects will be below the Steam temperature threshold.
for _ in range(2000):
    sim.step()

# Objects are not heated anymore.
report_states(obj1)
report_states(obj2)
report_states(obj3)
