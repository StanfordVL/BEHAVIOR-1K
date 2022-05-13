import os

import numpy as np

from igibson import ig_dataset_path
from igibson.robots.turtlebot import Turtlebot
from igibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from igibson.simulator_omni import Simulator

SCENE_ID = "Rs_int"
USD_TEMPLATE_FILE = f"{ig_dataset_path}/scenes/{SCENE_ID}/urdf/{SCENE_ID}_best_template.usd"
TEST_OUT_PATH = "/home/alanlou/svl/iGibson3/igibson/out"  # Define output directory here.

#### SAVE SIMULATION ENV #####
sim = Simulator()

# Create a scene.
scene = InteractiveTraversableScene(
    scene_model=SCENE_ID, usd_path=USD_TEMPLATE_FILE, load_object_categories=["bed", "bottom_cabinet"]
)

# Import the scene.
sim.import_scene(scene=scene)

# Report objects loaded.
for reg in sim.scene.registry.objects:
    print("=====")
    print(reg.name)
    for obj in reg.objects:
        print(f"Registry: {reg.name}, obj: {obj.name}")

# Take a look at the scene.
for _ in range(2000):
    sim.step()

# Canonical position can only be set when sim is stopped.
sim.stop()

# Move 2 objects.
o = sim.scene.object_registry("name", "bottom_cabinet_13")
o.set_position(np.array([0.2, -1.49805, 0.38631]))
cab_pos_1 = o.get_position()
print(f"{o.name} position: {cab_pos_1}")

o = sim.scene.object_registry("name", "bottom_cabinet_41")
o.set_position(np.array([0, 3.13717675, 0.46447614]))
cab_pos_2 = o.get_position()
print(f"{o.name} position: {cab_pos_2}")

# Import a robot.
robot = Turtlebot(prim_path=f"/World/robot", name="robot", obs_modalities=["proprio", "rgb", "scan", "occupancy_grid"])
sim.import_object(obj=robot)
sim.play()
sim.step()
robot.apply_action(np.array([1.0, 0]))

# Take some steps.
for _ in range(100):
    sim.step()

print(f"{robot.name} position: {robot.get_position()}")
print(f"{robot.name} linear_velocity: {robot.get_linear_velocity()}")
print(f"{robot.name} angular_velocity: {robot.get_angular_velocity()}")

# Report loaded prims in the world.
world_prim = sim.world_prim
for prim in world_prim.GetChildren():
    name = prim.GetName()
    # Only process prims that are an Xform.
    if prim.GetPrimTypeInfo().GetTypeName() == "Xform":
        name = prim.GetName()
        print(name)

# Take a look at the registries.
print([o.name for o in sim.scene.object_registry.objects])
print([o.name for o in sim.scene.robot_registry.objects])

# Save the simulation environment.
json_path = os.path.join(TEST_OUT_PATH, "saved_scene.json")
usd_path = os.path.join(TEST_OUT_PATH, "saved_stage.usd")
sim.save(json_path, usd_path)

#### LOAD SIMULATION ENV #####
## Optional: restart a session and skip the "SAVE SIMULATION ENV" section

# NOTE: Skip this line if did not restart the session.
sim = Simulator()

# Restore the saved sim environment.
json_path = os.path.join(TEST_OUT_PATH, "saved_scene.json")
usd_path = os.path.join(TEST_OUT_PATH, "saved_stage.usd")
sim.restore(json_path, usd_path)

# Take a look at the registries.
print([o.name for o in sim.scene.object_registry.objects])
print([o.name for o in sim.scene.robot_registry.objects])

# Check object positions.
o = sim.scene.object_registry("name", "bottom_cabinet_13")
print(f"{o.name} position: {o.get_position()}")

o = sim.scene.object_registry("name", "bottom_cabinet_41")
print(f"{o.name} position: {o.get_position()}")

# Check robot position and velocity.
robot = sim.scene.robot_registry("name", "robot")
print(f"{robot.name} position: {robot.get_position()}")
print(f"{robot.name} linear_velocity: {robot.get_linear_velocity()}")
print(f"{robot.name} angular_velocity: {robot.get_angular_velocity()}")

# Report objects loaded.
for reg in sim.scene.registry.objects:
    print("=====")
    print(reg.name)
    for obj in reg.objects:
        print(f"Registry: {reg.name}, obj: {obj.name}")

# Take a step.
sim.play()
sim.step()

# Take a look at the scene.
for _ in range(2000):
    sim.step()
