import matplotlib.pyplot as plt

from igibson import app, ig_dataset_path, Simulator
from igibson.utils.usd_utils import CollisionAPI
from igibson.scenes.empty_scene import EmptyScene
from igibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from igibson.robots.turtlebot import Turtlebot
from omni.isaac.core.utils.stage import add_reference_to_stage
import xml.etree.ElementTree as ET
import numpy as np
import igibson.utils.transform_utils as T
import json
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.prims import create_prim, set_prim_property, get_prim_at_path, get_prim_path, is_prim_path_valid, get_prim_children


##### SET THIS ######
SCENE_ID = "Rs_int"
USD_TEMPLATE_FILE = f"{ig_dataset_path}/scenes/Rs_int/urdf/Rs_int_best_template.usd"
#### YOU DONT NEED TO TOUCH ANYTHING BELOW HERE IDEALLY :) #####


sim = Simulator()

# sim.load_stage("/cvgl2/u/jdwong/tmp/test_rs_int.usd")

# Load scene
scene = InteractiveTraversableScene(
    scene_model=SCENE_ID,
    usd_path=USD_TEMPLATE_FILE,
)

# Import scene
sim.import_scene(scene=scene)

cab = sim.scene.object_registry("name", "bottom_cabinet_0")

for reg in sim.scene.registry.objects:
    for obj in reg.objects:
        print(f"Registry: {reg.name}, obj: {obj.name}")

sim.step()
sim.stop()

# Import robot
robot = Turtlebot(prim_path=f"/World/robot", name="robot", obs_modalities=["proprio", "rgb", "scan", "occupancy_grid"])
sim.import_object(obj=robot)

sim.play()

sim.step()
obs = robot.get_obs()
rgb = obs["robot:eyes_Camera_sensor_rgb"]
# plt.imsave("/cvgl2/u/jdwong/tmp/test6.png", rgb)


# toggle = 1
for i in range(1000000):
    robot.apply_action(np.array([1.0, 0]))
    sim.step()
    # print(f"collision groups: {CollisionAPI.ACTIVE_COLLISION_GROUPS}")
    # if i % 10 == 0:
    #     print("TOGGLING")
    #     if toggle:
    #         cab.root_link._remove_contact_sensor()
    #     else:
    #         cab.root_link._create_contact_sensor()
    #     toggle = 1 - toggle
    # print(f"Contact: {[c.body0 for c in cab.root_link.contact_list()]}")

app.close()
