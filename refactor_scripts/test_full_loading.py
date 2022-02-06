from igibson import app, ig_dataset_path
from igibson.simulator_omni import Simulator
from igibson.utils.usd_utils import CollisionAPI
from igibson.scenes.empty_scene import EmptyScene
from igibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from omni.isaac.core.utils.stage import add_reference_to_stage
import xml.etree.ElementTree as ET
import numpy as np
import igibson.utils.transform_utils as T
import json
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.prims import create_prim, set_prim_property, get_prim_at_path, get_prim_path, is_prim_path_valid, get_prim_children
from omni.kit.viewport import get_viewport_interface


##### SET THIS ######
SCENE_ID = "Rs_int"
USD_TEMPLATE_FILE = f"{ig_dataset_path}/scenes/Rs_int/urdf/Rs_int_best_template.usd"
#### YOU DONT NEED TO TOUCH ANYTHING BELOW HERE IDEALLY :) #####


sim = Simulator()

# Load scene
scene = InteractiveTraversableScene(
    scene_id=SCENE_ID,
    usd_path=USD_TEMPLATE_FILE,
)

# Import scene
sim.import_scene(scene=scene)

for i in range(1000000):
    sim.step()

app.close()
