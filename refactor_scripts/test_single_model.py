from igibson import app, ig_dataset_path
from igibson.simulator_omni import Simulator
from igibson.scenes.empty_scene import EmptyScene
import omni
from omni.isaac.core.utils.stage import add_reference_to_stage
import xml.etree.ElementTree as ET
import numpy as np
import igibson.utils.transform_utils as T
import json
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.prims import create_prim, set_prim_property
from omni.kit.viewport import get_viewport_interface
from omni.isaac.dynamic_control import _dynamic_control
from pxr import Usd, UsdGeom, Gf, Sdf, UsdPhysics, PhysxSchema
from omni.isaac.core.utils.constants import AXES_INDICES
from omni.isaac.core.utils.prims import get_prim_at_path, get_prim_path, is_prim_path_valid
from omni.isaac.core.utils.carb import set_carb_setting
from omni.isaac.core.utils.stage import get_current_stage, get_stage_units, traverse_stage


##### SET THIS ######
obj_category = "bottom_cabinet"
obj_model = "46380"
name = "cabinet"
#### YOU DONT NEED TO TOUCH ANYTHING BELOW HERE IDEALLY :) #####

sim = Simulator()
scene = EmptyScene(floor_plane_visible=False)
sim.import_scene(scene)

model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
usd_path = f"{model_root_path}/usd/{obj_model}.usd"
metadata_path = f"{model_root_path}/misc/metadata.json"

# Load metadata to get bb info
with open(metadata_path, "r") as f:
    metadata = json.load(f)

default_bb = np.array(metadata["bbox_size"])

# Import model
add_reference_to_stage(
    usd_path=usd_path,
    prim_path=f"/World/{name}",
)

# IMPORTANT! AT LEAST ONE SIM STEP MUST OCCUR AFTER LOADING BEFORE REFERENCING DC ARTICULATIONS!!
sim.play()
dc = _dynamic_control.acquire_dynamic_control_interface()
art = dc.get_articulation(f"/World/{name}")
rb = dc.get_articulation_root_body(art)


for i in range(1000000):
    sim.step()
