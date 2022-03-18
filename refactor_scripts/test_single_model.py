from igibson import app, ig_dataset_path, Simulator
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
from omni.isaac.contact_sensor import _contact_sensor
from pxr import Usd, UsdGeom, Gf, Sdf, UsdPhysics, PhysxSchema
from omni.isaac.core.utils.constants import AXES_INDICES
from omni.isaac.core.utils.prims import get_prim_at_path, get_prim_path, is_prim_path_valid
from omni.isaac.core.utils.carb import set_carb_setting
from omni.isaac.core.utils.stage import get_current_stage, get_stage_units, traverse_stage

from igibson.prims.entity_prim import EntityPrim


##### SET THIS ######
obj_categories = ["bottom_cabinet", "fridge"]
obj_models = ["46380", "11712"]
names = ["cabinet", "fridge"]
#### YOU DONT NEED TO TOUCH ANYTHING BELOW HERE IDEALLY :) #####

sim = Simulator()
scene = EmptyScene(floor_plane_visible=True)
sim.import_scene(scene)

# sim = Simulator(stage_units_in_meters=1.0)
# sim.scene.add_ground_plane()

prims = []
for obj_category, obj_model, name in zip(obj_categories, obj_models, names):
    model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
    usd_path = f"{model_root_path}/usd/{obj_model}.usd"
    metadata_path = f"{model_root_path}/misc/metadata.json"

    # Load metadata to get bb info
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    default_bb = np.array(metadata["bbox_size"])
    prim_path = f"/World/{name}"

    # Import model
    prims.append(add_reference_to_stage(
        usd_path=usd_path,
        prim_path=prim_path,
    ))

cab_name = names[0]
fridge_name = names[1]

# Move fridge on top of cabinet
fridge = Articulation(prim_path=f"/World/{fridge_name}", name=fridge_name)
fridge.set_world_pose(position=np.array([0, 0, 2.0]))

# Try adding sensor
cs = _contact_sensor.acquire_contact_sensor_interface()

props = _contact_sensor.SensorProperties()
props.radius = -1.0  # Negative value implies full body sensor
props.minThreshold = 0  # Minimum force to detect
props.maxThreshold = 100000000  # Maximum force to detect
props.sensorPeriod = 0.0  # Zero means in sync with the simulation period

root_prim_path = f"/World/{cab_name}/base_link"
# c_handle = cs.add_sensor_on_body(root_prim_path, props)


sim.play()

# cab = EntityPrim(prim_path=prim_path, name=name)
# cab.initialize()

# IMPORTANT! AT LEAST ONE SIM STEP MUST OCCUR AFTER LOADING BEFORE REFERENCING DC ARTICULATIONS!!
sim.pause()
dc = _dynamic_control.acquire_dynamic_control_interface()
art = dc.get_articulation(f"/World/{cab_name}")
rb = dc.get_articulation_root_body(art)
input(f"art: {art}, rb: {rb}")
from omni.isaac.core.objects import DynamicCuboid


# cs.get_body_contact_raw_data(root_prim_path)


for i in range(1000000):
    sim.step()
    print(cs.get_sensor_readings(c_handle))
