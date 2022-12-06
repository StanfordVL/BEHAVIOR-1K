from omnigibson import app, assets_path, og_dataset_path, Simulator
from omnigibson.scenes.empty_scene import EmptyScene
import omni
from omni.isaac.core.utils.stage import add_reference_to_stage
import xml.etree.ElementTree as ET
import numpy as np
import omnigibson.utils.transform_utils as T
import json
from omni.isaac.core.utils.prims import create_prim, set_prim_property
from omni.kit.viewport import get_viewport_interface
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.contact_sensor import _contact_sensor
from pxr import Usd, UsdGeom, Gf, Sdf, UsdPhysics, PhysxSchema
from omni.isaac.core.utils.constants import AXES_INDICES
from omni.isaac.core.utils.prims import get_prim_at_path, get_prim_path, is_prim_path_valid
from omni.isaac.core.utils.carb import set_carb_setting
from omni.isaac.core.utils.stage import get_current_stage, get_stage_units, traverse_stage

from omnigibson.prims.entity_prim import EntityPrim
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.systems.particle_system import DustSystem, StainSystem


# Macros
obj_category = "milk"
obj_model = "milk_000"
name = "milk"
system = None

# Create simulator and empty scene
sim = Simulator()
scene = EmptyScene(floor_plane_visible=True)
sim.import_scene(scene)

def steps(n_steps):
    global sim, system
    for i in range(n_steps):
        sim.step()
        system.update()

# Add milk carton object
milk = DatasetObject(
    prim_path=f"/World/{name}",
    usd_path=f"{og_dataset_path}/objects/{obj_category}/{obj_model}/usd/{obj_model}.usd",
    name=name,
)

# Import this object
sim.import_object(obj=milk)

# Move this object a bit upwards and disable gravity
milk.set_position_orientation(position=np.array([1.0, 1.0, 0.5]), orientation=np.array([0.0, 0.0, 0.707, 0.707]))

# Initialize dust system
system = StainSystem
system.initialize(simulator=sim)
dust = system.particle_object

sim.play()

# Generate particles on the cabinet
attachment_group = system.create_attachment_group(obj=milk)
system.generate_group_particles_on_object(group=attachment_group)

# start the sim so everything is initialized correctly
sim.step()
system.update()

milk.disable_gravity()

dc = _dynamic_control.acquire_dynamic_control_interface()


for i in range(1000000):
    sim.step()
    system.update()
