from igibson import app, assets_path, ig_dataset_path, Simulator
from igibson.scenes.empty_scene import EmptyScene
import omni
from omni.isaac.core.utils.stage import add_reference_to_stage
import xml.etree.ElementTree as ET
import numpy as np
import igibson.utils.transform_utils as T
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

from igibson.prims.entity_prim import EntityPrim
from igibson.objects.dataset_object import DatasetObject
from igibson.systems.particle_system import DustSystem, StainSystem


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

# Add cabinet object
cab = DatasetObject(
    prim_path=f"/World/{name}",
    usd_path=f"{ig_dataset_path}/objects/{obj_category}/{obj_model}/usd/{obj_model}.usd",
    name=name,
)

# Import this object
sim.import_object(obj=cab)

# Move this object a bit upwards and disable gravity
cab.set_position_orientation(position=np.array([1.0, 1.0, 0.5]), orientation=np.array([0.0, 0.0, 0.707, 0.707]))

# Initialize dust system
system = StainSystem
system.initialize(simulator=sim)
dust = system.particle_object

sim.play()

# Generate particles on the cabinet
# system.generate_particles_on_object(obj=cab)
attachment_group = system.create_attachment_group(obj=cab)
system.generate_group_particles(group=attachment_group, n_particles=100)
# dust_copy = system._attachment_groups[name]["dust_template_copy1"]

# start the sim so everything is initialized correctly
sim.step()
system.update()

cab.disable_gravity()

dc = _dynamic_control.acquire_dynamic_control_interface()





for i in range(1000000):
    sim.step()
    system.update()
