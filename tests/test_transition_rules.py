from omnigibson.macros import macros as m
from omnigibson.object_states import *
from omnigibson.systems import get_system, is_physical_particle_system, is_visual_particle_system
from omnigibson.utils.constants import PrimType
from omnigibson.utils.physx_utils import apply_force_at_pos, apply_torque
import omnigibson.utils.transform_utils as T
from omnigibson.utils.usd_utils import BoundingBoxAPI
from omnigibson.objects import DatasetObject
import omnigibson as og

from utils import og_test, get_random_pose, place_objA_on_objB_bbox, place_obj_on_floor_plane

import pytest
import numpy as np


@og_test
def test_blender_rule():
    blender = og.sim.scene.object_registry("name", "blender")

    blender.set_orientation([0, 0, 0, 1])
    place_obj_on_floor_plane(blender)
    og.sim.step()

    milk = get_system("whole_milk")
    chocolate_sauce = get_system("chocolate_sauce")
    milkshake = get_system("milkshake")
    milk.generate_particles(positions=np.array([[0, 0.02, 0.47]]))
    chocolate_sauce.generate_particles(positions=np.array([[0, -0.02, 0.47]]))

    ice_cream = DatasetObject(
        name="ice_cream",
        category="scoop_of_ice_cream",
        model="dodndj",
        bounding_box=[0.076, 0.077, 0.065],
    )
    og.sim.import_object(ice_cream)
    ice_cream.set_position([0, 0, 0.52])

    for i in range(5):
        og.sim.step()

    assert milkshake.n_particles == 0

    blender.states[ToggledOn].set_value(True)
    og.sim.step()

    assert milk.n_particles == 0
    assert chocolate_sauce.n_particles == 0
    assert milkshake.n_particles > 0

    # Remove objects and systems from recipe output
    milkshake.remove_all_particles()
    if og.sim.scene.object_registry("name", "ice_cream") is not None:
        og.sim.remove_object(obj=ice_cream)


@og_test
def test_cooking_rule():
    oven = og.sim.scene.object_registry("name", "oven")
    oven.keep_still()
    oven.set_orientation([0, 0, -0.707, 0.707])
    place_obj_on_floor_plane(oven)
    og.sim.step()

    sheet = DatasetObject(
        name="sheet",
        category="baking_sheet",
        model="yhurut",
        bounding_box=[0.520, 0.312, 0.0395],
    )

    og.sim.import_object(sheet)
    sheet.set_position_orientation([0.072, 0.004, 0.455], [0, 0, 0, 1])

    dough = DatasetObject(
        name="dough",
        category="sugar_cookie_dough",
        model="qewbbb",
        bounding_box=[0.200, 0.192, 0.0957],
    )
    og.sim.import_object(dough)
    dough.set_position_orientation([0.072, 0.004, 0.555], [0, 0, 0, 1])

    for i in range(10):
        og.sim.step()

    assert len(og.sim.scene.object_registry("category", "sugar_cookie", default_val=[])) == 0

    oven.states[ToggledOn].set_value(True)
    og.sim.step()
    og.sim.step()

    dough_exists = og.sim.scene.object_registry("name", "dough") is not None
    assert not dough_exists or not dough.states[OnTop].get_value(sheet)
    assert len(og.sim.scene.object_registry("category", "sugar_cookie")) > 0

    # Remove objects
    if dough_exists:
        og.sim.remove_object(dough)
    og.sim.remove_object(sheet)
